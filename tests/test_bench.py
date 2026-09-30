"""Benchmark harness logic: no dataset, no container runtime, no API calls."""

import ast
import asyncio
import inspect
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import polars as pl
import pytest
from google.genai import errors as genai_errors
from google.genai import types

import bench.runner as runner
from app.chat_loop import run_chat_turn
from app.config import CHAT_MODEL_IDS, LITE_MODEL, MAX_TOOL_ROUNDS
from app.sandbox.pool import SandboxBusy
from bench import provenance
from bench.__main__ import build_parser
from bench.grading import is_executable, stale_reference
from bench.ground_truth import parquet_identity, reference_sha
from bench.io import atomic_write_json, move_aside, read_trace
from bench.questions import BY_ID, QUESTIONS
from bench.report import build_report
from bench.usage import (
    RecordingClient,
    aggregate_usage,
    describe_response,
    merge_usage,
    terminal_model_error,
)

CHAT_MODEL = sorted(CHAT_MODEL_IDS)[0]


# ---------------------------------------------------------------------------
# parquet identity
# ---------------------------------------------------------------------------


def _write_parts(d: Path, frames: list[pl.DataFrame]) -> None:
    d.mkdir(parents=True)
    for i, df in enumerate(frames):
        df.write_parquet(d / f"part.{i}.parquet")


def _frames():
    return [
        pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        pl.DataFrame({"a": [4, 5], "b": ["u", "v"]}),
    ]


def test_parquet_identity_directory_of_parts(tmp_path):
    src = tmp_path / "one" / "data.parquet"
    _write_parts(src, _frames())
    ident = parquet_identity(src)
    assert len(ident) == 12

    # No absolute path, no mtime: a copy elsewhere and a touch keep it.
    copy = tmp_path / "two" / "elsewhere.parquet"
    shutil.copytree(src, copy)
    assert parquet_identity(copy) == ident
    os.utime(copy / "part.0.parquet", (1, 1))
    assert parquet_identity(copy) == ident

    # A symlink to the directory resolves to the same content.
    link = tmp_path / "link.parquet"
    link.symlink_to(src)
    assert parquet_identity(link) == ident

    # Renaming a part or rewriting data changes it.
    os.rename(copy / "part.1.parquet", copy / "part.9.parquet")
    assert parquet_identity(copy) != ident
    pl.DataFrame({"a": [4, 6], "b": ["u", "v"]}).write_parquet(src / "part.1.parquet")
    assert parquet_identity(src) != ident


def test_parquet_identity_single_file(tmp_path):
    f = tmp_path / "a.parquet"
    _frames()[0].write_parquet(f)
    ident = parquet_identity(f)
    other = tmp_path / "sub" / "b.parquet"
    other.parent.mkdir()
    shutil.copy(f, other)
    assert parquet_identity(other) == ident  # content, not name/path
    _frames()[1].write_parquet(f)
    assert parquet_identity(f) != ident


def test_parquet_identity_rejects_non_parquet(tmp_path):
    from bench.ground_truth import GroundTruthError

    bad = tmp_path / "x.parquet"
    bad.write_bytes(b"not a parquet file at all")
    with pytest.raises(GroundTruthError):
        parquet_identity(bad)


# ---------------------------------------------------------------------------
# error buckets / backoff
# ---------------------------------------------------------------------------


def _api_error(code: int, headers: dict | None = None):
    err = genai_errors.APIError(
        code, {"error": {"code": code, "message": "m", "status": "S"}}
    )
    if headers is not None:
        err.response = SimpleNamespace(headers=httpx.Headers(headers))
    return err


@pytest.mark.parametrize(
    ("exc", "ok_calls", "bucket"),
    [
        (_api_error(429), 0, "retryable"),
        (_api_error(503), 3, "retryable"),
        (_api_error(408), 0, "retryable"),
        (httpx.ConnectError("boom"), 0, "retryable"),
        (httpx.ReadTimeout("slow"), 2, "retryable"),
        (_api_error(401), 0, "fatal"),
        (_api_error(403), 5, "fatal"),
        (_api_error(404), 0, "fatal"),
        (_api_error(400), 0, "fatal"),  # first call: our request, not the model
        (_api_error(400), 2, "model"),  # after model output: the model's doing
        (SandboxBusy("no worker"), 0, "fatal"),
        (_api_error(418), 0, "infra"),
        (RuntimeError("chat turn ended without DoneEvent"), 1, "infra"),
    ],
)
def test_classify_error(exc, ok_calls, bucket):
    assert runner.classify_error(exc, ok_calls) == bucket


def test_backoff_honors_retry_after_and_rate_limits():
    assert runner.backoff_s(_api_error(503), 0) == runner.RETRY_BACKOFF_S[0]
    assert runner.backoff_s(_api_error(429), 0) == runner.RATE_LIMIT_BACKOFF_S[0]
    assert runner.backoff_s(_api_error(503, {"Retry-After": "95"}), 0) == 95
    # Never shorter than the base backoff, never longer than the cap.
    assert runner.backoff_s(_api_error(503, {"Retry-After": "1"}), 1) == 30
    assert (
        runner.backoff_s(_api_error(429, {"Retry-After": "99999"}), 0)
        == runner.MAX_BACKOFF_S
    )


# ---------------------------------------------------------------------------
# io + resume rules
# ---------------------------------------------------------------------------


def test_atomic_write_keeps_old_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "q" / "trial_00.json"
    atomic_write_json(path, {"v": 1})
    assert json.loads(path.read_text()) == {"v": 1}
    assert not list(path.parent.glob("*.tmp"))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}


def test_plan_trial_resume_rules(tmp_path):
    p = tmp_path / "L01" / "trial_00.json"
    assert runner.plan_trial(p) == ("run", None)
    atomic_write_json(p, {"question_id": "L01", "infra_error": None})
    assert runner.plan_trial(p) == ("skip", None)
    atomic_write_json(p, {"question_id": "L01", "model_error": "x"})
    assert runner.plan_trial(p) == ("skip", None)  # model-caused: a real result
    atomic_write_json(p, {"question_id": "L01", "infra_error": "503"})
    assert runner.plan_trial(p) == ("retry", "infra")
    p.write_text('{"truncated": ')
    assert runner.plan_trial(p) == ("retry", "corrupt")
    p.write_text("[1, 2]")
    assert runner.plan_trial(p) == ("retry", "corrupt")


def test_move_aside_leaves_trace_glob(tmp_path):
    p = tmp_path / "L01" / "trial_00.json"
    atomic_write_json(p, {"infra_error": "x"})
    dest = move_aside(p, "infra")
    assert not p.exists() and dest.exists()
    assert list(tmp_path.glob("*/trial_*.json")) == []
    # A second move of a same-named file never overwrites the first.
    atomic_write_json(p, {"infra_error": "y"})
    dest2 = move_aside(p, "infra")
    assert dest2 != dest and dest.exists() and dest2.exists()


# ---------------------------------------------------------------------------
# fingerprint drift
# ---------------------------------------------------------------------------


def _meta(**over):
    base = {key: f"v-{key}" for key in provenance.DRIFT_KEYS if key != "question_shas"}
    base.update(
        question_shas={"L01": "aaa", "L02": "bbb"},
        trials=3,
        argv=["run"],
        started_at="t0",
    )
    base.update(over)
    return base


def test_open_run_dir_refuses_drift(tmp_path):
    d = tmp_path / "run"
    runner.open_run_dir(d, _meta(), allow_drift=False)
    # Same fingerprint, a new question and more trials: fine, merged.
    meta = runner.open_run_dir(
        d,
        _meta(question_shas={"L01": "aaa", "A01": "ccc"}, trials=5, started_at="t1"),
        allow_drift=False,
    )
    assert meta["question_shas"] == {"L01": "aaa", "L02": "bbb", "A01": "ccc"}
    assert meta["trials"] == 5
    assert len(meta["resumes"]) == 1

    with pytest.raises(runner.BenchAbort, match="prompt_sha"):
        runner.open_run_dir(d, _meta(prompt_sha="changed"), allow_drift=False)
    with pytest.raises(runner.BenchAbort, match="question L01"):
        runner.open_run_dir(d, _meta(question_shas={"L01": "zzz"}), allow_drift=False)
    meta = runner.open_run_dir(d, _meta(git_sha="other"), allow_drift=True)
    assert any("git_sha" in x for x in meta["resumes"][-1]["drift"])


@pytest.mark.parametrize("key", provenance.IDENTITY_KEYS)
def test_allow_drift_never_waives_model_or_dataset(tmp_path, key):
    d = tmp_path / "run"
    runner.open_run_dir(d, _meta(), allow_drift=False)
    with pytest.raises(runner.BenchAbort, match=key):
        runner.open_run_dir(d, _meta(**{key: "other"}), allow_drift=True)
    meta = json.loads((d / "run_meta.json").read_text())
    assert meta[key] == f"v-{key}" and "resumes" not in meta


def test_legacy_run_meta_counts_as_drift():
    legacy = {"model": "m", "parquet_identity": "p", "trials": 1}
    diffs = provenance.check_drift(legacy, _meta(model="m", parquet_identity="p"))
    assert any("missing from run_meta" in d for d in diffs)


def test_question_sha_tracks_wording_not_reference():
    q = BY_ID["L03"]
    changed = type(q)(**{**q.__dict__, "text": q.text + " Please."})
    assert provenance.question_sha(changed) != provenance.question_sha(q)
    recoded = type(q)(**{**q.__dict__, "reference_code": q.reference_code + "\n"})
    assert provenance.question_sha(recoded) == provenance.question_sha(q)
    assert reference_sha(recoded) != reference_sha(q)


def test_prompt_sha_covers_prompt_and_tools():
    base = provenance.prompt_sha("sys", runner.BENCH_TOOLS)
    assert provenance.prompt_sha("sys ", runner.BENCH_TOOLS) != base
    fewer = types.Tool(
        function_declarations=runner.BENCH_TOOLS.function_declarations[:1]
    )
    assert provenance.prompt_sha("sys", fewer) != base
    assert provenance.prompt_sha("sys", runner.BENCH_TOOLS) == base


def test_sandbox_settings_from_env_mirror_production_defaults():
    s = provenance.sandbox_settings_from_env({})
    assert s == {
        "size": 2,
        "max_total": 3,
        "image": "kindling-worker:latest",
        "memory": "110g",
        "cpus": None,
        "pids": 8192,
        "max_threads": None,
        "worker_parquet_path": None,
    }
    s = provenance.sandbox_settings_from_env(
        {"KINDLING_SANDBOX_CPUS": "8", "KINDLING_SANDBOX_PIDS": "none"}
    )
    assert s["cpus"] == "8" and s["max_threads"] == 8 and s["pids"] is None


# ---------------------------------------------------------------------------
# models / interleaving / max_rounds parity
# ---------------------------------------------------------------------------


def test_parse_models_validates_against_config():
    args = SimpleNamespace(models=None, model=CHAT_MODEL)
    assert runner.parse_models(args) == [CHAT_MODEL]
    both = ",".join(sorted(CHAT_MODEL_IDS))
    args = SimpleNamespace(models=both, model="ignored")
    assert runner.parse_models(args) == sorted(CHAT_MODEL_IDS)
    with pytest.raises(runner.BenchAbort, match="unknown model"):
        runner.parse_models(SimpleNamespace(models="gemini-nope", model=None))
    with pytest.raises(runner.BenchAbort, match="duplicate"):
        runner.parse_models(
            SimpleNamespace(models=f"{CHAT_MODEL},{CHAT_MODEL}", model=None)
        )


def test_trial_cells_interleave_seeded():
    qs = QUESTIONS[:4]
    models = ["a", "b"]
    cells = runner.trial_cells(qs, 3, models, seed=7)
    keys = [(q.id, t, m) for q, t, m in cells]
    assert len(keys) == len(set(keys)) == 4 * 3 * 2
    assert keys == [
        (q.id, t, m) for q, t, m in runner.trial_cells(qs, 3, models, seed=7)
    ]
    assert keys != [
        (q.id, t, m) for q, t, m in runner.trial_cells(qs, 3, models, seed=8)
    ]
    single = runner.trial_cells(qs, 2, ["a"], seed=7)
    assert [(q.id, t) for q, t, _ in single] == [
        (q.id, t) for q in qs for t in range(2)
    ]
    assert runner.model_run_dirs(Path("r"), ["a"]) == {"a": Path("r")}
    assert runner.model_run_dirs(Path("r"), models) == {
        "a": Path("r/a"),
        "b": Path("r/b"),
    }


def test_max_rounds_parity():
    sig = inspect.signature(run_chat_turn)
    assert sig.parameters["max_rounds"].default == MAX_TOOL_ROUNDS
    assert build_parser().parse_args(["run"]).max_rounds == MAX_TOOL_ROUNDS
    assert (
        inspect.signature(runner.run_trial).parameters["max_rounds"].default
        == MAX_TOOL_ROUNDS
    )
    # tests/evals/conftest.py run_turn's inner _run defaults to the constant.
    src = (Path(__file__).parent / "evals" / "conftest.py").read_text()
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run"
    )
    names = [a.arg for a in fn.args.kwonlyargs]
    default = fn.args.kw_defaults[names.index("max_rounds")]
    assert isinstance(default, ast.Name) and default.id == "MAX_TOOL_ROUNDS"


def test_default_trials_is_three():
    assert build_parser().parse_args(["run"]).trials == 3


# ---------------------------------------------------------------------------
# usage capture
# ---------------------------------------------------------------------------


def _response(parts, *, finish="STOP", usage=None, version="served-1", content=True):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts) if content else None,
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**(usage or {})),
        model_version=version,
    )


def test_describe_and_aggregate_usage():
    r = _response(
        [types.Part(text="hi")],
        usage={
            "prompt_token_count": 100,
            "cached_content_token_count": 40,
            "candidates_token_count": 10,
            "thoughts_token_count": 5,
            "total_token_count": 115,
        },
    )
    d = describe_response(r)
    assert d["model_version"] == "served-1"
    assert d["finish_reason"] == "STOP"
    assert d["has_text"] and d["usage"]["thoughts_token_count"] == 5
    calls = [
        {"model": "m", **d},
        {"model": "m", **d},
        {"model": "lite", **describe_response(_response([types.Part(text="{}")]))},
        {"model": "m", "error": "503"},  # failed call: no usage
    ]
    agg = aggregate_usage(calls)
    assert agg["m"]["calls"] == 2
    assert agg["m"]["prompt_token_count"] == 200
    assert agg["m"]["cached_content_token_count"] == 80
    assert agg["lite"]["total_token_count"] == 0
    merged = merge_usage(agg, {"m": {"calls": 1, "total_token_count": 5}})
    assert merged["m"]["calls"] == 3 and merged["m"]["total_token_count"] == 235


def test_terminal_model_error():
    ok = {"purpose": "chat", **describe_response(_response([types.Part(text="a")]))}
    malformed = {
        "purpose": "chat",
        **describe_response(
            _response([], finish="MALFORMED_FUNCTION_CALL", content=False)
        ),
    }
    empty_max = {
        "purpose": "chat",
        **describe_response(_response([], finish="MAX_TOKENS")),
    }
    judge = {**malformed, "purpose": "judge"}
    assert terminal_model_error([ok]) is None
    assert "MALFORMED" in terminal_model_error([ok, malformed])
    assert terminal_model_error([malformed, ok]) is None  # recovered
    assert terminal_model_error([ok, judge]) is None  # judge calls don't count
    assert "MAX_TOKENS" in terminal_model_error([empty_max])
    assert terminal_model_error([]) is None


# ---------------------------------------------------------------------------
# run_trial end to end against fakes
# ---------------------------------------------------------------------------


class _FakeModels:
    """Scripted chat responses; code-judge (LITE_MODEL) calls always allow."""

    def __init__(self, script):
        self.script = list(script)
        self.chat_calls = 0

    def generate_content(self, *, model, contents, config):
        if model == LITE_MODEL:
            return _response([types.Part(text='{"allow": true, "reason": "ok"}')])
        self.chat_calls += 1
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step()
        return step


class _FakeInner:
    def __init__(self, script):
        self.models = _FakeModels(script)


class _FakeSession:
    def __init__(self, delay=0.0):
        self.delay = delay

    async def run_query(self, code):
        if self.delay:
            await asyncio.sleep(self.delay)
        if "boom" in code:
            return {"error": "NameError: boom"}
        return {"data": [{"n": 1}], "plots": ["/plots/plot-001.png?t=1"]}


class _FakePool:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.released = 0

    async def acquire_session(self):
        return _FakeSession(self.delay)

    def release_session(self, s):
        self.released += 1


def _call(code):
    return _response(
        [
            types.Part(
                function_call=types.FunctionCall(name="run_query", args={"code": code})
            )
        ],
        usage={"prompt_token_count": 50, "total_token_count": 60},
    )


def _text(t="The answer is 1."):
    return _response(
        [types.Part(text=t)],
        usage={"prompt_token_count": 70, "total_token_count": 80},
    )


async def _trial(tmp_path, script, *, pool=None, retries=2, timeout=60):
    client = RecordingClient(_FakeInner(script))
    path = tmp_path / "L03" / "trial_00.json"
    with runner.bench_tool_dispatch():
        trace = await runner.run_trial(
            client,
            CHAT_MODEL,
            pool or _FakePool(),
            BY_ID["L03"],
            27794,
            0,
            path,
            Path("x.parquet"),
            "ident",
            prompt_sha="p",
            question_sha="q",
            retries=retries,
            turn_timeout_s=timeout,
        )
    assert json.loads(path.read_text())["question_id"] == "L03"
    return trace


async def test_run_trial_records_usage_rejections_and_plots(tmp_path):
    web = _response(
        [types.Part(function_call=types.FunctionCall(name="web_search", args={}))]
    )
    missing = _response(
        [types.Part(function_call=types.FunctionCall(name="run_query", args={}))]
    )
    trace = await _trial(
        tmp_path, [_call("result = boom"), web, missing, _call("result = 1"), _text()]
    )
    assert trace["error_bucket"] is None and trace["model_error"] is None
    assert trace["model_version"] == "served-1"
    assert trace["finish_reason"] == "STOP"
    assert trace["reference_sha"] == reference_sha(BY_ID["L03"])
    assert trace["max_rounds"] == MAX_TOOL_ROUNDS
    assert [r["source"] for r in trace["rejections"]] == [
        "sandbox",
        "missing_code_arg",
    ]
    assert [r["round"] for r in trace["rejections"]] == [0, 2]
    assert trace["web_search_calls"] == [{"round": 1, "args": {}}]
    assert trace["missing_code_args"] == 1
    assert trace["query_records"][1]["plots"] == ["/plots/plot-001.png?t=1"]
    usage = trace["usage"]
    assert usage[CHAT_MODEL]["calls"] == 5
    assert usage[LITE_MODEL]["calls"] == 2  # code-judge ran on both real queries
    assert trace["turn_latency_s"] is not None
    assert is_executable(trace) == (True, "ok")


async def test_run_trial_malformed_final_is_model_error(tmp_path):
    malformed = _response([], finish="MALFORMED_FUNCTION_CALL", content=False)
    trace = await _trial(tmp_path, [_call("result = 1"), malformed])
    assert trace["error_bucket"] == "model"
    assert trace["infra_error"] is None
    assert "MALFORMED_FUNCTION_CALL" in trace["model_error"]
    assert is_executable(trace) == (False, "model_malformed")


async def test_run_trial_retries_retryable_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "backoff_s", lambda e, a: 0)
    trace = await _trial(tmp_path, [_api_error(503), _call("result = 1"), _text()])
    assert trace["retries_used"] == 1
    assert trace["attempts"][0]["bucket"] == "retryable"
    assert trace["infra_error"] is None
    assert [c["attempt"] for c in trace["calls"] if c["purpose"] == "chat"] == [
        0,
        1,
        1,
    ]


async def test_run_trial_exhausted_retries_is_infra(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "backoff_s", lambda e, a: 0)
    trace = await _trial(tmp_path, [_api_error(503)] * 2, retries=1)
    assert trace["error_bucket"] == "retryable"
    assert "503" in trace["infra_error"]
    assert runner.plan_trial(tmp_path / "L03" / "trial_00.json") == ("retry", "infra")


async def test_run_trial_fatal_and_model_400(tmp_path):
    trace = await _trial(tmp_path, [_api_error(401)])
    assert trace["error_bucket"] == "fatal" and trace["infra_error"]
    trace = await _trial(tmp_path, [_call("result = 1"), _api_error(400)])
    assert trace["error_bucket"] == "model" and trace["infra_error"] is None
    assert len(trace["query_records"]) == 1  # partial records kept


async def test_run_trial_turn_timeout_is_model_error(tmp_path):
    pool = _FakePool(delay=5)
    trace = await _trial(
        tmp_path, [_call("result = 1"), _text()], pool=pool, timeout=0.3
    )
    assert trace["error_bucket"] == "model"
    assert "timeout" in trace["model_error"]
    assert pool.released == 1


def test_bench_lock_is_exclusive(tmp_path):
    lock = tmp_path / "bench.lock"
    with runner.bench_lock(lock):
        with pytest.raises(runner.BenchAbort, match="another bench"):
            with runner.bench_lock(lock):
                pass
    with runner.bench_lock(lock):
        pass


# ---------------------------------------------------------------------------
# question wording invariants (grading itself: tests/test_bench_grading.py)
# ---------------------------------------------------------------------------


def test_exact_count_questions_are_exact():
    from bench.answers import ANSWERS

    for qid in ("L03", "L04", "L07", "L08", "M02"):
        q = BY_ID[qid]
        assert q.tolerance_abs == 0
        assert "Give the exact count." in q.text
        assert ANSWERS[qid].integer
    assert BY_ID["A03"].tolerance_abs == 0 and ANSWERS["A03"].integer


def test_incident_type_scope_is_stated():
    for qid in ("T01", "T02", "T03", "T04", "T05"):
        q = BY_ID[qid]
        assert "Incid_Type = 1" in q.text
        assert 'pl.col("Incid_Type") == 1' in q.reference_code
    for qid in ("A01", "M04"):
        assert "Count every incident type." in BY_ID[qid].text
    # M02 uses the review's rewrite, which states the same scope inline.
    assert "regardless of bs value or Incid_Type" in BY_ID["M02"].text


def test_stale_reference_refusal():
    q = BY_ID["L03"]
    good = {"question_id": "L03", "reference_sha": reference_sha(q)}
    assert stale_reference(good) is None
    assert "no reference_sha" in stale_reference({"question_id": "L03"})
    assert "!=" in stale_reference({"question_id": "L03", "reference_sha": "old"})


def test_grade_run_refuses_stale_before_judging(tmp_path):
    from bench.grading import StaleTraceError, grade_run

    atomic_write_json(
        tmp_path / "L03" / "trial_00.json",
        {"question_id": "L03", "trial": 0, "reference_sha": "old", "text": "x"},
    )
    with pytest.raises(StaleTraceError, match="L03"):
        grade_run(tmp_path, client=None)


# ---------------------------------------------------------------------------
# report robustness
# ---------------------------------------------------------------------------


def _graded(qid, trial, accurate=True, **extra):
    return {
        "question_id": qid,
        "trial": trial,
        "category": BY_ID[qid].category,
        "infra_error": None,
        "executable": True,
        "executability_reason": "ok",
        "accurate": accurate,
        "turn_latency_s": 3.0,
        "text": "t",
        "usage": {CHAT_MODEL: {"calls": 2, "total_token_count": 1000}},
        **extra,
    }


def test_report_survives_ungraded_corrupt_and_missing(tmp_path):
    atomic_write_json(
        tmp_path / "run_meta.json",
        {
            "model": CHAT_MODEL,
            "trials": 2,
            "question_shas": {"L03": "a", "L04": "b"},
        },
    )
    atomic_write_json(tmp_path / "L03" / "trial_00.json", _graded("L03", 0))
    ungraded = _graded("L03", 1)
    for k in ("accurate", "executable", "executability_reason"):
        ungraded.pop(k)
    atomic_write_json(tmp_path / "L03" / "trial_01.json", ungraded)
    (tmp_path / "L04").mkdir()
    (tmp_path / "L04" / "trial_00.json").write_text("{not json")
    atomic_write_json(
        tmp_path / "L04" / "trial_01.json",
        {**_graded("L04", 1), "infra_error": "503", "web_search_calls": [{}]},
    )

    report = build_report(tmp_path)
    assert "1 ungraded" in report
    assert "unreadable file(s) skipped" in report and "L04/trial_00.json" in report
    assert "L04 trial 0" in report  # missing cell (the corrupt one)
    assert "web_search was called" in report
    assert "infra errors counted as wrong: accuracy 1/2" in report
    assert "estimated cost: $" in report
    assert "Tokens per question" in report
    assert (tmp_path / "report.md").exists()


def test_estimate_usd_and_long_context_calls():
    from bench.prices import estimate_usd, long_context_calls

    usage = {
        "prompt_token_count": 1_000_000,
        "cached_content_token_count": 500_000,
        "candidates_token_count": 100_000,
        "thoughts_token_count": 100_000,
    }
    # 0.5M fresh * $2 + 0.5M cached * $0.20 + 0.2M output * $12
    assert estimate_usd("gemini-3.1-pro-preview", usage) == pytest.approx(3.5)
    assert estimate_usd("no-such-model", usage) is None
    calls = [
        {"model": "gemini-3.1-pro-preview", "usage": {"prompt_token_count": 250_000}},
        {"model": "gemini-3.1-pro-preview", "usage": {"prompt_token_count": 1_000}},
        {"model": "gemini-3.8-flash", "usage": {"prompt_token_count": 250_000}},
        {"model": "gemini-3.1-pro-preview", "error": "503"},
    ]
    assert long_context_calls(calls) == 1


def test_report_on_legacy_traces_without_usage(tmp_path):
    atomic_write_json(tmp_path / "run_meta.json", {"model": "m", "trials": 1})
    trace = _graded("L03", 0)
    trace.pop("usage")
    atomic_write_json(tmp_path / "L03" / "trial_00.json", trace)
    report = build_report(tmp_path)
    assert "No usage recorded" in report
    assert "all-trials-correct" in report


def test_read_trace_rejects_non_objects(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("3")
    assert read_trace(p)[0] is None

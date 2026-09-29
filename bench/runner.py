"""Benchmark runner: drives full chat turns in-process against real sandboxes.

Mirrors tests/evals/conftest.py::run_turn but standalone: no HTTP, no auth, no
route-level prompt-guard (it runs once per question as a preflight instead,
recorded in run_meta). The code-judge guard inside execute_function_call_async
stays active: it is part of the system under test. Web grounding is disabled by
handing the model a tool list without the web_search declaration, and a
web_search call the model makes anyway is answered "Unknown function" without
executing (and flagged in the trace).

Failure taxonomy (each trace gets `error_bucket`):
- retryable: API 408/429/5xx or an httpx transport error. The whole trial is
  retried with backoff (Retry-After honored); if retries run out it is an
  infra_error.
- fatal: API 401/403/404, a 400 on the turn's first call (bad request/config,
  not the model), or SandboxBusy. The trace is written, then the run aborts.
- model: a model-caused turn failure (malformed/blocked/empty final response,
  a 400 after the model has already produced output, or the whole-turn
  timeout). Recorded as `model_error`; graded executable=False
  (model_malformed) and kept in every denominator.
- infra: anything else unexpected. infra_error, excluded from denominators.
infra_error trials are retried on resume; 3 consecutive ones abort the run.
"""

import asyncio
import contextlib
import email.utils
import fcntl
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from google.genai import errors as genai_errors
from google.genai import types

import app.chat_loop as chat_loop
from app.chat_loop import DoneEvent, run_chat_turn
from app.config import CHAT_MODEL_IDS, MAX_TOOL_ROUNDS
from app.genai_client import make_client
from app.query_engine import configure
from app.sandbox.pool import SandboxBusy
from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION
from bench import provenance
from bench.ground_truth import get_expected, parquet_identity, reference_sha
from bench.io import atomic_write_json, move_aside, read_trace
from bench.questions import Question, select
from bench.usage import (
    RecordingClient,
    aggregate_usage,
    call_anomalies,
    chat_calls,
    terminal_model_error,
)

logger = logging.getLogger(__name__)

BENCH_TOOLS = types.Tool(
    function_declarations=[
        d for d in FIRE_DATA_TOOLS.function_declarations if d.name != "web_search"
    ]
)
assert len(BENCH_TOOLS.function_declarations) == 2, (
    "expected exactly get_dataset_info + run_query after filtering web_search"
)

RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})
FATAL_CODES = frozenset({400, 401, 403, 404})
RETRY_BACKOFF_S = (10, 30)
RATE_LIMIT_BACKOFF_S = (60, 120)  # 429: quota windows are ~a minute
MAX_BACKOFF_S = 600
TRIAL_RETRIES = 2
MAX_CONSECUTIVE_INFRA = 3
TURN_TIMEOUT_S = 1800

# Bench-only HTTP policy (the production client keeps SDK defaults). The
# per-request timeout is generous because a thinking model's single call can
# run for minutes; the SDK retries transient codes itself before the trial-level
# retry above ever sees them.
HTTP_TIMEOUT_MS = 300_000
HTTP_OPTIONS = types.HttpOptions(
    timeout=HTTP_TIMEOUT_MS,
    retry_options=types.HttpRetryOptions(
        attempts=4,
        initial_delay=2.0,
        max_delay=60.0,
        http_status_codes=sorted(RETRYABLE_CODES),
    ),
)

LOCK_PATH = Path(os.environ.get("KINDLING_BENCH_LOCK", "/tmp/kindling-bench.lock"))

_DATA_CAP_CHARS = 20_000


class BenchAbort(Exception):
    """Stop the run (fatal config error or too many consecutive infra errors)."""


# ---------------------------------------------------------------------------
# Host-wide lock: two benches at once can each push a query to 80-90 GB on a
# 125 GB host. gt and run both hold it.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def bench_lock(path: Path = LOCK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.pread(fd, 64, 0).decode(errors="replace").strip()
            raise BenchAbort(
                f"another bench holds {path} (pid {holder or '?'}); only one "
                "bench may run per host"
            ) from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, str(os.getpid()).encode(), 0)
        yield
    finally:
        os.close(fd)  # releases the flock


# ---------------------------------------------------------------------------
# Error buckets
# ---------------------------------------------------------------------------


def classify_error(exc: BaseException, chat_calls_ok: int = 0) -> str:
    """retryable | fatal | model | infra (see module docstring)."""
    if isinstance(exc, SandboxBusy):
        return "fatal"
    if isinstance(exc, genai_errors.APIError):
        code = exc.code
        if code in RETRYABLE_CODES:
            return "retryable"
        if code == 400 and chat_calls_ok > 0:
            return "model"  # the model's own output made the request invalid
        if code in FATAL_CODES:
            return "fatal"
        return "infra"
    if isinstance(exc, httpx.TransportError):
        return "retryable"
    return "infra"


def retry_after_s(exc: BaseException) -> float | None:
    """Seconds from a Retry-After header on an APIError's response, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def backoff_s(exc: BaseException, attempt: int) -> float:
    code = getattr(exc, "code", None)
    table = RATE_LIMIT_BACKOFF_S if code == 429 else RETRY_BACKOFF_S
    base = table[min(attempt, len(table) - 1)]
    hinted = retry_after_s(exc)
    return min(MAX_BACKOFF_S, max(base, hinted or 0.0))


# ---------------------------------------------------------------------------
# Resume rules
# ---------------------------------------------------------------------------


def plan_trial(path: Path) -> tuple[str, str | None]:
    """("run", None) | ("skip", None) | ("retry", "corrupt" | "infra").

    Only a trace that parses and has infra_error null counts as done; the
    caller moves a corrupt or infra trace aside before re-running it."""
    if not path.exists():
        return "run", None
    trace, _ = read_trace(path)
    if trace is None:
        return "retry", "corrupt"
    if trace.get("infra_error") is not None:
        return "retry", "infra"
    return "skip", None


# ---------------------------------------------------------------------------
# Per-turn recording
# ---------------------------------------------------------------------------


@dataclass
class RecordingSession:
    """Wraps a SandboxSession to capture per-query latency and raw results,
    plus the bench-side tool events (rejections by source, web_search calls).

    Results pass through untouched: the chat loop (and model) see exactly what
    they would in production. Code-judge rejections short-circuit before the
    session, so they appear in `rejections` (source code_judge) only.
    """

    inner: object
    calls: list[dict]  # the trial's generate_content sink (for round numbers)
    records: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    web_search_calls: list[dict] = field(default_factory=list)
    missing_code_args: int = 0

    def current_round(self) -> int:
        return max(0, len(chat_calls(self.calls)) - 1)

    async def run_query(self, code: str) -> dict:
        t0 = time.monotonic()
        result = await self.inner.run_query(code)
        plots = result.get("plots") or []
        record = {
            "code": code,
            "round": self.current_round(),
            "latency_s": round(time.monotonic() - t0, 3),
            "error": result.get("error"),
            "total_rows": result.get("total_rows"),
            "truncated": result.get("truncated"),
            # The worker returns plot URLs (tools.py turns them into entries
            # after this returns); keep the raw URLs.
            "plots": [p if isinstance(p, str) else p.get("url") for p in plots],
        }
        data = result.get("data")
        if data is not None:
            data_str = json.dumps(data, default=str)
            record["data_capped"] = len(data_str) > _DATA_CAP_CHARS
            record["data"] = data_str[:_DATA_CAP_CHARS]
        self.records.append(record)
        return result


_ORIG_EXECUTE = chat_loop.execute_function_call_async


async def bench_execute(name, args, client, model, session):
    """chat_loop's tool dispatcher, wrapped: web_search never executes (it is
    not declared to the model) and every run_query error is recorded with its
    round and source. Everything else is the production dispatcher."""
    if not isinstance(session, RecordingSession):
        return await _ORIG_EXECUTE(name, args, client, model, session)
    rnd = session.current_round()
    if name == "web_search":
        session.web_search_calls.append({"round": rnd, "args": dict(args or {})})
        logger.error("model called undeclared web_search (blocked): %s", args)
        return json.dumps({"error": f"Unknown function: {name}"}), []
    n_before = len(session.records)
    result_str, plots = await _ORIG_EXECUTE(name, args, client, model, session)
    if name == "run_query":
        try:
            data = json.loads(result_str)
        except ValueError:
            data = {}
        if isinstance(data, dict) and "error" in data:
            code = (args or {}).get("code")
            if len(session.records) > n_before:
                source = "sandbox"
            elif not isinstance(code, str) or not code.strip():
                source = "missing_code_arg"
                session.missing_code_args += 1
            else:
                source = "code_judge"
            session.rejections.append(
                {"round": rnd, "source": source, "error": data["error"], "code": code}
            )
    return result_str, plots


@contextlib.contextmanager
def bench_tool_dispatch():
    chat_loop.execute_function_call_async = bench_execute
    try:
        yield
    finally:
        chat_loop.execute_function_call_async = _ORIG_EXECUTE


@dataclass
class Attempt:
    session: RecordingSession | None = None
    result: object | None = None
    latency_s: float | None = None


async def _run_turn_once(client, model, question, pool, max_rounds, attempt):
    """One full chat turn in a fresh sandbox; fills `attempt` as it goes so a
    timeout or error still leaves the partial records behind."""
    contents = [
        types.Content(role="user", parts=[types.Part(text=question.text)]),
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[BENCH_TOOLS],
    )
    sandbox = await pool.acquire_session()
    t0 = time.monotonic()  # after checkout: latency excludes sandbox waits
    attempt.session = RecordingSession(inner=sandbox, calls=client.sink)
    try:
        async for ev in run_chat_turn(
            client, model, contents, config, attempt.session, max_rounds=max_rounds
        ):
            if isinstance(ev, DoneEvent):
                attempt.result = ev.result
    finally:
        attempt.latency_s = round(time.monotonic() - t0, 3)
        pool.release_session(sandbox)
    if attempt.result is None:
        raise RuntimeError("chat turn ended without DoneEvent")


def _session_fields(session: RecordingSession | None) -> dict:
    if session is None:
        return {
            "query_records": [],
            "rejections": [],
            "web_search_calls": [],
            "missing_code_args": 0,
        }
    return {
        "query_records": session.records,
        "rejections": session.rejections,
        "web_search_calls": session.web_search_calls,
        "missing_code_args": session.missing_code_args,
    }


async def run_trial(
    client: RecordingClient,
    model: str,
    pool,
    question: Question,
    expected,
    trial: int,
    trial_path: Path,
    parquet: Path,
    identity: str,
    *,
    prompt_sha: str,
    question_sha: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
    retries: int = TRIAL_RETRIES,
    turn_timeout_s: float = TURN_TIMEOUT_S,
) -> dict:
    """Run one trial (with transient-error retries) and write its trace JSON.

    Returns the trace; its `error_bucket` tells the caller whether to abort."""
    trace = {
        "prompt": question.text,
        "trial": trial,
        "model": model,
        "model_version": None,
        "question_id": question.id,
        "category": question.category,
        "answer_kind": question.answer_kind,
        "parquet": str(parquet),
        "parquet_identity": identity,
        "prompt_sha": prompt_sha,
        "question_sha": question_sha,
        "reference_sha": reference_sha(question),
        "max_rounds": max_rounds,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "expected": expected,
        "infra_error": None,
        "model_error": None,
        "error_bucket": None,
        "retries_used": 0,
    }
    all_calls: list[dict] = []
    attempt_log: list[dict] = []
    for attempt_no in range(retries + 1):
        client.sink = []
        client.chat_model = model
        client.attempt = attempt_no
        att = Attempt()
        error: BaseException | None = None
        timed_out = False
        try:
            async with asyncio.timeout(turn_timeout_s) as cm:
                await _run_turn_once(client, model, question, pool, max_rounds, att)
        except TimeoutError as e:
            if cm.expired():
                timed_out = True
            else:
                error = e
        except Exception as e:  # noqa: BLE001 - classified below
            error = e
        calls = client.sink
        all_calls += calls
        trace["retries_used"] = attempt_no

        if error is not None:
            ok = sum(1 for c in chat_calls(calls) if "error" not in c)
            bucket = classify_error(error, ok)
            msg = f"{type(error).__name__}: {error}"
            attempt_log.append({"attempt": attempt_no, "bucket": bucket, "error": msg})
            if bucket == "retryable" and attempt_no < retries:
                delay = backoff_s(error, attempt_no)
                logger.warning(
                    "%s trial %d: retryable error (%s), retry %d/%d in %.0fs",
                    question.id,
                    trial,
                    msg[:300],
                    attempt_no + 1,
                    retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if bucket == "model":
                trace["model_error"] = msg
                trace["error_bucket"] = "model"
            else:
                trace["infra_error"] = msg
                trace["error_bucket"] = bucket
            logger.error("%s trial %d: %s error: %s", question.id, trial, bucket, msg)
        elif timed_out:
            trace["model_error"] = f"turn timeout after {turn_timeout_s:.0f}s"
            trace["error_bucket"] = "model"
            logger.error("%s trial %d: turn timed out", question.id, trial)
        else:
            trace["model_error"] = terminal_model_error(calls)
            if trace["model_error"]:
                trace["error_bucket"] = "model"
        result = att.result
        trace.update(
            {
                "tool_calls": result.tool_calls if result else [],
                "queries_run": result.queries_run if result else [],
                "rejected_queries": result.rejected_queries if result else [],
                "loop_exhausted": result.loop_exhausted if result else False,
                "text": result.text if result else "",
                "plots": [p.get("name") for p in result.plots] if result else [],
                "turn_latency_s": att.latency_s,
                **_session_fields(att.session),
            }
        )
        break

    served = sorted(
        {c["model_version"] for c in chat_calls(all_calls) if c.get("model_version")}
    )
    last_chat = [c for c in chat_calls(client.sink) if "error" not in c]
    trace.update(
        {
            "model_version": ",".join(served) or None,
            "finish_reason": last_chat[-1].get("finish_reason") if last_chat else None,
            "model_anomalies": call_anomalies(client.sink),
            "attempts": attempt_log,
            "calls": all_calls,
            "usage": aggregate_usage(all_calls),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    if trace["web_search_calls"]:
        logger.error(
            "%s trial %d: model called web_search %d time(s); blocked, flagged",
            question.id,
            trial,
            len(trace["web_search_calls"]),
        )
    atomic_write_json(trial_path, trace)
    return trace


# ---------------------------------------------------------------------------
# run_meta
# ---------------------------------------------------------------------------


def open_run_dir(run_dir: Path, meta: dict, *, allow_drift: bool) -> dict:
    """Create run_meta.json, or on resume refuse a drifted fingerprint.

    Returns the meta now on disk (resumes are appended under `resumes`)."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(meta_path, meta)
        return meta
    existing, err = read_trace(meta_path)
    if existing is None:
        raise BenchAbort(f"{meta_path} is unreadable ({err}); use a new --run-dir")
    hard = provenance.identity_drift(existing, meta)
    if hard:
        raise BenchAbort(
            f"run dir {run_dir} was started with a different model or dataset:\n  "
            + "\n  ".join(hard)
            + "\nUse a new --run-dir (--allow-drift does not waive this)."
        )
    diffs = provenance.check_drift(existing, meta)
    if diffs and not allow_drift:
        raise BenchAbort(
            f"run dir {run_dir} fingerprint drifted since it was started:\n  "
            + "\n  ".join(diffs)
            + "\nUse a new --run-dir, or pass --allow-drift to resume anyway."
        )
    if diffs:
        logger.warning("resuming despite drift (--allow-drift): %s", diffs)
    existing.setdefault("resumes", []).append(
        {"started_at": meta["started_at"], "argv": meta["argv"], "drift": diffs}
    )
    existing["question_shas"] = {
        **existing.get("question_shas", {}),
        **meta["question_shas"],
    }
    existing["question_set_sha"] = provenance.question_set_sha(
        existing["question_shas"]
    )
    existing["trials"] = max(existing.get("trials", 0), meta["trials"])
    existing.pop("ended_at", None)
    atomic_write_json(meta_path, existing)
    return existing


def update_meta(run_dir: Path, **fields) -> None:
    meta_path = run_dir / "run_meta.json"
    meta, _ = read_trace(meta_path)
    meta = meta or {}
    meta.update(fields)
    atomic_write_json(meta_path, meta)


def guard_preflight(client, questions: list[Question], shas: dict, done: dict):
    """Prompt-guard verdict per question (the route-level guard the bench
    bypasses). Makes one flash-lite call per question not already recorded
    with the same question_sha. Only ever called from `run`."""
    from app.guards import guard_prompt

    out = dict(done or {})
    for q in questions:
        prev = out.get(q.id)
        if prev and prev.get("question_sha") == shas[q.id]:
            continue
        allow, reason = guard_prompt(q.text, client)
        out[q.id] = {"allow": allow, "reason": reason, "question_sha": shas[q.id]}
        if not allow:
            logger.warning(
                "prompt-guard would BLOCK %s in production: %s", q.id, reason
            )
    return out


# ---------------------------------------------------------------------------
# The `run` subcommand
# ---------------------------------------------------------------------------


def parse_models(args) -> list[str]:
    raw = args.models or args.model
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise BenchAbort("no model given")
    unknown = [m for m in models if m not in CHAT_MODEL_IDS]
    if unknown:
        raise BenchAbort(
            f"unknown model(s) {unknown}; choose from {sorted(CHAT_MODEL_IDS)} "
            "(app/config.py CHAT_MODELS)"
        )
    if len(set(models)) != len(models):
        raise BenchAbort(f"duplicate model in {models}")
    return models


def model_run_dirs(run_dir: Path, models: list[str]) -> dict[str, Path]:
    """One standard single-model run dir per model: the run dir itself for a
    single model, <run_dir>/<model>/ for an interleaved multi-model run."""
    if len(models) == 1:
        return {models[0]: run_dir}
    return {m: run_dir / m for m in models}


def trial_cells(questions, trials: int, models: list[str], seed: int) -> list:
    """(question, trial, model) cells. Multi-model runs are shuffled with a
    seeded RNG so neither model systematically runs earlier (API load, cache
    warmth, time of day); a single-model run stays question-major."""
    cells = [(q, t, m) for q in questions for t in range(trials) for m in models]
    if len(models) > 1:
        random.Random(seed).shuffle(cells)
    return cells


def preflight(settings: dict) -> str:
    """Cheap local checks before ground truth (no API call): key present,
    container runtime, worker image in the host store. Returns the runtime."""
    from app.sandbox.pool import detect_runtime, runtime_available

    if not os.environ.get("GEMINI_API_KEY"):
        raise BenchAbort("GEMINI_API_KEY not set (env or .env)")
    if not runtime_available():
        raise BenchAbort("no container runtime (query execution is container-only)")
    runtime = settings.get("runtime") or detect_runtime()
    if provenance.image_id(runtime, settings["image"]) is None:
        raise BenchAbort(
            f"worker image {settings['image']} not in the {runtime} image store; "
            "build it: make build-worker"
        )
    return runtime


async def run_bench(args) -> Path:
    """The `run` subcommand."""
    from app.sandbox.pool import SandboxPool

    models = parse_models(args)
    parquet = Path(args.parquet)
    if not parquet.exists():
        raise BenchAbort(f"parquet not found: {parquet}")
    settings = provenance.sandbox_settings_from_env()
    runtime = preflight(settings)

    configure(parquet)  # get_dataset_info reads schema/samples from this
    questions = select(args.questions)
    identity = parquet_identity(parquet)
    # Ground truth first: fail fast on reference bugs before any API spend.
    expected = {q.id: get_expected(q, parquet) for q in questions}

    client = RecordingClient(
        make_client(os.environ["GEMINI_API_KEY"], http_options=HTTP_OPTIONS)
    )
    pool = SandboxPool(parquet, runtime=runtime, **settings)
    sandbox = provenance.effective_sandbox(pool, settings)
    worker_image = {
        "image": settings["image"],
        "id": provenance.image_id(runtime, settings["image"]),
        "versions": provenance.worker_versions(runtime, settings["image"]),
    }
    p_sha = provenance.prompt_sha(SYSTEM_INSTRUCTION, BENCH_TOOLS)
    q_shas = {q.id: provenance.question_sha(q) for q in questions}
    http_meta = {
        "timeout_ms": HTTP_TIMEOUT_MS,
        "sdk_retry_attempts": HTTP_OPTIONS.retry_options.attempts,
        "trial_retries": TRIAL_RETRIES,
        "turn_timeout_s": TURN_TIMEOUT_S,
    }

    run_dir = Path(args.run_dir or f".bench-runs/{time.strftime('%Y%m%d-%H%M%S')}")
    dirs = model_run_dirs(run_dir, models)
    started = datetime.now().isoformat(timespec="seconds")
    metas = {}
    for m, d in dirs.items():
        fp = provenance.build_fingerprint(
            model=m,
            parquet_identity=identity,
            prompt=p_sha,
            question_shas=q_shas,
            max_rounds=args.max_rounds,
            sandbox=sandbox,
            worker_image=worker_image,
            http_options=http_meta,
        )
        metas[m] = open_run_dir(
            d,
            {
                **fp,
                "parquet": str(parquet),
                "trials": args.trials,
                "interleaved_models": models if len(models) > 1 else None,
                "seed": args.seed,
                "generation_config": "app defaults (thinking and temperature "
                "not overridden)",
                "started_at": started,
            },
            allow_drift=args.allow_drift,
        )
    log_handler = logging.FileHandler(run_dir / "bench.log")
    logging.getLogger().addHandler(log_handler)

    first_meta = metas[models[0]]
    verdicts = guard_preflight(
        client, questions, q_shas, first_meta.get("prompt_guard_preflight")
    )
    for d in dirs.values():
        update_meta(d, prompt_guard_preflight=verdicts)

    cells = trial_cells(questions, args.trials, models, args.seed)
    await pool.start()
    done = skipped = 0
    consecutive_infra = 0
    try:
        with bench_tool_dispatch():
            for q, trial, m in cells:
                trial_path = dirs[m] / q.id / f"trial_{trial:02d}.json"
                action, why = plan_trial(trial_path)
                if action == "skip":
                    skipped += 1
                    continue
                if action == "retry":
                    moved = move_aside(trial_path, why)
                    logger.info("%s trial %d: %s trace -> %s", q.id, trial, why, moved)
                logger.info(
                    "=== %s %s [%s] trial %d: %s", m, q.id, q.category, trial, q.text
                )
                trace = await run_trial(
                    client,
                    m,
                    pool,
                    q,
                    expected[q.id],
                    trial,
                    trial_path,
                    parquet,
                    identity,
                    prompt_sha=p_sha,
                    question_sha=q_shas[q.id],
                    max_rounds=args.max_rounds,
                )
                done += 1
                tokens = sum(u["total_token_count"] for u in trace["usage"].values())
                logger.info(
                    "%s %s trial %d: %s in %.1fs, %d tokens",
                    m,
                    q.id,
                    trial,
                    trace["error_bucket"] or "done",
                    trace.get("turn_latency_s") or 0.0,
                    tokens,
                )
                if trace["error_bucket"] == "fatal":
                    raise BenchAbort(
                        f"fatal error on {m} {q.id} trial {trial}: "
                        f"{trace['infra_error']} (fix the config and resume)"
                    )
                if trace["infra_error"]:
                    consecutive_infra += 1
                    if consecutive_infra >= MAX_CONSECUTIVE_INFRA:
                        raise BenchAbort(
                            f"{consecutive_infra} consecutive infra errors, last: "
                            f"{trace['infra_error']} (resume retries them)"
                        )
                else:
                    consecutive_infra = 0
    finally:
        await pool.drain()
        ended = datetime.now().isoformat(timespec="seconds")
        for d in dirs.values():
            update_meta(d, ended_at=ended)
        logging.getLogger().removeHandler(log_handler)
    logger.info("run complete: %d trials run, %d skipped -> %s", done, skipped, run_dir)
    return run_dir


def main_run(args) -> None:
    try:
        with bench_lock():
            asyncio.run(run_bench(args))
    except BenchAbort as e:
        sys.exit(f"bench run aborted: {e}")

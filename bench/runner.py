"""Benchmark runner: drives full chat turns in-process against real sandboxes.

Mirrors tests/evals/conftest.py::run_turn but standalone: no HTTP, no auth, no
prompt-guard (route-level). The code-judge guard inside
execute_function_call_async stays active — it is part of the system under test.
Web grounding is disabled by handing the model a tool list without the
web_search declaration.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from google.genai import types

from app.chat_loop import DoneEvent, run_chat_turn
from app.genai_client import make_client
from app.query_engine import configure
from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION
from bench.ground_truth import get_expected, parquet_identity
from bench.questions import Question, select

logger = logging.getLogger(__name__)

BENCH_TOOLS = types.Tool(
    function_declarations=[
        d for d in FIRE_DATA_TOOLS.function_declarations if d.name != "web_search"
    ]
)
assert len(BENCH_TOOLS.function_declarations) == 2, (
    "expected exactly get_dataset_info + run_query after filtering web_search"
)

# Substrings of google-genai exceptions worth retrying a whole trial for.
TRANSIENT_MARKERS = (
    "429",
    "500",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "INTERNAL",
    "DeadlineExceeded",
    "overloaded",
)
RETRY_BACKOFF_S = (10, 30)

_DATA_CAP_CHARS = 20_000


@dataclass
class RecordingSession:
    """Wraps a SandboxSession to capture per-query latency and raw results.

    Results pass through untouched — the chat loop (and model) see exactly what
    they would in production. Code-judge rejections short-circuit before the
    session, so they appear only in tool_calls/rejected_queries, not here.
    """

    inner: object
    records: list[dict] = field(default_factory=list)

    async def run_query(self, code: str) -> dict:
        t0 = time.monotonic()
        result = await self.inner.run_query(code)
        record = {
            "code": code,
            "latency_s": round(time.monotonic() - t0, 3),
            "error": result.get("error"),
            "total_rows": result.get("total_rows"),
            "truncated": result.get("truncated"),
            "plots": [
                p["name"] for p in result.get("plots", []) if isinstance(p, dict)
            ],
        }
        data = result.get("data")
        if data is not None:
            data_str = json.dumps(data, default=str)
            record["data_capped"] = len(data_str) > _DATA_CAP_CHARS
            record["data"] = data_str[:_DATA_CAP_CHARS]
        self.records.append(record)
        return result


def _is_transient(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    return any(marker in msg for marker in TRANSIENT_MARKERS)


async def _run_turn_once(client, model, question, pool, max_rounds):
    """One full chat turn in a fresh sandbox; returns (result, records, latency)."""
    contents = [
        types.Content(role="user", parts=[types.Part(text=question.text)]),
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[BENCH_TOOLS],
    )
    t0 = time.monotonic()
    sandbox = await pool.acquire_session()
    session = RecordingSession(inner=sandbox)
    result = None
    try:
        async for ev in run_chat_turn(
            client, model, contents, config, session, max_rounds=max_rounds
        ):
            if isinstance(ev, DoneEvent):
                result = ev.result
    finally:
        pool.release_session(sandbox)
    latency = round(time.monotonic() - t0, 3)
    if result is None:
        raise RuntimeError("chat turn ended without DoneEvent")
    return result, session.records, latency


async def run_trial(
    client,
    model: str,
    pool,
    question: Question,
    expected,
    trial: int,
    trial_path: Path,
    parquet: Path,
    identity: str,
    max_rounds: int = 15,
    retries: int = 2,
) -> dict:
    """Run one trial (with transient-error retries) and write its trace JSON."""
    started_at = datetime.now().isoformat(timespec="seconds")
    trace = {
        "prompt": question.text,
        "trial": trial,
        "model": model,
        "question_id": question.id,
        "category": question.category,
        "answer_kind": question.answer_kind,
        "parquet": str(parquet),
        "parquet_identity": identity,
        "started_at": started_at,
        "expected": expected,
        "infra_error": None,
        "retries_used": 0,
    }
    attempt = 0
    while True:
        try:
            result, records, latency = await _run_turn_once(
                client, model, question, pool, max_rounds
            )
            trace.update(
                {
                    "tool_calls": result.tool_calls,
                    "queries_run": result.queries_run,
                    "rejected_queries": result.rejected_queries,
                    "loop_exhausted": result.loop_exhausted,
                    "text": result.text,
                    "plots": [p.get("name") for p in result.plots],
                    "turn_latency_s": latency,
                    "query_records": records,
                    "retries_used": attempt,
                }
            )
            break
        except Exception as e:  # noqa: BLE001 — classify, retry or record
            if _is_transient(e) and attempt < retries:
                backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                attempt += 1
                logger.warning(
                    "%s trial %d: transient error (%s), retry %d/%d in %ds",
                    question.id,
                    trial,
                    e,
                    attempt,
                    retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            logger.error("%s trial %d: giving up: %s", question.id, trial, e)
            trace["infra_error"] = f"{type(e).__name__}: {e}"
            trace["retries_used"] = attempt
            break

    trial_path.parent.mkdir(parents=True, exist_ok=True)
    trial_path.write_text(json.dumps(trace, indent=2, default=str))
    return trace


def _check_run_meta(run_dir: Path, meta: dict) -> None:
    """On resume, refuse to mix models or datasets within one run dir."""
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        for key in ("model", "parquet_identity"):
            if existing.get(key) != meta[key]:
                sys.exit(
                    f"run dir {run_dir} was started with {key}="
                    f"{existing.get(key)!r}, now {meta[key]!r} — use a new --run-dir"
                )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


async def run_bench(args) -> Path:
    """The `run` subcommand: sequential trials with resume-by-skip."""
    import os

    from app.sandbox.pool import SandboxPool, runtime_available

    parquet = Path(args.parquet)
    if not parquet.exists():
        sys.exit(f"parquet not found: {parquet}")
    configure(parquet)  # get_dataset_info reads schema/samples from this

    questions = select(args.questions)
    identity = parquet_identity(parquet)

    # Ground truth first: fail fast on reference bugs before any API spend.
    expected = {q.id: get_expected(q, parquet) for q in questions}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set (env or .env)")
    client = make_client(api_key)

    if not runtime_available():
        sys.exit("no container runtime (query execution is container-only)")

    run_dir = Path(args.run_dir or f".bench-runs/{time.strftime('%Y%m%d-%H%M%S')}")
    _check_run_meta(
        run_dir,
        {
            "model": args.model,
            "parquet": str(parquet),
            "parquet_identity": identity,
            "trials": args.trials,
            "max_rounds": args.max_rounds,
            "argv": sys.argv[1:],
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    logging.getLogger().addHandler(logging.FileHandler(run_dir / "bench.log"))

    pool = SandboxPool(parquet, size=1, max_total=2)
    await pool.start()
    done = skipped = 0
    try:
        for q in questions:
            for trial in range(args.trials):
                trial_path = run_dir / q.id / f"trial_{trial:02d}.json"
                if trial_path.exists():
                    skipped += 1
                    logger.info("%s trial %d: trace exists, skipping", q.id, trial)
                    continue
                logger.info("=== %s [%s] trial %d: %s", q.id, q.category, trial, q.text)
                trace = await run_trial(
                    client,
                    args.model,
                    pool,
                    q,
                    expected[q.id],
                    trial,
                    trial_path,
                    parquet,
                    identity,
                    max_rounds=args.max_rounds,
                )
                done += 1
                logger.info(
                    "%s trial %d: %s in %.1fs",
                    q.id,
                    trial,
                    "infra_error" if trace["infra_error"] else "done",
                    trace.get("turn_latency_s", 0.0),
                )
    finally:
        await pool.drain()
    logger.info("run complete: %d trials run, %d skipped -> %s", done, skipped, run_dir)
    return run_dir

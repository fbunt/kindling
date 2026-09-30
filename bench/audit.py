"""Hand-audit hook: a seeded sample of graded trials for a human verdict.

`python -m bench audit --run-dir DIR` writes <run_dir>/audit.json with a
random `fraction` (default 10%) of the graded non-infra trials. Selection
ranks trials by sha256(seed, question, trial), so the same seed picks the
same trials as long as the graded set is unchanged. Fill `human_verdict`
(pass/fail) per entry; re-running keeps every filled verdict, including for
trials that fall outside a new sample. report compares the filled verdicts
with the grader's automated verdict (`auto_accurate`, before adjudication).
Each entry records the trace's response_sha; a filled entry keeps the
response it was judged on, and one whose trace has changed since (a trial
re-run on resume) is stale and left out of the agreement.
"""

import hashlib
import json
import logging
import math
from pathlib import Path

from bench.grading import is_current_grade, parse_verdict, response_sha
from bench.io import atomic_write_json, read_trace
from bench.questions import BY_ID

logger = logging.getLogger(__name__)

AUDIT_FILE = "audit.json"


class AuditFileError(Exception):
    pass


def _rank(seed: int, qid: str, trial) -> str:
    return hashlib.sha256(f"{seed}:{qid}:{trial}".encode()).hexdigest()


def load_audit(run_dir: Path) -> dict | None:
    """The audit file, or None if absent. Raises AuditFileError when it is
    unreadable (it holds hand-filled verdicts: never overwrite it blindly)."""
    path = Path(run_dir) / AUDIT_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise AuditFileError(f"{path}: unreadable ({e}); fix or remove it") from e
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise AuditFileError(f"{path}: expected an object with an `entries` list")
    return data


def eligible(traces: list[dict]) -> list[dict]:
    """Graded by the current grader, not infra errors, no grader error."""
    return [t for t in traces if not t.get("infra_error") and is_current_grade(t)]


def _entry(trace: dict) -> dict:
    return {
        "question_id": trace["question_id"],
        "trial": trace["trial"],
        "trace_path": trace.get("_path"),
        "question": BY_ID[trace["question_id"]].text,
        "expected": trace.get("expected"),
        "response": trace.get("text") or "",
        "response_sha": response_sha(trace),
        "extraction": trace.get("extraction"),
        "grader_verdict": trace.get("auto_accurate"),
        "grader_reason": trace.get("auto_reason"),
        "partial_data": trace.get("partial_data"),
        "human_verdict": "",
        "human_note": "",
    }


def _keep_judged(entry: dict, prev: dict) -> dict:
    """entry with prev's hand-filled fields and the response (and hash) the
    human judged; a prev without a hash stays without one."""
    entry["human_verdict"] = prev.get("human_verdict", "")
    entry["human_note"] = prev.get("human_note", "")
    for f in ("response", "response_sha"):
        if f in prev:
            entry[f] = prev[f]
        else:
            entry.pop(f, None)
    return entry


def write_audit(run_dir: Path, *, fraction: float = 0.1, seed: int = 0) -> Path:
    run_dir = Path(run_dir)
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    old = load_audit(run_dir)
    traces = []
    for path in sorted(run_dir.glob("*/trial_*.json")):
        trace, _ = read_trace(path)
        if trace is not None and "question_id" in trace:
            trace["_path"] = str(path)
            traces.append(trace)
    pool = eligible(traces)
    k = math.ceil(fraction * len(pool)) if pool else 0
    picked = sorted(pool, key=lambda t: _rank(seed, t["question_id"], t["trial"]))[:k]
    by_key = {(t["question_id"], t["trial"]): t for t in pool}
    kept = {}
    for e in (old or {}).get("entries", []):
        key = (e.get("question_id"), e.get("trial"))
        if str(e.get("human_verdict") or "").strip() or e.get("human_note"):
            kept[key] = e
    entries = []
    for t in picked:
        key = (t["question_id"], t["trial"])
        e = _entry(t)
        if key in kept:
            e = _keep_judged(e, kept.pop(key))
        entries.append(e)
    for key, prev in sorted(kept.items(), key=lambda kv: str(kv[0])):
        # Filled outside the current sample: keep, refreshed if still graded.
        e = _entry(by_key[key]) if key in by_key else dict(prev)
        entries.append(_keep_judged(e, prev))
    path = run_dir / AUDIT_FILE
    atomic_write_json(
        path,
        {"seed": seed, "fraction": fraction, "eligible": len(pool), "entries": entries},
    )
    logger.info(
        "audit: %d of %d graded trials -> %s (fill human_verdict with pass/fail)",
        len(entries),
        len(pool),
        path,
    )
    return path


def agreement(audit: dict | None, traces: list[dict]) -> dict | None:
    """{audited, filled, agree, pct, stale} against each trace's current
    automated verdict; None without an audit file. A filled entry whose
    response_sha no longer matches its trace is stale: counted, not compared."""
    if not audit:
        return None
    by_key = {(t["question_id"], t.get("trial")): t for t in traces}
    entries = audit.get("entries") or []
    filled = agree = stale = 0
    for e in entries:
        human = parse_verdict(e.get("human_verdict"))
        if human is None:
            continue
        trace = by_key.get((e.get("question_id"), e.get("trial")))
        if trace and e.get("response_sha") and e["response_sha"] != response_sha(trace):
            stale += 1
            continue
        auto = trace.get("auto_accurate") if trace else e.get("grader_verdict")
        if auto is None:
            continue
        filled += 1
        agree += human == bool(auto)
    return {
        "audited": len(entries),
        "filled": filled,
        "agree": agree,
        "pct": 100 * agree / filled if filled else None,
        "stale": stale,
    }

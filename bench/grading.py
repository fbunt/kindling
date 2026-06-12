"""Grading: executability and accuracy predicates over trial traces.

Accuracy is judged primarily by an LLM judge (3 independent flash-lite votes,
majority) against a criterion rendered from the question template with the
ground-truth value and tolerance injected. For scalar answers a mechanical
numeric check runs alongside; it never overrides the judge, but disagreements
are flagged for manual spot-checking. Failure-mode classification is manual:
grade_run emits triage.json with a blank failure_mode per failed trial.
"""

import json
import logging
import re
from pathlib import Path

from bench.questions import BY_ID, Question
from tests.evals.judge import judge

logger = logging.getLogger(__name__)

JUDGE_VOTES = 3

# Exact substrings of the pool/worker resource-kill messages
# (app/sandbox/worker.py, app/sandbox/pool.py).
_RESOURCE_MARKERS = ("timed out", "terminated unexpectedly")

STANDARD_SUFFIX = (
    " If the response asks a clarifying question or declines to give a "
    "definite final answer, answer 'no'."
)

FAILURE_MODES = ("logical_error", "domain_semantic_error", "performance_violation")


# ---------------------------------------------------------------------------
# Executability
# ---------------------------------------------------------------------------


def _run_query_outcomes(trace: dict) -> list[dict | None]:
    """Per run_query tool call (in order): its query_record, or None if it
    never reached the sandbox (code-judge rejection)."""
    records = list(trace.get("query_records") or [])
    outcomes = []
    for call in trace.get("tool_calls") or []:
        if call.get("name") != "run_query":
            continue
        code = (call.get("args") or {}).get("code")
        if records and records[0]["code"] == code:
            outcomes.append(records.pop(0))
        else:
            outcomes.append(None)
    return outcomes


def _outcome_error(trace: dict, outcome: dict | None) -> str | None:
    if outcome is not None:
        return outcome.get("error")
    # Code-judge rejection: only the last rejection is kept in the trace.
    rejected = trace.get("rejected_queries") or []
    if rejected:
        return rejected[-1].get("error", "blocked by safety review")
    return "blocked by safety review"


def is_executable(trace: dict) -> tuple[bool, str]:
    """The paper's executability predicate: the run made at least one
    run_query call, its final query completed without error (intermediate
    self-corrected errors are fine), and the tool loop terminated normally."""
    if trace.get("infra_error"):
        return False, "infra_error"  # excluded from denominators, not failed
    outcomes = _run_query_outcomes(trace)
    if not outcomes:
        return False, "no_queries"
    if trace.get("loop_exhausted"):
        return False, "loop_exhausted"
    if _outcome_error(trace, outcomes[-1]) is not None:
        return False, "last_query_error"
    return True, "ok"


def _all_errors(trace: dict) -> list[str]:
    errs = [r.get("error") for r in trace.get("query_records") or [] if r.get("error")]
    errs += [q.get("error", "") for q in trace.get("rejected_queries") or []]
    return [e for e in errs if e]


def resource_violation(trace: dict) -> bool:
    return any(
        marker in err for err in _all_errors(trace) for marker in _RESOURCE_MARKERS
    )


# ---------------------------------------------------------------------------
# Criterion rendering
# ---------------------------------------------------------------------------


def _fmt(x) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if x == int(x) and abs(x) < 1e15:
            return f"{int(x):,}"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.4g}"
    return str(x)


def scalar_band(question: Question, expected: float) -> float:
    if question.tolerance_abs is not None:
        return question.tolerance_abs
    return question.tolerance_rel * abs(expected)


def _fmt_expected(question: Question, expected) -> str:
    if question.answer_kind == "series":
        return "; ".join(f"{k}: {_fmt(v)}" for k, v in expected.items())
    if question.answer_kind == "set":
        return "; ".join(str(v) for v in expected)
    if question.answer_kind == "text":
        if isinstance(expected, list):
            return " or ".join(f"'{v}'" for v in expected)
        return f"'{expected}'"
    return _fmt(expected)  # scalar


def render_criterion(question: Question, expected) -> str:
    crit = question.criterion.replace("{expected}", _fmt_expected(question, expected))
    if question.answer_kind == "scalar":
        band = scalar_band(question, float(expected))
        if band > 0:
            crit += (
                f" Accept any stated value between {_fmt(expected - band)} and "
                f"{_fmt(expected + band)} (a tolerance of ±{_fmt(band)}); "
                "rounding within that range is fine."
            )
        else:
            crit += " The value must match exactly."
    elif question.answer_kind == "series":
        pct = question.tolerance_rel * 100
        crit += (
            f" Each listed value must appear within ±{pct:g}% of the stated "
            "number; extra commentary is fine; a missing or out-of-band value "
            "means 'no'."
        )
    elif question.answer_kind == "set":
        crit += (
            " Order may differ and case/punctuation differences are fine, but "
            "the response must name exactly these and no substitutes."
        )
    return crit + STANDARD_SUFFIX


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def judge_majority(
    client, text: str, criterion: str, votes: int = JUDGE_VOTES
) -> tuple[bool, list[bool]]:
    results = [
        judge(client, response_text=text, criterion=criterion) for _ in range(votes)
    ]
    return sum(results) > votes / 2, results


_NUMBER_RE = re.compile(
    r"(-?\$?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*(thousand|million|billion|trillion))?",
    re.IGNORECASE,
)
_MULTIPLIERS = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}


def extract_numbers(text: str) -> list[float]:
    out = []
    for num, mult in _NUMBER_RE.findall(text):
        try:
            value = float(num.replace(",", "").replace("$", ""))
        except ValueError:
            continue
        if mult:
            value *= _MULTIPLIERS[mult.lower()]
        out.append(value)
    return out


def numbers_close(got: float, expected: float, band: float) -> bool:
    return abs(got - expected) <= band


def mechanical_scalar_check(
    text: str, question: Question, expected: float
) -> bool | None:
    """True if ANY number in the response falls within the tolerance band;
    None when the response contains no parseable numbers. A weak signal — it
    never overrides the judge, only flags disagreements."""
    numbers = extract_numbers(text)
    if not numbers:
        return None
    band = scalar_band(question, float(expected))
    return any(numbers_close(n, float(expected), band) for n in numbers)


# ---------------------------------------------------------------------------
# Run-level grading + triage
# ---------------------------------------------------------------------------


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/trial_*.json"))


def grade_trace(trace: dict, client) -> dict:
    """Compute the grade fields for one trace (pure; no I/O)."""
    question = BY_ID[trace["question_id"]]
    executable, reason = is_executable(trace)
    fields = {
        "executable": executable,
        "executability_reason": reason,
        "resource_violation": resource_violation(trace),
        "had_query_errors": bool(_all_errors(trace)),
        "judge_criterion": None,
        "judge_votes": None,
        "judge_verdict": None,
        "mechanical_verdict": None,
        "judge_mechanical_disagree": False,
        "accurate": False,
    }
    if not executable:
        return fields
    criterion = render_criterion(question, trace["expected"])
    verdict, votes = judge_majority(client, trace["text"], criterion)
    fields.update(
        {"judge_criterion": criterion, "judge_votes": votes, "judge_verdict": verdict}
    )
    if question.answer_kind == "scalar":
        mech = mechanical_scalar_check(trace["text"], question, trace["expected"])
        fields["mechanical_verdict"] = mech
        fields["judge_mechanical_disagree"] = mech is not None and mech != verdict
    fields["accurate"] = verdict
    return fields


def grade_run(run_dir: Path, client, *, regrade: bool = False) -> list[dict]:
    """Grade every trace in a run dir (merging fields in place) + triage."""
    traces = []
    for path in _trace_paths(run_dir):
        trace = json.loads(path.read_text())
        if "accurate" in trace and not regrade:
            logger.info("%s: already graded, skipping", path)
        else:
            trace.update(grade_trace(trace, client))
            path.write_text(json.dumps(trace, indent=2, default=str))
            logger.info(
                "%s trial %d: executable=%s accurate=%s",
                trace["question_id"],
                trace["trial"],
                trace.get("executable"),
                trace.get("accurate"),
            )
        trace["_path"] = str(path)
        traces.append(trace)
    write_triage(run_dir, traces)
    return traces


def write_triage(run_dir: Path, traces: list[dict]) -> None:
    """triage.json: one entry per failed (non-accurate, non-infra) trial, for
    MANUAL failure_mode annotation (logical_error / domain_semantic_error /
    performance_violation = first point of divergence from the reference
    query). Hand-filled failure_mode values survive re-grading; entries for
    now-passing trials are dropped."""
    triage_path = run_dir / "triage.json"
    existing = {}
    if triage_path.exists():
        for entry in json.loads(triage_path.read_text()):
            existing[(entry["question_id"], entry["trial"])] = entry
    entries = []
    for trace in traces:
        if trace.get("infra_error") or trace.get("accurate"):
            continue
        if "accurate" not in trace:
            continue  # ungraded (shouldn't happen after grade_run)
        key = (trace["question_id"], trace["trial"])
        old = existing.get(key, {})
        entries.append(
            {
                "question_id": trace["question_id"],
                "trial": trace["trial"],
                "trace_path": trace["_path"],
                "executable": trace["executable"],
                "executability_reason": trace["executability_reason"],
                "resource_violation": trace["resource_violation"],
                "summary": (trace.get("text") or "")[:300],
                "failure_mode": old.get("failure_mode", ""),
            }
        )
    triage_path.write_text(json.dumps(entries, indent=2))
    unannotated = sum(1 for e in entries if not e["failure_mode"])
    logger.info(
        "triage: %d failed trials (%d unannotated) -> %s",
        len(entries),
        unannotated,
        triage_path,
    )

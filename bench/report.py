"""Markdown report over a benchmark run directory (graded or not).

Hand-filled overlays are applied at load time, so editing adjudication.json
or audit.json and re-running `report` needs no regrade."""

import json
import logging
from pathlib import Path
from statistics import mean, median

from bench.audit import AuditFileError, agreement, load_audit
from bench.grading import (
    AdjudicationFileError,
    adjudication_index,
    apply_adjudication,
    grader_sha,
    load_adjudication,
)
from bench.io import read_trace
from bench.prices import (
    LONG_CONTEXT_THRESHOLD,
    PRICES_AS_OF,
    estimate_usd,
    long_context_calls,
)
from bench.questions import CATEGORIES
from bench.usage import merge_usage, total_tokens

logger = logging.getLogger(__name__)


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict], list[str]]:
    """(meta, traces, triage, corrupt paths), with adjudication.json applied
    to the traces. Unreadable files are reported, never raised."""
    meta, _ = read_trace(run_dir / "run_meta.json")
    traces, corrupt = [], []
    for path in sorted(run_dir.glob("*/trial_*.json")):
        trace, err = read_trace(path)
        if trace is None or "question_id" not in trace:
            corrupt.append(f"{path} ({err or 'no question_id'})")
            continue
        trace["_path"] = str(path)
        traces.append(trace)
    try:
        decided = adjudication_index(load_adjudication(run_dir))
    except AdjudicationFileError as e:
        corrupt.append(f"{e} (overlay not applied)")
        decided = {}
    for trace in traces:
        apply_adjudication(trace, decided)
    triage = []
    triage_path = run_dir / "triage.json"
    if triage_path.exists():
        try:
            triage = json.loads(triage_path.read_text())
        except ValueError:
            corrupt.append(f"{triage_path} (unreadable)")
    return meta or {}, traces, triage, corrupt


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "-"


def _median_or_dash(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    return f"{median(values):.1f}" if values else "-"


def _trial_target(meta: dict, traces: list[dict]) -> int:
    target = meta.get("trials")
    if isinstance(target, int) and target > 0:
        return target
    return max((t.get("trial", 0) + 1 for t in traces), default=0)


def is_graded(t: dict) -> bool:
    """A non-infra trial with a verdict (a grader error is not one)."""
    return not t.get("infra_error") and "accurate" in t and not t.get("grader_error")


def _clarified(t: dict) -> bool:
    return (t.get("extraction") or {}).get("status") == "clarification"


def rates(ts: list[dict]) -> dict:
    """Headline rates over graded non-infra trials (model-caused failures stay
    in every denominator). Accuracy is strict: clarifications, multiple
    candidates and partial-data answers fail."""
    n = len(ts)
    execs = [t for t in ts if t.get("executable")]
    return {
        "n": n,
        "accurate": sum(bool(t.get("accurate")) for t in ts),
        "executable": len(execs),
        "joint": sum(bool(t.get("accurate")) for t in execs),
        "clarified": sum(_clarified(t) for t in ts),
    }


def _category_rows(graded: list[dict], target: int) -> list[list[str]]:
    rows = []
    cats = [c for c in CATEGORIES if any(t["category"] == c for t in graded)]
    for cat in [*cats, "overall"]:
        ts = graded if cat == "overall" else [t for t in graded if t["category"] == cat]
        qids = sorted({t["question_id"] for t in ts})
        all_correct = sum(
            1
            for qid in qids
            if all(t.get("accurate") for t in ts if t["question_id"] == qid)
            and sum(t["question_id"] == qid for t in ts) >= target
        )
        r = rates(ts)
        rows.append(
            [
                cat,
                len(qids),
                _pct(r["accurate"], r["n"]),
                _pct(r["executable"], r["n"]),
                _pct(r["joint"], r["n"]),
                _pct(r["clarified"], r["n"]),
                _pct(r["joint"], r["executable"]),
                f"{all_correct}/{len(qids)}",
                _median_or_dash([t.get("turn_latency_s") for t in ts]),
            ]
        )
    return rows


def _flags(trials: list[dict]) -> str:
    sha = grader_sha()
    checks = (
        ("resource_violation", lambda t: t.get("resource_violation")),
        ("needs_review", lambda t: t.get("needs_review")),
        ("adjudicated", lambda t: t.get("adjudicated")),
        ("ADJUDICATION_ISSUE", lambda t: t.get("adjudication_issue")),
        ("partial_data", lambda t: t.get("partial_data")),
        ("clarified", _clarified),
        ("loop_exhausted", lambda t: t.get("loop_exhausted")),
        ("model_error", lambda t: t.get("model_error")),
        ("infra_error", lambda t: t.get("infra_error")),
        ("GRADER_ERROR", lambda t: t.get("grader_error")),
        ("WEB_SEARCH_CALLED", lambda t: t.get("web_search_calls")),
        ("ungraded", lambda t: not t.get("infra_error") and "accurate" not in t),
        (
            "old_grader",
            lambda t: "accurate" in t and t.get("grader_sha") != sha,
        ),
    )
    return ", ".join(name for name, pred in checks if any(pred(t) for t in trials))


def _question_rows(traces: list[dict], target: int) -> list[list[str]]:
    rows = []
    for qid in sorted({t["question_id"] for t in traces}):
        trials = [t for t in traces if t["question_id"] == qid]
        graded = [t for t in trials if is_graded(t)]
        n = len(graded)
        acc = sum(bool(t.get("accurate")) for t in graded)
        query_lats = [
            r.get("latency_s") for t in graded for r in t.get("query_records") or []
        ]
        reasons: dict[str, int] = {}
        for t in graded:
            if not t.get("accurate") and t.get("accuracy_reason"):
                reasons[t["accuracy_reason"]] = reasons.get(t["accuracy_reason"], 0) + 1
        rows.append(
            [
                qid,
                trials[0].get("category", "?"),
                f"{sum(bool(t.get('executable')) for t in graded)}/{n}",
                f"{acc}/{n}",
                f"{acc / n:.2f}" if n else "-",
                "yes" if n >= target and acc == n else "no",
                _median_or_dash([t.get("turn_latency_s") for t in graded]),
                _median_or_dash(query_lats),
                ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())),
                _flags(trials),
            ]
        )
    return rows


def _grading_section(
    run_dir: Path, graded: list[dict], errors: list[dict]
) -> list[str]:
    sha = grader_sha()
    shas = sorted({str(t.get("grader_sha")) for t in graded})
    versions = sorted(
        {
            (t.get("extractor") or {}).get("model_version") or "?"
            for t in graded
            if (t.get("extractor") or {}).get("attempts")
        }
    )
    n = len(graded)
    partial = [t for t in graded if t.get("partial_data")]
    disclosed = sum(bool(t.get("sampled_disclosed")) for t in partial)
    partial_passed = sum(
        1 for t in partial if t.get("adjudicated") and t.get("accurate")
    )
    judged = [t for t in graded if t.get("grounded") is not None]
    grounded = sum(bool(t.get("grounded")) for t in judged)
    adjudicated = [t for t in graded if t.get("adjudicated")]
    to_pass = sum(
        1 for t in adjudicated if t["adjudicated"]["overrode"] and t["accurate"]
    )
    to_fail = sum(
        1 for t in adjudicated if t["adjudicated"]["overrode"] and not t["accurate"]
    )
    pending = pending_review(graded)
    issues = {
        kind: [
            t for t in graded if (t.get("adjudication_issue") or {}).get("kind") == kind
        ]
        for kind in ("unparseable", "stale")
    }
    why: dict[str, int] = {}
    for t in pending:
        for r in t.get("review_reasons") or ():
            why[r] = why.get(r, 0) + 1
    parts = [
        "## Grading",
        "",
        f"- grader sha: {', '.join(f'`{s}`' for s in shas)}"
        + ("" if shas == [sha] else f" - **current grader is `{sha}`; re-run grade**"),
        f"- extractor served version(s): {', '.join(versions) or '-'}",
        f"- grader errors (extractor failed twice; counted as ungraded): "
        f"{len(errors)}" + (" - **re-run grade to retry**" if errors else ""),
        f"- partial data: {_pct(len(partial), n)} of trials; disclosed "
        f"{_pct(disclosed, n)}, undisclosed {_pct(len(partial) - disclosed, n)} "
        + (
            f"(strict accuracy fails {len(partial) - partial_passed} of them; "
            f"{partial_passed} adjudicated to pass)"
            if partial_passed
            else "(strict accuracy fails all of them)"
        ),
        f"- grounded (answer appears in a query result): {_pct(grounded, len(judged))}",
        f"- adjudicated: {len(adjudicated)} ({to_pass} overridden to pass, "
        f"{to_fail} to fail); pending review: {len(pending)}"
        + (
            f" ({', '.join(f'{k} x{v}' for k, v in sorted(why.items()))})"
            if why
            else ""
        ),
    ]
    for kind, what in (
        ("unparseable", "verdict is not pass/fail; write pass or fail"),
        (
            "stale",
            "the trace's response changed since it was adjudicated; delete the "
            "entry and re-run grade to re-add it",
        ),
    ):
        if issues[kind]:
            parts.append(
                f"- **{len(issues[kind])} {kind} adjudication(s), not applied** "
                f"({what}): "
                + ", ".join(
                    f"{t['question_id']} trial {t['trial']} "
                    f"({t['adjudication_issue']['verdict']!r})"
                    for t in issues[kind]
                )
            )
    try:
        agree = agreement(load_audit(run_dir), graded)
    except AuditFileError as e:
        agree = None
        parts.append(f"- **audit file unreadable**: {e}")
    if agree:
        parts.append(
            f"- hand audit: {agree['audited']} sampled, {agree['filled']} filled; "
            "judge-human agreement "
            + (
                f"{agree['agree']}/{agree['filled']} ({agree['pct']:.0f}%)"
                if agree["filled"]
                else "- (no human_verdict filled yet)"
            )
            + (
                f"; {agree['stale']} stale (trace changed since audit; ignored)"
                if agree.get("stale")
                else ""
            )
        )
    else:
        parts.append("- hand audit: none (run `python -m bench audit`)")
    parts.append("")
    if pending:
        parts += [
            "### Pending adjudication (review flags)",
            "",
            f"Fill `verdict` (pass/fail) in `{run_dir / 'adjudication.json'}`:",
            "",
            *(
                f"- {t['question_id']} trial {t['trial']}: "
                f"{', '.join(t.get('review_reasons') or ())}; auto="
                f"{t.get('auto_accurate')} ({t.get('auto_reason')}), extracted "
                f"{_extracted(t)!r}, unit {(t.get('extraction') or {}).get('unit')!r}"
                for t in pending
            ),
            "",
        ]
    return parts


def pending_review(graded: list[dict]) -> list[dict]:
    """Trials flagged for review without an applied adjudication."""
    return [t for t in graded if t.get("needs_review") and not t.get("adjudicated")]


def _extracted(trace: dict):
    ex = trace.get("extraction") or {}
    return ex.get("value", ex.get("values", ex.get("names")))


def missing_cells(meta: dict, traces: list[dict], target: int) -> list[str]:
    """(question, trial) cells with no readable trace, over the questions the
    run was started with (meta question_shas) and trials 0..target-1."""
    qids = sorted(meta.get("question_shas") or {t["question_id"] for t in traces})
    have = {(t["question_id"], t.get("trial")) for t in traces}
    return [
        f"{qid} trial {i}"
        for qid in qids
        for i in range(target)
        if (qid, i) not in have
    ]


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def _cost_section(traces: list[dict], accurate: int) -> list[str]:
    with_usage = [t for t in traces if t.get("usage")]
    if not with_usage:
        return [
            "## Tokens and cost",
            "",
            "No usage recorded (traces predate usage capture).",
            "",
        ]
    totals = merge_usage(*(t["usage"] for t in with_usage))
    rows, usd_total, usd_missing = [], 0.0, []
    for model, u in sorted(totals.items()):
        usd = estimate_usd(model, u)
        if usd is None:
            usd_missing.append(model)
        else:
            usd_total += usd
        rows.append(
            [
                model,
                _fmt_int(u["calls"]),
                _fmt_int(u["prompt_token_count"]),
                _fmt_int(u["cached_content_token_count"]),
                _fmt_int(u["candidates_token_count"]),
                _fmt_int(u["thoughts_token_count"]),
                _fmt_int(u["total_token_count"]),
                f"${usd:.2f}" if usd is not None else "-",
            ]
        )
    per_trial = [total_tokens(t["usage"]) for t in with_usage]
    grand = sum(per_trial)
    parts = [
        "## Tokens and cost",
        "",
        "All generate_content calls the trials made: chat rounds plus code-judge "
        "verdicts, including attempts that were later retried. Grading "
        "(extractor) and prompt-guard preflight calls are not included. Thinking and "
        "temperature are the app defaults (not overridden).",
        "",
        _md_table(
            [
                "model",
                "calls",
                "prompt",
                "cached",
                "output",
                "thoughts",
                "total",
                "est. USD",
            ],
            rows,
        ),
        "",
        f"- trials with usage: {len(with_usage)}; tokens per trial: mean "
        f"{_fmt_int(mean(per_trial))}, median {_fmt_int(median(per_trial))}, "
        f"max {_fmt_int(max(per_trial))}",
        "- tokens per correct answer: "
        + (f"{_fmt_int(grand / accurate)} ({accurate} correct)" if accurate else "-"),
    ]
    if usd_missing:
        parts.append(
            f"- **no dollar estimate for {', '.join(usd_missing)}: prices are not "
            "filled in** (bench/prices.py); token totals above are complete"
        )
    else:
        parts.append(
            f"- estimated cost: ${usd_total:.2f} (prices as of {PRICES_AS_OF}); "
            + (f"${usd_total / accurate:.3f} per correct answer" if accurate else "")
        )
    n_long = sum(long_context_calls(t.get("calls") or []) for t in with_usage)
    if n_long:
        parts.append(
            f"- **{n_long} call(s) had prompts over {_fmt_int(LONG_CONTEXT_THRESHOLD)} "
            "tokens**: billed at the long-context rate, so the estimate "
            "undercounts them"
        )
    q_rows = []
    for qid in sorted({t["question_id"] for t in with_usage}):
        toks = [total_tokens(t["usage"]) for t in with_usage if t["question_id"] == qid]
        q_rows.append([qid, len(toks), _fmt_int(sum(toks)), _fmt_int(mean(toks))])
    parts += [
        "",
        "### Tokens per question",
        "",
        _md_table(["id", "trials", "total tokens", "mean per trial"], q_rows),
        "",
    ]
    return parts


def build_report(run_dir: Path) -> str:
    meta, traces, triage, corrupt = load_run(run_dir)
    target = _trial_target(meta, traces)
    infra = [t for t in traces if t.get("infra_error")]
    live = [t for t in traces if not t.get("infra_error")]
    graded = [t for t in live if is_graded(t)]
    ungraded = [t for t in live if not is_graded(t)]
    grader_errors = [t for t in live if t.get("grader_error")]
    missing = missing_cells(meta, traces, target)
    web = [t for t in traces if t.get("web_search_calls")]
    served = sorted({t["model_version"] for t in traces if t.get("model_version")})
    buckets: dict[str, int] = {}
    for t in traces:
        if t.get("error_bucket"):
            buckets[t["error_bucket"]] = buckets.get(t["error_bucket"], 0) + 1

    parts = [
        f"# Benchmark report - {run_dir.name}",
        "",
        f"- model: `{meta.get('model', '?')}`"
        + (
            f" (interleaved with {', '.join(meta['interleaved_models'])}, seed "
            f"{meta.get('seed')})"
            if meta.get("interleaved_models")
            else ""
        ),
        f"- served model version(s): {', '.join(served) or '?'}",
        f"- parquet: `{meta.get('parquet', '?')}` "
        f"(identity `{meta.get('parquet_identity', '?')}`)",
        f"- prompt sha `{meta.get('prompt_sha', '?')}`, question set sha "
        f"`{meta.get('question_set_sha', '?')}`, git `{meta.get('git_sha', '?')}`"
        + (" (dirty)" if meta.get("git_dirty") else ""),
        f"- trials per question: {target}",
        f"- trials graded: {len(graded)}"
        + (f" - **{len(ungraded)} ungraded (run `grade`)**" if ungraded else ""),
        f"- trials excluded as infra errors: {len(infra)}",
        f"- error buckets: {buckets or 'none'}",
        "",
    ]
    if corrupt:
        parts += [
            f"**{len(corrupt)} unreadable file(s) skipped:**",
            "",
            *(f"- `{c}`" for c in corrupt),
            "",
        ]
    if missing:
        parts += [
            f"**{len(missing)} missing (question, trial) cell(s)** (no readable "
            "trace; resume the run to fill them): " + ", ".join(missing),
            "",
        ]
    if web:
        parts += [
            "**INVALID: web_search was called (and blocked) in "
            f"{len(web)} trial(s):** "
            + ", ".join(f"{t['question_id']} trial {t['trial']}" for t in web),
            "",
        ]

    if graded:
        acc = sum(bool(t.get("accurate")) for t in graded)
        parts += [
            "## Per category",
            "",
            _md_table(
                [
                    "category",
                    "questions",
                    "accuracy (strict)",
                    "executability",
                    "exec & accurate",
                    "clarification",
                    "exec-gated accuracy",
                    "all-trials-correct",
                    "median turn latency (s)",
                ],
                _category_rows(graded, target),
            ),
            "",
            "Accuracy is strict and not gated on executability: a clarification, "
            "a multiple-candidate answer or a partial-data answer fails even if a "
            "number lands in the band. Exec-gated accuracy (accurate among "
            "executable trials) is secondary.",
            "",
            f"Sensitivity, infra errors counted as wrong: accuracy "
            f"{_pct(acc, len(graded) + len(infra))}.",
            "",
            "Latency medians include failed trials and start after sandbox "
            "checkout; infra-error trials are excluded everywhere else. "
            "Model-caused failures (model_malformed) stay in every denominator.",
            "",
        ]
    if traces:
        parts += [
            "## Per question",
            "",
            _md_table(
                [
                    "id",
                    "category",
                    "exec",
                    "acc",
                    "pass@1",
                    "all-correct",
                    "med turn (s)",
                    "med query (s)",
                    "fail reasons",
                    "flags",
                ],
                _question_rows(traces, target),
            ),
            "",
        ]

    if graded:
        parts += _grading_section(run_dir, graded, grader_errors)

    parts += _cost_section(traces, sum(bool(t.get("accurate")) for t in graded))

    parts += ["## Failure triage", ""]
    if triage:
        unannotated = sum(1 for e in triage if not e.get("failure_mode"))
        parts += [
            _md_table(
                [
                    "question",
                    "trial",
                    "executable",
                    "exec reason",
                    "accuracy reason",
                    "failure mode",
                ],
                [
                    [
                        e.get("question_id"),
                        e.get("trial"),
                        "yes" if e.get("executable") else "no",
                        str(e.get("executability_reason"))
                        + (" (resource)" if e.get("resource_violation") else ""),
                        e.get("accuracy_reason") or "-",
                        e.get("failure_mode") or "**UNANNOTATED**",
                    ]
                    for e in triage
                ],
            ),
            "",
        ]
        annotated = [e for e in triage if e.get("failure_mode")]
        if annotated:
            dist: dict[str, int] = {}
            for e in annotated:
                dist[e["failure_mode"]] = dist.get(e["failure_mode"], 0) + 1
            parts += [
                "### Failure-mode distribution (annotated only)",
                "",
                _md_table(
                    ["failure mode", "count"],
                    [[k, v] for k, v in sorted(dist.items())],
                ),
                "",
            ]
        if unannotated:
            parts.append(
                f"{unannotated} of {len(triage)} failures unannotated - fill "
                f"`failure_mode` in `{run_dir / 'triage.json'}` and re-run `report`."
            )
            parts.append("")
    else:
        parts += ["No failed trials (or not graded yet).", ""]

    report = "\n".join(parts)
    out = run_dir / "report.md"
    out.write_text(report)
    logger.info("report written to %s", out)
    return report

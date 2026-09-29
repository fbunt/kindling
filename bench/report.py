"""Markdown report over a benchmark run directory (graded or not)."""

import json
import logging
from pathlib import Path
from statistics import mean, median

from bench.io import read_trace
from bench.prices import PRICES_AS_OF, estimate_usd
from bench.questions import CATEGORIES
from bench.usage import merge_usage, total_tokens

logger = logging.getLogger(__name__)


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict], list[str]]:
    """(meta, traces, triage, corrupt paths). Unreadable files are reported,
    never raised."""
    meta, _ = read_trace(run_dir / "run_meta.json")
    traces, corrupt = [], []
    for path in sorted(run_dir.glob("*/trial_*.json")):
        trace, err = read_trace(path)
        if trace is None or "question_id" not in trace:
            corrupt.append(f"{path} ({err or 'no question_id'})")
            continue
        trace["_path"] = str(path)
        traces.append(trace)
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
        rows.append(
            [
                cat,
                len(qids),
                _pct(sum(bool(t.get("executable")) for t in ts), len(ts)),
                _pct(sum(bool(t.get("accurate")) for t in ts), len(ts)),
                f"{all_correct}/{len(qids)}",
                _median_or_dash([t.get("turn_latency_s") for t in ts]),
            ]
        )
    return rows


def _flags(trials: list[dict]) -> str:
    checks = (
        ("resource_violation", lambda t: t.get("resource_violation")),
        ("judge_disagree", lambda t: t.get("judge_mechanical_disagree")),
        ("loop_exhausted", lambda t: t.get("loop_exhausted")),
        ("model_error", lambda t: t.get("model_error")),
        ("infra_error", lambda t: t.get("infra_error")),
        ("WEB_SEARCH_CALLED", lambda t: t.get("web_search_calls")),
        ("ungraded", lambda t: not t.get("infra_error") and "accurate" not in t),
    )
    return ", ".join(name for name, pred in checks if any(pred(t) for t in trials))


def _question_rows(traces: list[dict], target: int) -> list[list[str]]:
    rows = []
    for qid in sorted({t["question_id"] for t in traces}):
        trials = [t for t in traces if t["question_id"] == qid]
        graded = [t for t in trials if not t.get("infra_error") and "accurate" in t]
        n = len(graded)
        acc = sum(bool(t.get("accurate")) for t in graded)
        query_lats = [
            r.get("latency_s") for t in graded for r in t.get("query_records") or []
        ]
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
                _flags(trials),
            ]
        )
    return rows


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
        "verdicts, including attempts that were later retried. Grading-judge "
        "and prompt-guard preflight calls are not included. Thinking and "
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
    graded = [t for t in traces if not t.get("infra_error") and "accurate" in t]
    ungraded = [t for t in traces if not t.get("infra_error") and "accurate" not in t]
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
                    "executability",
                    "accuracy",
                    "all-trials-correct",
                    "median turn latency (s)",
                ],
                _category_rows(graded, target),
            ),
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
                    "flags",
                ],
                _question_rows(traces, target),
            ),
            "",
        ]

    parts += _cost_section(traces, sum(bool(t.get("accurate")) for t in graded))

    parts += ["## Failure triage", ""]
    if triage:
        unannotated = sum(1 for e in triage if not e.get("failure_mode"))
        parts += [
            _md_table(
                ["question", "trial", "executable", "reason", "failure mode"],
                [
                    [
                        e.get("question_id"),
                        e.get("trial"),
                        "yes" if e.get("executable") else "no",
                        str(e.get("executability_reason"))
                        + (" (resource)" if e.get("resource_violation") else ""),
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

    disagreements = [t for t in graded if t.get("judge_mechanical_disagree")]
    if disagreements:
        parts += [
            "## Judge vs mechanical-check disagreements (spot-check these)",
            "",
            *(
                f"- {t['question_id']} trial {t['trial']}: judge="
                f"{t.get('judge_verdict')} mechanical={t.get('mechanical_verdict')} "
                f"- `{t['_path']}`"
                for t in disagreements
            ),
            "",
        ]

    report = "\n".join(parts)
    out = run_dir / "report.md"
    out.write_text(report)
    logger.info("report written to %s", out)
    return report

"""Markdown report over a graded benchmark run directory."""

import json
import logging
from pathlib import Path
from statistics import median

from bench.questions import CATEGORIES

logger = logging.getLogger(__name__)


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    traces = []
    for path in sorted(run_dir.glob("*/trial_*.json")):
        trace = json.loads(path.read_text())
        trace["_path"] = str(path)
        traces.append(trace)
    triage_path = run_dir / "triage.json"
    triage = json.loads(triage_path.read_text()) if triage_path.exists() else []
    return meta, traces, triage


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "–"


def _median_or_dash(values: list[float]) -> str:
    return f"{median(values):.1f}" if values else "–"


def _category_rows(graded: list[dict]) -> list[list[str]]:
    rows = []
    cats = [c for c in CATEGORIES if any(t["category"] == c for t in graded)]
    for cat in [*cats, "overall"]:
        ts = graded if cat == "overall" else [t for t in graded if t["category"] == cat]
        qids = sorted({t["question_id"] for t in ts})
        all_correct = sum(
            1
            for qid in qids
            if all(t["accurate"] for t in ts if t["question_id"] == qid)
            and sum(t["question_id"] == qid for t in ts) >= 3
        )
        rows.append(
            [
                cat,
                len(qids),
                _pct(sum(t["executable"] for t in ts), len(ts)),
                _pct(sum(t["accurate"] for t in ts), len(ts)),
                f"{all_correct}/{len(qids)}",
                _median_or_dash([t["turn_latency_s"] for t in ts]),
            ]
        )
    return rows


def _flags(trials: list[dict]) -> str:
    flags = []
    if any(t.get("resource_violation") for t in trials):
        flags.append("resource_violation")
    if any(t.get("judge_mechanical_disagree") for t in trials):
        flags.append("judge_disagree")
    if any(t.get("loop_exhausted") for t in trials):
        flags.append("loop_exhausted")
    if any(t.get("infra_error") for t in trials):
        flags.append("infra_error")
    return ", ".join(flags)


def _question_rows(traces: list[dict]) -> list[list[str]]:
    rows = []
    qids = sorted({t["question_id"] for t in traces})
    for qid in qids:
        trials = [t for t in traces if t["question_id"] == qid]
        graded = [t for t in trials if not t.get("infra_error")]
        n = len(graded)
        acc = sum(t["accurate"] for t in graded)
        query_lats = [
            r["latency_s"] for t in graded for r in t.get("query_records") or []
        ]
        rows.append(
            [
                qid,
                trials[0]["category"],
                f"{sum(t['executable'] for t in graded)}/{n}",
                f"{acc}/{n}",
                f"{acc / n:.2f}" if n else "–",
                "yes" if n >= 3 and acc == n else "no",
                _median_or_dash([t["turn_latency_s"] for t in graded]),
                _median_or_dash(query_lats),
                _flags(trials),
            ]
        )
    return rows


def build_report(run_dir: Path) -> str:
    meta, traces, triage = load_run(run_dir)
    graded = [t for t in traces if not t.get("infra_error") and "accurate" in t]
    ungraded = [t for t in traces if not t.get("infra_error") and "accurate" not in t]
    infra = [t for t in traces if t.get("infra_error")]

    parts = [
        f"# Benchmark report — {run_dir.name}",
        "",
        f"- model: `{meta.get('model', '?')}`",
        f"- parquet: `{meta.get('parquet', '?')}` "
        f"(identity `{meta.get('parquet_identity', '?')}`)",
        f"- trials per question: {meta.get('trials', '?')}",
        f"- trials graded: {len(graded)}"
        + (f" — **{len(ungraded)} ungraded (run `grade`)**" if ungraded else ""),
        f"- trials excluded as infra errors: {len(infra)}",
        "",
    ]

    if graded:
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
                _category_rows(graded),
            ),
            "",
            "Latency medians include failed trials (a wrong answer's latency is "
            "still a real latency); infra-error trials are excluded everywhere.",
            "",
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
                _question_rows(traces),
            ),
            "",
        ]

    parts += ["## Failure triage", ""]
    if triage:
        unannotated = sum(1 for e in triage if not e["failure_mode"])
        parts += [
            _md_table(
                ["question", "trial", "executable", "reason", "failure mode"],
                [
                    [
                        e["question_id"],
                        e["trial"],
                        "yes" if e["executable"] else "no",
                        e["executability_reason"]
                        + (" (resource)" if e["resource_violation"] else ""),
                        e["failure_mode"] or "**UNANNOTATED**",
                    ]
                    for e in triage
                ],
            ),
            "",
        ]
        annotated = [e for e in triage if e["failure_mode"]]
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
                f"{unannotated} of {len(triage)} failures unannotated — fill "
                f"`failure_mode` in `{run_dir / 'triage.json'}` and re-run `report`."
            )
            parts.append("")
    else:
        parts += ["No failed trials.", ""]

    disagreements = [t for t in graded if t.get("judge_mechanical_disagree")]
    if disagreements:
        parts += [
            "## Judge vs mechanical-check disagreements (spot-check these)",
            "",
            *(
                f"- {t['question_id']} trial {t['trial']}: judge="
                f"{t['judge_verdict']} mechanical={t['mechanical_verdict']} "
                f"— `{t['_path']}`"
                for t in disagreements
            ),
            "",
        ]

    report = "\n".join(parts)
    out = run_dir / "report.md"
    out.write_text(report)
    logger.info("report written to %s", out)
    return report

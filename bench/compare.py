"""Paired, question-clustered comparison of two graded runs (`bench compare`).

The unit of analysis is the question: each model's per-question pass rate
(accurate / graded non-infra trials, adjudication applied) is averaged with
equal weight per question (macro average = pass@1). Uncertainty comes from
resampling questions, not trials, because trials of one question are not
independent:

- bootstrap: resample the question set with replacement (seeded), 95%
  percentile interval for each model's macro rate and for the paired
  difference A - B;
- sign-flip permutation test on the per-question differences (two-sided,
  seeded Monte Carlo; 2^25 exact enumerations is too many).

compare refuses runs that are not comparable: different parquet identity,
different reference_sha for any question, different grader_sha, different
question sets, or any ungraded / grader-error trial. Per-category rows are
descriptive only (a handful of questions each; no interval). A review-status
table shows, per run, the flagged trials still pending review and the
unparseable / stale adjudications, since the headline rates use automated
verdicts for all of those.
"""

import random
from math import comb
from pathlib import Path
from statistics import mean, median

from bench.prices import estimate_usd
from bench.questions import BY_ID, CATEGORIES
from bench.report import _md_table, is_graded, load_run, pending_review
from bench.usage import merge_usage, total_tokens

DEFAULT_RESAMPLES = 10_000
DEFAULT_PERMUTATIONS = 10_000


class CompareError(Exception):
    pass


def _load(run_dir: Path) -> tuple[dict, list[dict]]:
    if not (run_dir / "run_meta.json").exists():
        raise CompareError(f"{run_dir}: no run_meta.json (not a single-model run dir)")
    meta, traces, _triage, corrupt = load_run(run_dir)
    if corrupt:
        raise CompareError(f"{run_dir}: unreadable file(s): {', '.join(corrupt)}")
    bad = [t for t in traces if not t.get("infra_error") and not is_graded(t)]
    if bad:
        raise CompareError(
            f"{run_dir}: {len(bad)} ungraded or grader-error trial(s) (run "
            "`python -m bench grade`): "
            + ", ".join(f"{t['question_id']} trial {t.get('trial')}" for t in bad[:10])
        )
    graded = [t for t in traces if is_graded(t)]
    legacy = [t for t in graded if not t.get("grader_sha")]
    if legacy:
        raise CompareError(
            f"{run_dir}: {len(legacy)} trial(s) graded before grader_sha existed "
            "(the old 3-vote judge); run `python -m bench grade` to regrade"
        )
    return meta, graded


def _one(values: set, what: str) -> object:
    if len(values) != 1:
        raise CompareError(f"runs differ in {what}: {sorted(map(str, values))}")
    return next(iter(values))


def check_comparable(runs: list[tuple[Path, dict, list[dict]]]) -> None:
    identities = {meta.get("parquet_identity") for _, meta, _ in runs}
    identities |= {t.get("parquet_identity") for _, _, ts in runs for t in ts}
    identities.discard(None)
    if not identities:
        raise CompareError("neither run records a parquet identity")
    _one(identities, "parquet identity")
    _one({t.get("grader_sha") for _, _, ts in runs for t in ts}, "grader_sha")
    for qid in sorted({t["question_id"] for _, _, ts in runs for t in ts}):
        shas = {
            t.get("reference_sha")
            for _, _, ts in runs
            for t in ts
            if t["question_id"] == qid
        }
        _one(shas, f"reference_sha for {qid}")
    qsets = [frozenset(t["question_id"] for t in ts) for _, _, ts in runs]
    if qsets[0] != qsets[1]:
        only_a = sorted(qsets[0] - qsets[1])
        only_b = sorted(qsets[1] - qsets[0])
        raise CompareError(
            f"runs cover different questions (only in A: {only_a}; only in B: "
            f"{only_b}); a paired comparison needs the same set"
        )


def per_question(traces: list[dict]) -> dict[str, tuple[int, int]]:
    """question -> (accurate trials, graded trials)."""
    out: dict[str, list[int]] = {}
    for t in traces:
        c = out.setdefault(t["question_id"], [0, 0])
        c[0] += bool(t.get("accurate"))
        c[1] += 1
    return {q: (c, n) for q, (c, n) in out.items()}


def pass_hat_k(c: int, n: int, k: int) -> float:
    """Unbiased pass^k for one question: P(k draws without replacement from
    its n trials all pass) = C(c, k) / C(n, k)."""
    return comb(c, k) / comb(n, k)


def _quantile(sorted_xs: list[float], q: float) -> float:
    return sorted_xs[min(len(sorted_xs) - 1, max(0, round(q * (len(sorted_xs) - 1))))]


def bootstrap(
    pa: list[float], pb: list[float], *, resamples: int, seed: int
) -> dict[str, tuple[float, float]]:
    """95% percentile intervals for mean(pa), mean(pb), mean(pa - pb) over
    question resamples (the same resample for both models: paired)."""
    rng = random.Random(seed)
    n = len(pa)
    a_s, b_s, d_s = [], [], []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = sum(pa[i] for i in idx) / n
        mb = sum(pb[i] for i in idx) / n
        a_s.append(ma)
        b_s.append(mb)
        d_s.append(ma - mb)
    out = {}
    for name, xs in (("a", a_s), ("b", b_s), ("diff", d_s)):
        xs.sort()
        out[name] = (_quantile(xs, 0.025), _quantile(xs, 0.975))
    return out


def sign_flip_p(diffs: list[float], *, permutations: int, seed: int) -> float:
    """Two-sided Monte Carlo sign-flip p-value for mean(diffs) != 0,
    (1 + #{|T*| >= |T|}) / (1 + permutations)."""
    if not any(diffs):
        return 1.0
    rng = random.Random(seed)
    obs = abs(sum(diffs))
    hits = 0
    for _ in range(permutations):
        t = sum(d if rng.random() < 0.5 else -d for d in diffs)
        hits += abs(t) >= obs - 1e-12
    return (1 + hits) / (1 + permutations)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(lo_hi: tuple[float, float]) -> str:
    return f"[{_pct(lo_hi[0])}, {_pct(lo_hi[1])}]"


def _pp(x: float) -> str:
    return f"{100 * x:+.1f} pp"


def _final_attempt_calls(t: dict) -> list[dict]:
    used = t.get("retries_used") or 0
    return [c for c in t.get("calls") or [] if c.get("attempt", 0) == used]


def _cost_rows(label: str, traces: list[dict]) -> list[str]:
    usage = merge_usage(*(t.get("usage") or {} for t in traces))
    usd = [estimate_usd(m, u) for m, u in usage.items()]
    usd_s = "-" if not usage or None in usd else f"${sum(usd):.2f}"
    correct = sum(bool(t.get("accurate")) for t in traces)
    toks = sum(total_tokens(t.get("usage")) for t in traces)

    def med(values):
        values = [v for v in values if v is not None]
        return f"{median(values):.1f}" if values else "-"

    chat_s = [
        sum(
            c.get("elapsed_s") or 0
            for c in _final_attempt_calls(t)
            if c.get("purpose") == "chat"
        )
        for t in traces
        if t.get("calls")
    ]
    judge_s = [
        sum(
            c.get("elapsed_s") or 0
            for c in _final_attempt_calls(t)
            if c.get("purpose") == "judge"
        )
        for t in traces
        if t.get("calls")
    ]
    sandbox_s = [
        sum(r.get("latency_s") or 0 for r in t.get("query_records") or [])
        for t in traces
    ]
    return [
        label,
        str(len(traces)),
        f"{toks:,}",
        usd_s,
        f"{toks // correct:,}" if correct else "-",
        med([t.get("turn_latency_s") for t in traces]),
        med(chat_s),
        med(judge_s),
        med(sandbox_s),
    ]


def compare_runs(
    a: Path,
    b: Path,
    *,
    seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> str:
    """Markdown comparison of run dirs A and B (raises CompareError)."""
    meta_a, ta = _load(a)
    meta_b, tb = _load(b)
    check_comparable([(a, meta_a, ta), (b, meta_b, tb)])
    name_a = meta_a.get("model") or a.name
    name_b = meta_b.get("model") or b.name
    if name_a == name_b:
        name_a, name_b = f"{name_a} ({a.name})", f"{name_b} ({b.name})"
    qa, qb = per_question(ta), per_question(tb)
    qids = sorted(qa)
    if not qids:
        raise CompareError("no graded trials to compare")
    pa = [qa[q][0] / qa[q][1] for q in qids]
    pb = [qb[q][0] / qb[q][1] for q in qids]
    diffs = [x - y for x, y in zip(pa, pb, strict=True)]
    ci = bootstrap(pa, pb, resamples=resamples, seed=seed)
    p = sign_flip_p(diffs, permutations=permutations, seed=seed)
    k = min(n for _, n in [*qa.values(), *qb.values()])
    hat_a = mean(pass_hat_k(*qa[q], k) for q in qids)
    hat_b = mean(pass_hat_k(*qb[q], k) for q in qids)

    review_rows, unreviewed = [], 0
    for name, ts in ((name_a, ta), (name_b, tb)):
        pending = len(pending_review(ts))
        unreviewed += pending
        kinds = [(t.get("adjudication_issue") or {}).get("kind") for t in ts]
        counts = [pending, kinds.count("unparseable"), kinds.count("stale")]
        review_rows.append([name, *map(str, counts)])

    parts = [
        f"# Compare - {name_a} vs {name_b}",
        "",
        f"- A: `{a}` ({len(ta)} graded trials)",
        f"- B: `{b}` ({len(tb)} graded trials)",
        f"- {len(qids)} questions; parquet identity "
        f"`{meta_a.get('parquet_identity', '?')}`; grader `{ta[0].get('grader_sha')}`",
        f"- question-clustered bootstrap: {resamples:,} resamples; sign-flip test: "
        f"{permutations:,} Monte Carlo permutations; seed {seed}",
        "- infra-error trials are excluded; model-caused failures count as wrong",
        "",
        "## Review status",
        "",
        _md_table(
            [
                "model",
                "pending review",
                "unparseable adjudications",
                "stale adjudications",
            ],
            review_rows,
        ),
        "",
        (
            f"**{unreviewed} flagged trial(s) are still unreviewed: the rates below "
            "use their automated verdicts** (fill adjudication.json, then compare "
            "again)."
            if unreviewed
            else "Every flagged trial has an applied adjudication."
        ),
        "",
        "## Macro accuracy (equal weight per question)",
        "",
        _md_table(
            ["model", "pass@1", "95% CI", f"pass^{k}"],
            [
                [name_a, _pct(mean(pa)), _ci(ci["a"]), _pct(hat_a)],
                [name_b, _pct(mean(pb)), _ci(ci["b"]), _pct(hat_b)],
            ],
        ),
        "",
        f"pass^{k}: probability that {k} trials of a question all pass (unbiased "
        f"estimator C(c,{k})/C(n,{k}), k = fewest graded trials of any question), "
        "averaged over questions.",
        "",
        "## Paired difference (A - B)",
        "",
        f"- mean per-question difference: {_pp(mean(diffs))}, 95% CI "
        f"[{_pp(ci['diff'][0])}, {_pp(ci['diff'][1])}]",
        f"- sign-flip permutation p = {p:.4f} (two-sided)",
        f"- questions where the models differ: {sum(1 for d in diffs if d)} of "
        f"{len(qids)}",
        "",
    ]
    disagree = [(q, x, y) for q, x, y in zip(qids, pa, pb, strict=True) if x != y]
    if disagree:
        parts += [
            "### Questions where the models disagree",
            "",
            _md_table(
                ["id", "category", "A", "B", "A - B"],
                [
                    [
                        q,
                        BY_ID[q].category,
                        f"{qa[q][0]}/{qa[q][1]}",
                        f"{qb[q][0]}/{qb[q][1]}",
                        _pp(x - y),
                    ]
                    for q, x, y in disagree
                ],
            ),
            "",
        ]
    rows = []
    for cat in CATEGORIES:
        cq = [i for i, q in enumerate(qids) if BY_ID[q].category == cat]
        if not cq:
            continue
        ma = mean(pa[i] for i in cq)
        mb = mean(pb[i] for i in cq)
        rows.append([cat, str(len(cq)), _pct(ma), _pct(mb), _pp(ma - mb)])
    parts += [
        "## Per category (descriptive only; no interval at this n)",
        "",
        _md_table(["category", "n questions", "A", "B", "A - B"], rows),
        "",
        "## Tokens, cost and latency",
        "",
        _md_table(
            [
                "model",
                "trials",
                "tokens",
                "est. USD",
                "tokens per correct",
                "med turn (s)",
                "med model time (s)",
                "med code-judge time (s)",
                "med sandbox time (s)",
            ],
            [_cost_rows(name_a, ta), _cost_rows(name_b, tb)],
        ),
        "",
        "Tokens and cost cover every generate_content call of the graded trials "
        "(chat rounds and code-judge verdicts, retried attempts included; no "
        "grading calls). Latency medians are per trial: model and code-judge "
        "time sum the final attempt's call durations; sandbox time sums query "
        "latencies.",
        "",
    ]
    return "\n".join(parts)


def resolve_pair(first: Path, second: Path | None) -> tuple[Path, Path]:
    """(A, B): two run dirs, or an interleaved parent with exactly two model
    subdirs when only one path is given."""
    if second is not None:
        return first, second
    subs = sorted(p.parent for p in first.glob("*/run_meta.json"))
    if (first / "run_meta.json").exists() or len(subs) != 2:
        raise CompareError(
            f"{first}: give two run dirs, or an interleaved run dir with exactly "
            f"two model subdirectories (found {len(subs)})"
        )
    return subs[0], subs[1]


def default_out(a: Path, b: Path) -> Path:
    """compare-<A>-vs-<B>.md next to A (inside the parent of an interleaved
    run's per-model dirs)."""
    return a.resolve().parent / f"compare-{a.name}-vs-{b.name}.md"

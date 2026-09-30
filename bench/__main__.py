"""Benchmark CLI.

    python -m bench gt     [--parquet P] [--questions SPEC] [--force]
    python -m bench run    [--trials 3] [--questions SPEC] [--parquet P]
                           [--model M | --models A,B] [--seed 0]
                           [--run-dir DIR] [--max-rounds 20] [--allow-drift]
    python -m bench grade   --run-dir DIR [--regrade]
    python -m bench report  --run-dir DIR
    python -m bench audit   --run-dir DIR [--fraction 0.1] [--seed 0]
    python -m bench compare A [B] [--out FILE] [--seed 0] [--resamples 10000]

SPEC = comma-separated question ids ("L01,M04") or a category name
("lookup", "aggregation", "trend", "multistep").

gt is local Polars only (no API call); on the full parquet it takes ~1.5 min
and single queries reach 80-90 GB, so run nothing heavy alongside it. gt and
run hold a host-wide lock ($KINDLING_BENCH_LOCK, default
/tmp/kindling-bench.lock): one bench per host.

run checks the API key (presence only), container runtime and worker image,
computes ground truth, then makes one prompt-guard call per question (recorded
in run_meta) before the trials. --models A,B interleaves (question, trial,
model) in a --seed shuffled order and writes one standard run dir per model
(DIR/<model>/); grade and report accept either DIR/<model> or DIR (every
per-model dir under it).

Resume: re-running `run` with the same --run-dir skips trials whose trace
parses and has infra_error null; corrupt or infra-error traces are moved aside
(*.bak) and re-run. It refuses if the run fingerprint (model, prompt/tool sha,
questions, parquet identity, git sha/dirty, SDK/polars/worker-image versions,
sandbox settings, max rounds) changed, unless --allow-drift. A different model
or parquet identity is refused even with --allow-drift (one dir = one model on
one dataset).
grade makes one blind extractor call (flash-lite, temperature 0; it never
sees the expected value) per non-infra trial and compares the extraction with
ground truth in Python (bench/answers.py holds the per-question units,
precision, aliases and series keys). It refuses traces whose reference_sha no
longer matches questions.py, skips traces already graded by the current
grader_sha (--regrade forces all) and retries grader errors. Trials flagged
for review (a regex/name cross-check disagreeing with the verdict, a right
number failed on its unit, an in-band answer failed only by the head/limit/
slice heuristic) become pending entries in DIR/adjudication.json:
fill `verdict` (pass/fail), `reason` and `adjudicator`; grade and report apply
decided entries over the automated verdict, and grade never overwrites them.

audit samples --fraction of the graded trials (seeded) into DIR/audit.json for
a hand `human_verdict`; re-running keeps filled verdicts, and report shows
judge-human agreement.

compare A B takes two graded single-model run dirs (or one interleaved dir with
exactly two model subdirs) and writes compare-<A>-vs-<B>.md next to A (or
--out): macro pass@1 and pass^k, question-clustered bootstrap 95% CIs, a
sign-flip permutation test on per-question differences, the questions where
the models disagree, and tokens/cost/latency. It refuses runs whose parquet
identity, reference_shas, grader_sha or question sets differ, or that have
ungraded trials.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.config import DEFAULT_CHAT_MODEL, MAX_TOOL_ROUNDS

DEFAULT_PARQUET = "data/mtbs_pix_data.parquet"
DEFAULT_MODEL = DEFAULT_CHAT_MODEL
DEFAULT_TRIALS = 3  # cost-sizing default; the review recommends 5 for comparisons


def _client():
    from app.genai_client import make_client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set (env or .env)")
    return make_client(api_key)


def expand_run_dirs(run_dir: Path) -> list[Path]:
    """A run dir, or (for an interleaved multi-model run) its per-model dirs."""
    if (run_dir / "run_meta.json").exists():
        return [run_dir]
    subs = sorted(p.parent for p in run_dir.glob("*/run_meta.json"))
    if not subs:
        sys.exit(f"{run_dir}: no run_meta.json here or in its subdirectories")
    return subs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gt = sub.add_parser("gt", help="compute/refresh ground truth")
    p_gt.add_argument("--parquet", default=DEFAULT_PARQUET)
    p_gt.add_argument("--questions", default=None)
    p_gt.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run", help="run benchmark trials")
    p_run.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p_run.add_argument("--questions", default=None)
    p_run.add_argument("--parquet", default=DEFAULT_PARQUET)
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument(
        "--models",
        default=None,
        help="comma-separated models to interleave in one run (overrides --model)",
    )
    p_run.add_argument("--seed", type=int, default=0, help="interleave order seed")
    p_run.add_argument("--run-dir", default=None)
    p_run.add_argument("--max-rounds", type=int, default=MAX_TOOL_ROUNDS)
    p_run.add_argument(
        "--allow-drift",
        action="store_true",
        help="resume despite non-model/dataset fingerprint drift (logged in run_meta)",
    )

    p_grade = sub.add_parser("grade", help="grade a run's traces")
    p_grade.add_argument("--run-dir", required=True)
    p_grade.add_argument("--regrade", action="store_true")

    p_report = sub.add_parser("report", help="render report.md for a run")
    p_report.add_argument("--run-dir", required=True)

    p_audit = sub.add_parser("audit", help="sample graded trials for hand audit")
    p_audit.add_argument("--run-dir", required=True)
    p_audit.add_argument("--fraction", type=float, default=0.1)
    p_audit.add_argument("--seed", type=int, default=0)

    p_cmp = sub.add_parser("compare", help="paired comparison of two graded runs")
    p_cmp.add_argument("a", help="run dir A (or an interleaved dir of two models)")
    p_cmp.add_argument("b", nargs="?", default=None, help="run dir B")
    p_cmp.add_argument("--out", default=None, help="markdown output path")
    p_cmp.add_argument("--seed", type=int, default=0)
    p_cmp.add_argument("--resamples", type=int, default=10_000)
    return parser


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()

    if args.command == "gt":
        from bench.ground_truth import precompute
        from bench.questions import select
        from bench.runner import BenchAbort, bench_lock

        try:
            with bench_lock():
                precompute(select(args.questions), args.parquet, force=args.force)
        except BenchAbort as e:
            sys.exit(f"bench gt aborted: {e}")
    elif args.command == "run":
        from bench.runner import main_run

        main_run(args)
    elif args.command == "grade":
        from bench.grading import AdjudicationFileError, StaleTraceError, grade_run
        from bench.judge import ExtractorFatal

        dirs = expand_run_dirs(Path(args.run_dir))
        client = _client()
        try:
            for d in dirs:
                grade_run(d, client, regrade=args.regrade)
        except (StaleTraceError, AdjudicationFileError, ExtractorFatal) as e:
            sys.exit(str(e))
    elif args.command == "report":
        from bench.report import build_report

        for d in expand_run_dirs(Path(args.run_dir)):
            print(build_report(d))
    elif args.command == "audit":
        from bench.audit import AuditFileError, write_audit

        try:
            for d in expand_run_dirs(Path(args.run_dir)):
                print(write_audit(d, fraction=args.fraction, seed=args.seed))
        except (AuditFileError, ValueError) as e:
            sys.exit(str(e))
    elif args.command == "compare":
        from bench.compare import CompareError, compare_runs, default_out, resolve_pair

        try:
            a, b = resolve_pair(Path(args.a), Path(args.b) if args.b else None)
            text = compare_runs(a, b, seed=args.seed, resamples=args.resamples)
        except CompareError as e:
            sys.exit(f"bench compare refused: {e}")
        out = Path(args.out) if args.out else default_out(a, b)
        out.write_text(text)
        print(text)
        print(f"\nwritten to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

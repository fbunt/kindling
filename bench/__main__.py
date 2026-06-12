"""Benchmark CLI.

    python -m bench gt     [--parquet P] [--questions SPEC] [--force]
    python -m bench run    [--trials 3] [--questions SPEC] [--parquet P]
                           [--model M] [--run-dir DIR] [--max-rounds 15]
    python -m bench grade  --run-dir DIR [--regrade]
    python -m bench report --run-dir DIR

SPEC = comma-separated question ids ("L01,M04") or a category name
("lookup", "aggregation", "trend", "multistep"). Re-running `run` with the
same --run-dir resumes: trials whose trace JSON exists are skipped (delete a
trace file to retry it).
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_PARQUET = "data/mtbs_pix_data.parquet"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _client():
    from app.genai_client import make_client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set (env or .env)")
    return make_client(api_key)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gt = sub.add_parser("gt", help="compute/refresh ground truth")
    p_gt.add_argument("--parquet", default=DEFAULT_PARQUET)
    p_gt.add_argument("--questions", default=None)
    p_gt.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run", help="run benchmark trials")
    p_run.add_argument("--trials", type=int, default=3)
    p_run.add_argument("--questions", default=None)
    p_run.add_argument("--parquet", default=DEFAULT_PARQUET)
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--run-dir", default=None)
    p_run.add_argument("--max-rounds", type=int, default=15)

    p_grade = sub.add_parser("grade", help="grade a run's traces")
    p_grade.add_argument("--run-dir", required=True)
    p_grade.add_argument("--regrade", action="store_true")

    p_report = sub.add_parser("report", help="render report.md for a run")
    p_report.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.command == "gt":
        from bench.ground_truth import precompute
        from bench.questions import select

        precompute(select(args.questions), args.parquet, force=args.force)
    elif args.command == "run":
        from bench.runner import run_bench

        asyncio.run(run_bench(args))
    elif args.command == "grade":
        from bench.grading import grade_run

        grade_run(Path(args.run_dir), _client(), regrade=args.regrade)
    elif args.command == "report":
        from bench.report import build_report

        print(build_report(Path(args.run_dir)))


if __name__ == "__main__":
    main()

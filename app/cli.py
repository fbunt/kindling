import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="kindling",
        description="Launch the kindling query interface for a parquet dataset.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the parquet file to query",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Warm sandbox containers to keep ready (default: 2). Requires a "
        "container runtime (podman or docker) and a built kindling-worker image.",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    os.environ["KINDLING_LOG_LEVEL"] = args.log_level
    if args.pool_size is not None:
        os.environ["KINDLING_POOL_SIZE"] = str(args.pool_size)

    from app.query_engine import configure

    configure(args.path)

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

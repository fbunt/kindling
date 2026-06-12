"""In-container sandbox worker.

Runs INSIDE a locked-down Podman container. Speaks newline-delimited JSON over
stdin/stdout to the host pool (app/sandbox/pool.py). Holds a persistent
namespace (a "kernel") so variables defined in one run_query call survive across
calls within a single chat turn.

The container is the security boundary (--network none, --read-only, --cap-drop
ALL, non-root, ephemeral, no secrets), so the code runs with FULL Python
builtins and unrestricted imports — any library in the image is available. There
is no AST/blocklist filtering here; safety comes from the box, not from the code.

This module must NOT import anything from the `app` package: the container image
only ships this file plus the scientific-python stack.
"""

import base64
import io
import json
import os
import sys
import tempfile
import threading

# Point matplotlib at a config/cache dir this process actually owns. Under
# --read-only the root fs is immutable and any image-baked dir under /tmp gets
# copied into the tmpfs root-owned (unwritable by the non-root runner), so we
# carve out a fresh writable dir before importing matplotlib (read on import).
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mpl-")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["figure.figsize"] = (10, 6)
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

MAX_ROWS = 100
QUERY_TIMEOUT = 480  # seconds — soft timeout; preserves the kernel on expiry
# Hard cap on a serialized reply frame. The host reads frames with an 8MB
# asyncio stream limit (pool._STREAM_LIMIT); a longer line raises there and
# surfaces as a bogus "worker died" with a desynced pipe. Cap with margin and
# degrade gracefully here instead. MAX_ROWS bounds rows, not cell width — wide
# string cells or many base64 plot PNGs in one frame can still blow past it.
MAX_REPLY_BYTES = 6 * 1024 * 1024


def _too_large_error(size: int) -> dict:
    return {
        "error": (
            f"Result too large to return (~{size // 1024}KB serialized, cap "
            f"{MAX_REPLY_BYTES // 1024}KB). Return fewer/smaller rows or "
            "fewer plots."
        )
    }


def _fit_reply(output: dict) -> dict:
    """Shrink an oversized run_query reply until it fits MAX_REPLY_BYTES.

    Halves tabular rows first (marking truncation); if the frame is still too
    big (huge cells, or plots dominating), replaces it with an actionable
    error the model can react to."""
    size = len(json.dumps(output, default=str))
    while size > MAX_REPLY_BYTES and isinstance(output.get("data"), list):
        rows = output["data"]
        if len(rows) <= 1:
            break
        output["data"] = rows[: len(rows) // 2]
        output["truncated"] = True
        output["note"] = "rows truncated to fit the reply size cap"
        size = len(json.dumps(output, default=str))
    if size > MAX_REPLY_BYTES:
        return _too_large_error(size)
    return output


def _build_lazyframe(path):
    """Reproduce app.query_engine._build_lazyframe VERBATIM.

    Any divergence from the host expression changes query semantics relative to
    what the eval suite validates.
    """
    return pl.scan_parquet(path).drop("__null_dask_index__")


_PARQUET_PATH = os.environ["KINDLING_PARQUET_PATH"]
LF = _build_lazyframe(_PARQUET_PATH)


class _Namespace(dict):
    """Tracks whether `result` was (re)assigned during the current call so
    `result` can persist (be readable) across calls within a turn while we still
    detect whether THIS call produced an output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_assigned = False

    def __setitem__(self, key, value):
        if key == "result":
            self.result_assigned = True
        super().__setitem__(key, value)


def build_namespace() -> _Namespace:
    """Create the persistent execution namespace.

    No "__builtins__" key is set, so exec() injects the FULL builtins — the
    container is the boundary, not a restricted builtins table. Only two names are
    preloaded: `lf` (the bound LazyFrame, which can't be imported) and `pl`
    (polars, used in every query). Every other library in the image — numpy,
    pandas, matplotlib, seaborn, scipy, scikit-learn, xgboost — is importable on
    demand; the model imports what it uses."""
    return _Namespace(
        {
            "pl": pl,
            "lf": LF.clone(),
        }
    )


def _capture_plots() -> list[str]:
    """Encode every open matplotlib figure as a base64 PNG and close it.

    Mirrors app/query_engine.py plot capture (bbox_inches/dpi) but returns bytes
    over the protocol instead of writing files; the host materializes them.
    """
    out = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def handle_run_query(code: str, namespace: dict) -> dict:
    """Execute query code with full builtins. No validation here — the container
    is the security boundary."""
    # Keep any prior `result` readable; reset the flag to detect this call's output.
    namespace.result_assigned = False

    result_box = [None]

    def _run():
        try:
            # Single dict so globals == locals: nested scopes (lambdas,
            # comprehensions, def'd functions) resolve free vars through globals.
            exec(code, namespace)
        except Exception as e:
            result_box[0] = {"error": f"Execution error: {type(e).__name__}: {e}"}

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=QUERY_TIMEOUT)
    if t.is_alive():
        # Soft timeout: abandon the query but keep the kernel. The leaked daemon
        # thread dies when the host kills this container at turn end.
        plt.close("all")
        return {"error": f"Query timed out (exceeded {QUERY_TIMEOUT} seconds)"}
    if result_box[0] is not None:
        plt.close("all")
        return result_box[0]

    plots = _capture_plots()
    # Only this call's freshly-assigned result counts as output.
    result = namespace.get("result") if namespace.result_assigned else None
    if result is None and not plots:
        return {"error": "No result produced. Code must assign to `result`."}

    try:
        output: dict = {}
        if result is not None:
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            if isinstance(result, pl.DataFrame):
                total_rows = len(result)
                if total_rows > MAX_ROWS:
                    result = result.head(MAX_ROWS)
                output["data"] = result.to_dicts()
                output["total_rows"] = total_rows
                output["truncated"] = total_rows > MAX_ROWS
            else:
                output["data"] = str(result)
        if plots:
            output["plots"] = plots  # base64 PNG strings; host materializes
            if "data" not in output:
                output["data"] = "Plot(s) generated successfully."
        return _fit_reply(output)
    except Exception as e:
        return {"error": f"Result processing error: {e}"}


def main() -> None:
    # The protocol owns fd 1. Capture a private handle to the real stdout, then
    # redirect Python-level stdout to stderr so stray user/library writes —
    # including `print` in query code — can't corrupt the JSON stream.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout = sys.stderr

    namespace = build_namespace()

    def reply(obj: dict) -> None:
        frame = json.dumps(obj, default=str)
        # Belt-and-braces: _fit_reply already bounds run_query output, but no
        # frame may ever exceed the host's stream limit (it would desync the
        # pipe), so guard every reply path.
        if len(frame) > MAX_REPLY_BYTES:
            frame = json.dumps(_too_large_error(len(frame)))
        proto.write(frame)
        proto.write("\n")
        proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            reply({"error": "malformed request"})
            continue
        op = req.get("op")
        if op == "ping":
            reply({"op": "pong"})
        elif op == "run_query":
            reply(handle_run_query(req.get("code", ""), namespace))
        else:
            reply({"error": f"unknown op: {op!r}"})


if __name__ == "__main__":
    main()

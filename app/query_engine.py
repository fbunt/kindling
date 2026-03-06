import ast
import logging
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

PARQUET_PATH = Path("data/mtbs_pix_data.parquet")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
MAX_ROWS = 100
QUERY_TIMEOUT = 30  # seconds

_plot_counter = 0
_zombie_threads: list[threading.Thread] = []
logger = logging.getLogger(__name__)

LF = pl.scan_parquet(PARQUET_PATH)

# Pre-compute schema info once at module load
_SCHEMA = LF.collect_schema()
_SCHEMA_INFO = {name: str(dtype) for name, dtype in _SCHEMA.items()}


def get_dataset_info() -> dict:
    """Return schema, row count, and sample rows for Gemini context."""
    sample = LF.head(5).collect()
    return {
        "columns": _SCHEMA_INFO,
        "row_count_approx": "~79 million",
        "sample_rows": sample.to_dicts(),
    }


# --- AST Validation ---

_FORBIDDEN_NODE_TYPES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Delete,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
    ast.ClassDef, ast.Yield, ast.YieldFrom,
)

_FORBIDDEN_BUILTINS = {
    "open", "exec", "eval", "compile", "__import__", "input",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "breakpoint", "exit", "quit", "help", "print",
    "memoryview", "type", "super", "classmethod", "staticmethod",
    "property",
}

_FORBIDDEN_STRING_PATTERNS = {"__", "import ", "eval(", "exec(", "open("}


class ValidationError(Exception):
    pass


def validate_code(code: str) -> None:
    """Parse and validate polars query code via AST inspection."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}")

    for node in ast.walk(tree):
        # Reject forbidden node types
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise ValidationError(
                f"Forbidden statement: {type(node).__name__}"
            )

        # Reject calls to forbidden builtins
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BUILTINS:
                raise ValidationError(f"Forbidden call: {func.id}()")

        # Reject forbidden names used as decorators
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in _FORBIDDEN_BUILTINS:
                    raise ValidationError(f"Forbidden decorator: {dec.id}")

        # Reject references to dunder names (e.g. __builtins__)
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            raise ValidationError(f"Forbidden name: {node.id}")

        # Reject dunder attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise ValidationError(
                    f"Forbidden attribute access: {node.attr}"
                )

        # Reject suspicious string literals
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.lower()
            for pat in _FORBIDDEN_STRING_PATTERNS:
                if pat in val:
                    raise ValidationError(
                        f"Forbidden string content: '{pat}'"
                    )



def execute_query(code: str) -> dict:
    """Validate and execute polars query code. Returns result dicts or error."""
    try:
        validate_code(code)
    except ValidationError as e:
        return {"error": str(e)}

    namespace = {"pl": pl, "lf": LF.clone(), "plt": plt, "sns": sns}
    restricted_builtins = {
        "True": True, "False": False, "None": None,
        "len": len, "range": range, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "bool": bool, "abs": abs,
        "min": min, "max": max, "sum": sum, "round": round, "sorted": sorted,
    }
    global_ns = {"__builtins__": restricted_builtins}

    # Check for and clean up zombie threads from previous timed-out queries
    still_alive = [t for t in _zombie_threads if t.is_alive()]
    if still_alive:
        logger.warning(f"{len(still_alive)} zombie query thread(s) still running")
    _zombie_threads.clear()
    _zombie_threads.extend(still_alive)

    result_box = [None]

    def _run():
        try:
            exec(code, global_ns, namespace)
        except Exception as e:
            result_box[0] = {"error": f"Execution error: {type(e).__name__}: {e}"}

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=QUERY_TIMEOUT)
    if t.is_alive():
        _zombie_threads.append(t)
        logger.warning("Query timed out, thread still running in background")
        return {"error": "Query timed out (exceeded 30 seconds)"}
    if result_box[0] is not None:
        return result_box[0]

    # Capture any matplotlib plots
    global _plot_counter
    plot_urls = []
    for fig_num in plt.get_fignums():
        _plot_counter += 1
        filename = f"plot-{_plot_counter:03d}.png"
        fig = plt.figure(fig_num)
        fig.savefig(PLOTS_DIR / filename, bbox_inches="tight", dpi=150)
        plt.close(fig)
        plot_urls.append(f"/plots/{filename}")

    result = namespace.get("result")
    if result is None and not plot_urls:
        return {"error": "No result produced. Code must assign to `result`."}

    try:
        output = {}
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
        if plot_urls:
            output["plots"] = plot_urls
            if "data" not in output:
                output["data"] = "Plot(s) generated successfully."
        return output
    except Exception as e:
        return {"error": f"Result processing error: {e}"}

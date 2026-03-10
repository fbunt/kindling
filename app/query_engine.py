import ast
import logging
import threading
import time
from pathlib import Path

import math
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.figsize"] = (10, 6)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl

PARQUET_PATH = Path("data/mtbs_pix_data.parquet")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
MAX_ROWS = 100
QUERY_TIMEOUT = 300  # seconds

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
    "getattr", "setattr", "delattr", "globals", "vars",
    "breakpoint", "exit", "quit", "help", "print",
    "memoryview", "type", "super", "classmethod", "staticmethod",
    "property",
}

_FORBIDDEN_ATTRIBUTES = {
    "write_parquet", "write_csv", "write_json", "write_ipc", "write_excel",
    "write_ndjson", "write_avro", "write_clipboard", "write_database",
    "sink_parquet", "sink_csv", "sink_ipc", "sink_ndjson",
    "savefig",
    # Pandas write methods (reachable via .to_pandas())
    "to_csv", "to_excel", "to_parquet", "to_json", "to_sql",
    "to_hdf", "to_feather", "to_stata", "to_pickle", "to_clipboard",
    "to_latex", "to_gbq",
    # Numpy file I/O
    "save", "savez", "savez_compressed", "savetxt", "load", "fromfile",
}

_FORBIDDEN_STRING_PATTERNS = {"__", "eval(", "exec(", "open("}


class ValidationError(Exception):
    pass


# Imports the model is allowed to include (already in the execution namespace)
_ALLOWED_IMPORTS = {
    # import X as Y — keyed by module name, value is expected alias (or None for no alias)
    "numpy": "np",
    "polars": "pl",
    "math": None,
    "seaborn": "sns",
}

_ALLOWED_IMPORT_FROMS = {
    # (module, name, alias) — matches "from matplotlib import pyplot as plt"
    ("matplotlib", "pyplot", "plt"),
}


def _strip_allowed_imports(tree: ast.Module) -> ast.Module:
    """Remove import statements that match the allowed set from the AST."""
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            # Handle "import numpy as np", "import matplotlib.pyplot as plt", etc.
            remaining = []
            for alias in node.names:
                module = alias.name
                asname = alias.asname
                # Check plain module imports: import numpy as np
                if module in _ALLOWED_IMPORTS and (asname == _ALLOWED_IMPORTS[module] or (asname is None and _ALLOWED_IMPORTS[module] is None)):
                    continue
                # Check dotted imports: import matplotlib.pyplot as plt
                if module == "matplotlib.pyplot" and asname == "plt":
                    continue
                remaining.append(alias)
            if remaining:
                node.names = remaining
                new_body.append(node)
        elif isinstance(node, ast.ImportFrom):
            # Handle "from matplotlib import pyplot as plt"
            remaining = []
            for alias in node.names:
                if (node.module, alias.name, alias.asname) in _ALLOWED_IMPORT_FROMS:
                    continue
                remaining.append(alias)
            if remaining:
                node.names = remaining
                new_body.append(node)
        else:
            new_body.append(node)
    tree.body = new_body
    return tree


def validate_code(code: str) -> str:
    """Parse, strip allowed imports, and validate query code. Returns cleaned code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}")

    tree = _strip_allowed_imports(tree)

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

        # Reject dunder and write-related attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise ValidationError(
                    f"Forbidden attribute access: {node.attr}"
                )
            if node.attr in _FORBIDDEN_ATTRIBUTES:
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

    return ast.unparse(tree)



def execute_query(code: str) -> dict:
    """Validate and execute polars query code. Returns result dicts or error."""
    try:
        code = validate_code(code)
    except ValidationError as e:
        logger.warning(f"Validation rejected code: {e}\n{code}")
        return {"error": str(e)}

    namespace = {"pl": pl, "np": np, "math": math, "lf": LF.clone(), "plt": plt, "sns": sns}
    restricted_builtins = {
        "True": True, "False": False, "None": None,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float,
        "hasattr": hasattr, "int": int, "len": len, "list": list, "locals": locals,
        "map": map, "max": max, "min": min, "pow": pow, "range": range,
        "round": round, "set": set, "slice": slice, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
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
            logger.warning(f"Runtime error in query: {type(e).__name__}: {e}\n{code}")
            result_box[0] = {"error": f"Execution error: {type(e).__name__}: {e}"}

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=QUERY_TIMEOUT)
    if t.is_alive():
        _zombie_threads.append(t)
        logger.warning("Query timed out, thread still running in background")
        return {"error": f"Query timed out (exceeded {QUERY_TIMEOUT} seconds)"}
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
        plot_urls.append(f"/plots/{filename}?t={int(time.time())}")

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

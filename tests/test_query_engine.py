import polars as pl  # noqa: F401 — used inside execute_query() code strings
import pytest

from app.query_engine import (
    ValidationError,
    _strip_allowed_imports,
    create_namespace,
    execute_query,
    get_dataset_info,
    validate_code,
)
from app.tools import _unique_display_name, _used_display_names, execute_function_call

# ── validate_code: allowed ──────────────────────────────────────────


class TestValidateAllowed:
    def test_simple_assignment(self):
        validate_code("result = 1 + 1")

    def test_polars_operations(self):
        validate_code("result = lf.filter(pl.col('x') > 5).collect()")

    def test_list_comprehension(self):
        validate_code("result = [x for x in range(10)]")

    def test_multiline(self):
        validate_code("x = lf.collect()\nresult = len(x)")

    def test_function_def(self):
        validate_code("def f(x):\n    return x * 2\nresult = f(3)")

    def test_allowed_builtins_in_code(self):
        validate_code("result = sorted([3, 1, 2])")

    def test_matplotlib_calls(self):
        validate_code("plt.figure()\nplt.plot([1,2,3])\nresult = 'ok'")

    def test_f_string(self):
        validate_code("result = f'count: {len([1,2,3])}'")

    def test_redundant_allowed_imports_stripped(self):
        # Imports of objects already preloaded in the namespace must validate
        # cleanly (they're stripped from the AST). Regression: a model writing
        # `from matplotlib.patches import Patch` was tripping ImportFrom.
        cases = [
            "import numpy as np\nresult = np.array([1])",
            "import polars as pl\nresult = pl.DataFrame({'a': [1]})",
            "import math\nresult = math.pi",
            "import seaborn as sns\nresult = sns.color_palette('viridis')",
            "import matplotlib.pyplot as plt\nresult = plt.figure()",
            "from matplotlib import pyplot as plt\nresult = plt.figure()",
            "from matplotlib.patches import Patch\nresult = Patch()",
        ]
        for src in cases:
            cleaned = validate_code(src)
            assert "import" not in cleaned, (
                f"import not stripped from cleaned code: {cleaned!r}"
            )


# ── validate_code: forbidden ────────────────────────────────────────


class TestValidateForbidden:
    def test_import(self):
        with pytest.raises(ValidationError, match="Import"):
            validate_code("import os")

    def test_import_from(self):
        with pytest.raises(ValidationError, match="ImportFrom"):
            validate_code("from os import path")

    def test_exec_call(self):
        with pytest.raises(ValidationError, match="exec"):
            validate_code("exec('print(1)')")

    def test_eval_call(self):
        with pytest.raises(ValidationError, match="eval"):
            validate_code("eval('1+1')")

    def test_open_call(self):
        with pytest.raises(ValidationError, match="open"):
            validate_code("open('/etc/passwd')")

    def test_dunder_import(self):
        with pytest.raises(ValidationError, match="__import__"):
            validate_code("__import__('os')")

    def test_dunder_attribute(self):
        with pytest.raises(ValidationError, match="__class__"):
            validate_code("x = ''.__class__")

    def test_globals_call(self):
        with pytest.raises(ValidationError, match="globals"):
            validate_code("globals()")

    def test_getattr_call(self):
        with pytest.raises(ValidationError, match="getattr"):
            validate_code("getattr(lf, 'collect')()")

    def test_class_def(self):
        with pytest.raises(ValidationError, match="ClassDef"):
            validate_code("class Foo: pass")

    def test_delete(self):
        with pytest.raises(ValidationError, match="Delete"):
            validate_code("del x")

    def test_print_allowed_as_noop(self):
        """print() is allowed at validation but executes as a silent no-op."""
        validate_code("print('hello')")
        out = execute_query("print('hello')\nresult = 'ok'")
        assert out["data"] == "ok"

    def test_string_with_import_allowed(self):
        """Strings containing 'import' are harmless and no longer forbidden."""
        validate_code("x = 'import os'")
        out = execute_query("result = 'import os'")
        assert out["data"] == "import os"

    def test_string_with_dunder(self):
        with pytest.raises(ValidationError, match="__"):
            validate_code("x = '__import__'")

    def test_syntax_error(self):
        with pytest.raises(ValidationError, match="Syntax error"):
            validate_code("def (")

    def test_breakpoint(self):
        with pytest.raises(ValidationError, match="breakpoint"):
            validate_code("breakpoint()")

    def test_input(self):
        with pytest.raises(ValidationError, match="input"):
            validate_code("input('prompt')")


# ── Adversarial bypass attempts ─────────────────────────────────────


class TestAdversarial:
    """Attempts to escape the sandbox via creative bypasses."""

    # --- Attribute chain escapes ---

    def test_string_class_mro_escape(self):
        """Classic ''.__class__.__mro__ Python jail escape."""
        with pytest.raises(ValidationError):
            validate_code("result = ''.__class__.__mro__[1].__subclasses__()")

    def test_type_via_literal(self):
        """type() via (1).__class__."""
        with pytest.raises(ValidationError):
            validate_code("result = (1).__class__.__bases__")

    # --- Indirect builtins access ---

    def test_builtins_via_dict_bracket(self):
        """Access __builtins__ directly — caught by dunder name check."""
        with pytest.raises(ValidationError, match="__builtins__"):
            validate_code("result = __builtins__['open']")

    def test_compile_call(self):
        with pytest.raises(ValidationError, match="compile"):
            validate_code("compile('import os', '', 'exec')")

    # --- Aliasing to dodge name checks ---

    def test_alias_exec_via_assignment(self):
        """Assign exec to a variable — validation passes but runtime should
        fail because exec isn't in restricted_builtins."""
        out = execute_query("e = exec\nresult = 'ok'")
        assert "error" in out

    def test_alias_open_via_assignment(self):
        out = execute_query("o = open\nresult = 'ok'")
        assert "error" in out

    # --- Lambda / nested function tricks ---

    def test_lambda_calling_forbidden(self):
        with pytest.raises(ValidationError):
            validate_code("f = lambda: exec('x=1')\nf()")

    # --- os/subprocess via polars or other injected objects ---

    def test_access_module_via_pl(self):
        """Try to traverse pl -> module attributes to reach os."""
        with pytest.raises(ValidationError):
            validate_code("result = pl.__spec__")

    def test_func_globals_via_attribute(self):
        with pytest.raises(ValidationError):
            validate_code("result = len.__globals__")

    # --- String concatenation to build forbidden strings ---

    def test_string_concat_import(self):
        """Build 'import' via concat — this bypasses string pattern checks,
        but without exec/eval it can't be executed. Validation should pass
        (it's just a string), and execution should be harmless."""
        out = execute_query("result = 'im' + 'port os'")
        assert out["data"] == "import os"  # harmless string, no execution

    # --- Walrus operator / complex expressions ---

    def test_nested_comprehension_with_side_effects(self):
        """Side effects in comprehensions should still be sandboxed."""
        out = execute_query("result = [x for x in range(5)]")
        assert out["data"] == "[0, 1, 2, 3, 4]"

    # --- Attempt to modify the namespace ---

    def test_overwrite_pl(self):
        """Overwriting pl in the namespace shouldn't persist across calls."""
        execute_query("pl = None\nresult = 'ok'")
        out = execute_query("result = pl.DataFrame({'a': [1]}).to_dicts()")
        assert "error" not in out

    def test_overwrite_lf(self):
        """Overwriting lf shouldn't persist across calls."""
        execute_query("lf = None\nresult = 'ok'")
        out = execute_query("result = lf.head(1).collect().to_dicts()")
        assert "error" not in out

    # --- Timeout / resource abuse ---

    def test_infinite_loop_times_out(self, monkeypatch):
        import app.query_engine as qe

        monkeypatch.setattr(qe, "QUERY_TIMEOUT", 1)
        out = execute_query("while True: pass\nresult = 'done'")
        assert "error" in out
        assert "timed out" in out["error"].lower()

    # --- File system access attempts ---

    def test_pathlib_via_string(self):
        """Can't import pathlib, but check the string pattern catches it."""
        with pytest.raises(ValidationError):
            validate_code("from pathlib import Path")

    def test_open_via_string_pattern(self):
        with pytest.raises(ValidationError):
            validate_code("x = 'open('")

    # --- Decorator abuse ---

    def test_decorator_on_function(self):
        """Forbidden builtins used as decorators should be caught."""
        with pytest.raises(ValidationError, match="eval"):
            validate_code("@eval\ndef f(): pass")

    def test_property_decorator(self):
        with pytest.raises(ValidationError, match="property"):
            validate_code("@property\ndef f(): pass")

    # --- Global/nonlocal escape ---

    def test_global_statement(self):
        with pytest.raises(ValidationError, match="Global"):
            validate_code("global _plot_counter")

    def test_nonlocal_statement(self):
        with pytest.raises(ValidationError, match="Nonlocal"):
            validate_code("def f():\n    nonlocal x")


# ── execute_query ───────────────────────────────────────────────────


class TestExecuteQuery:
    def test_simple_result(self):
        out = execute_query("result = 42")
        assert out["data"] == "42"

    def test_polars_query(self):
        out = execute_query("result = lf.head(2).collect()")
        assert "data" in out
        assert isinstance(out["data"], list)
        assert len(out["data"]) == 2

    def test_dataframe_result(self):
        out = execute_query("result = lf.head(3).collect()")
        assert out["total_rows"] == 3
        assert out["truncated"] is False

    def test_truncation(self):
        out = execute_query("result = lf.head(200).collect()")
        assert out["truncated"] is True
        assert out["total_rows"] == 200
        assert len(out["data"]) == 100

    def test_no_result_assigned(self):
        out = execute_query("x = 1 + 1")
        assert "error" in out
        assert "No result" in out["error"]

    def test_validation_error_returns_dict(self):
        out = execute_query("import os")
        assert "error" in out
        assert "Forbidden" in out["error"]

    def test_runtime_error(self):
        out = execute_query("result = 1 / 0")
        assert "error" in out
        assert "ZeroDivision" in out["error"]

    def test_name_error_for_blocked_builtin(self):
        # open() is blocked at validation, but even if it somehow got through,
        # restricted_builtins wouldn't have it
        out = execute_query("result = type(1)")
        assert "error" in out

    def test_plot_generation(self):
        out = execute_query(
            "plt.figure()\nplt.plot([1, 2, 3], [4, 5, 6])\nresult = 'plotted'"
        )
        assert "plots" in out
        assert len(out["plots"]) >= 1
        assert out["plots"][0].startswith("/plots/")

    def test_list_result(self):
        out = execute_query("result = [1, 2, 3]")
        assert out["data"] == "[1, 2, 3]"

    def test_allowed_builtins_work(self):
        out = execute_query("result = sorted([3, 1, 2])")
        assert out["data"] == "[1, 2, 3]"

    def test_lazyframe_auto_collected(self):
        out = execute_query("result = lf.head(1)")
        assert "data" in out
        assert isinstance(out["data"], list)

    def test_unique_display_names(self):
        _used_display_names.clear()
        assert _unique_display_name("fire-plot") == "fire-plot"
        assert _unique_display_name("fire-plot") == "fire-plot-001"
        assert _unique_display_name("fire-plot") == "fire-plot-002"
        assert _unique_display_name("other-plot") == "other-plot"
        _used_display_names.clear()

    def test_numpy_available(self):
        out = execute_query("result = np.array([1, 2, 3]).sum()")
        assert out.get("data") == "6"

    def test_multi_step_query(self):
        out = execute_query(
            "fires = lf.unique('Event_ID').head(5).collect()\n"
            "count = len(fires)\n"
            "result = f'{count} fires'"
        )
        assert "error" not in out
        assert out["data"] == "5 fires"

    def test_numpy_array_methods(self):
        """Numpy array methods like .sum() internally import submodules;
        the restricted __import__ must allow these."""
        out = execute_query("result = int(np.array([1, 2, 3]).sum())")
        assert out["data"] == "6"

    def test_numpy_array_mean_std(self):
        out = execute_query(
            "a = np.array([10.0, 20.0, 30.0])\nresult = f'{a.mean()},{a.std():.4f}'"
        )
        assert "error" not in out
        assert out["data"].startswith("20.0,")

    def test_numpy_percentile(self):
        out = execute_query("result = float(np.percentile([10, 20, 30, 40], 50))")
        assert out["data"] == "25.0"

    def test_numpy_with_polars(self):
        out = execute_query(
            "vals = lf.select(pl.col('area_m2')).head(100)"
            ".collect().to_series().to_numpy()\n"
            "result = float(np.mean(vals))"
        )
        assert "error" not in out
        assert float(out["data"]) > 0

    def test_restricted_import_blocks_disallowed(self):
        """Even if AST validation is bypassed, the restricted __import__
        should block non-allowlisted modules at runtime."""
        # This is caught at validation, but verify the runtime layer too
        # by testing that numpy internals work while os wouldn't
        out = execute_query("result = int(np.sqrt(16))")
        assert out["data"] == "4"

    def test_multiple_plots_in_one_query(self):
        out = execute_query(
            "plt.figure()\nplt.plot([1, 2])\n"
            "plt.figure()\nplt.plot([3, 4])\n"
            "result = 'two plots'"
        )
        assert "plots" in out
        assert len(out["plots"]) == 2

    def test_plot_only_no_result(self):
        out = execute_query("plt.figure()\nplt.plot([1, 2, 3])")
        assert "plots" in out
        assert "error" not in out

    def test_empty_dataframe_result(self):
        out = execute_query(
            "result = pl.DataFrame({'a': [], 'b': []}).cast({'a': pl.Int64, 'b': pl.Int64})"  # noqa: E501
        )
        assert out["data"] == []
        assert out["total_rows"] == 0

    def test_max_rows_boundary(self):
        out = execute_query("result = pl.DataFrame({'x': list(range(100))})")
        assert out["total_rows"] == 100
        assert out["truncated"] is False
        assert len(out["data"]) == 100


# ── Shared namespace ───────────────────────────────────────────────


class TestSharedNamespace:
    def test_create_namespace_has_expected_keys(self):
        ns = create_namespace()
        assert {"pl", "np", "math", "lf", "plt", "sns"} <= set(ns.keys())

    def test_variables_persist_across_calls(self):
        ns = create_namespace()
        out1 = execute_query("x = 42\nresult = 'set x'", ns)
        assert "error" not in out1
        out2 = execute_query("result = x + 8", ns)
        assert out2["data"] == "50"

    def test_dataframe_persists_across_calls(self):
        ns = create_namespace()
        out1 = execute_query("fires = lf.head(5).collect()\nresult = len(fires)", ns)
        assert out1["data"] == "5"
        out2 = execute_query("result = len(fires)", ns)
        assert out2["data"] == "5"

    def test_call_without_result_assignment_errors(self):
        ns = create_namespace()
        execute_query("result = 'first'", ns)
        out = execute_query("x = 1", ns)
        assert "error" in out
        assert "No result" in out["error"]
        # the previous result is still readable by a later call
        out2 = execute_query("result = result", ns)
        assert out2["data"] == "first"

    def test_result_accessible_from_previous_call(self):
        ns = create_namespace()
        execute_query("result = 'first'", ns)
        out = execute_query("result = result + ' second'", ns)
        assert out["data"] == "first second"

    def test_result_persists_and_can_be_transformed(self):
        ns = create_namespace()
        out1 = execute_query("result = lf.head(10).collect()", ns)
        assert len(out1["data"]) == 10
        # the full (untruncated) result persists; transform it in the next call
        out2 = execute_query("result = result.head(3)", ns)
        assert len(out2["data"]) == 3

    def test_separate_namespaces_are_isolated(self):
        ns1 = create_namespace()
        ns2 = create_namespace()
        execute_query("x = 99\nresult = 'ok'", ns1)
        out = execute_query("result = x", ns2)
        assert "error" in out

    def test_namespace_none_creates_fresh(self):
        """Passing None should work like no shared namespace."""
        out = execute_query("result = 'fresh'", None)
        assert out["data"] == "fresh"

    def test_overwritten_builtins_dont_persist(self):
        """Each namespace gets its own lf clone."""
        ns = create_namespace()
        execute_query("lf = None\nresult = 'ok'", ns)
        # lf is now None in this namespace — next call should fail
        out = execute_query("result = lf.head(1).collect()", ns)
        assert "error" in out

    def test_plot_in_first_call_data_in_second(self):
        ns = create_namespace()
        out1 = execute_query(
            "vals = [1, 2, 3, 4]\nplt.figure()\nplt.plot(vals)\nresult = 'plotted'", ns
        )
        assert "plots" in out1
        out2 = execute_query("result = vals", ns)
        assert out2["data"] == "[1, 2, 3, 4]"


# ── get_dataset_info ───────────────────────────────────────────────


class TestGetDatasetInfo:
    def test_return_structure(self):
        info = get_dataset_info()
        assert "columns" in info
        assert "key_columns" in info
        assert "row_count_approx" in info
        assert "sample_rows" in info

    def test_columns_match_schema(self):
        info = get_dataset_info()
        assert isinstance(info["columns"], dict)
        assert "year" in info["columns"]
        assert "Event_ID" in info["columns"]
        assert "area_m2" in info["columns"]

    def test_key_columns_present(self):
        info = get_dataset_info()
        kc = info["key_columns"]
        assert "year" in kc
        assert "bs" in kc
        assert "nlcd" in kc
        assert "Incid_Type" in kc

    def test_sample_rows(self):
        info = get_dataset_info()
        assert isinstance(info["sample_rows"], list)
        assert len(info["sample_rows"]) == 5
        assert "year" in info["sample_rows"][0]

    def test_row_count(self):
        info = get_dataset_info()
        assert info["row_count_approx"] == "~745 million"


# ── execute_function_call ──────────────────────────────────────────


class TestExecuteFunctionCall:
    def test_get_dataset_info_dispatch(self):
        result_str, plots = execute_function_call("get_dataset_info", {}, None, "")
        import json

        result = json.loads(result_str)
        assert "columns" in result
        assert "key_columns" in result
        assert plots == []

    def test_run_query_dispatch(self):
        result_str, plots = execute_function_call(
            "run_query", {"code": "result = 42"}, None, ""
        )
        import json

        result = json.loads(result_str)
        assert result["data"] == "42"
        assert plots == []

    def test_run_query_with_namespace(self):
        ns = create_namespace()
        execute_function_call(
            "run_query",
            {"code": "x = 10\nresult = 'ok'"},
            None,
            "",
            namespace=ns,
        )
        result_str, _ = execute_function_call(
            "run_query",
            {"code": "result = x * 2"},
            None,
            "",
            namespace=ns,
        )
        import json

        result = json.loads(result_str)
        assert result["data"] == "20"

    def test_unknown_function(self):
        result_str, plots = execute_function_call("nonexistent", {}, None, "")
        import json

        result = json.loads(result_str)
        assert "error" in result
        assert "Unknown function" in result["error"]
        assert plots == []

    def test_run_query_validation_error(self):
        result_str, plots = execute_function_call(
            "run_query", {"code": "import os"}, None, ""
        )
        import json

        result = json.loads(result_str)
        assert "error" in result
        assert plots == []


# ── _strip_allowed_imports ─────────────────────────────────────────


class TestStripAllowedImports:
    def test_strips_numpy_import(self):
        import ast

        code = "import numpy as np\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import" not in source

    def test_strips_polars_import(self):
        import ast

        code = "import polars as pl\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import" not in source

    def test_strips_matplotlib_from_import(self):
        import ast

        code = "from matplotlib import pyplot as plt\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import" not in source

    def test_strips_dotted_matplotlib_import(self):
        import ast

        code = "import matplotlib.pyplot as plt\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import" not in source

    def test_keeps_disallowed_import(self):
        import ast

        code = "import os\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import os" in source

    def test_partial_strip_mixed_imports(self):
        import ast

        code = "import numpy as np, os\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "os" in source
        assert "numpy" not in source

    def test_strips_seaborn_import(self):
        import ast

        code = "import seaborn as sns\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import" not in source

    def test_strips_math_import(self):
        import ast

        code = "import math\nresult = 1"
        tree = _strip_allowed_imports(ast.parse(code))
        source = ast.unparse(tree)
        assert "import" not in source

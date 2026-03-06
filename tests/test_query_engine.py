import pytest
import polars as pl

from app.query_engine import validate_code, execute_query, ValidationError


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

    def test_print(self):
        with pytest.raises(ValidationError, match="print"):
            validate_code("print('hello')")

    def test_string_with_import(self):
        with pytest.raises(ValidationError, match="import"):
            validate_code("x = 'import os'")

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

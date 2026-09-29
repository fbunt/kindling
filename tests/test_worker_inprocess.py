"""In-process tests of the sandbox worker's result shaping.

The worker imports matplotlib at module load and these cases need pandas; both
are worker-only deps (`uv sync --extra worker`), so this module skips on the
host/CI env. The container path is covered in test_sandbox_container.py.
"""

import importlib
import sys

import polars as pl
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "w.parquet"
    pl.DataFrame({"__null_dask_index__": [0, 1], "year": [1990, 1991]}).write_parquet(
        path
    )
    mp = pytest.MonkeyPatch()
    mp.setenv("KINDLING_PARQUET_PATH", str(path))
    sys.modules.pop("app.sandbox.worker", None)
    try:
        yield importlib.import_module("app.sandbox.worker")
    finally:
        sys.modules.pop("app.sandbox.worker", None)
        mp.undo()


def _run(worker, code):
    return worker.handle_run_query(code, worker.build_namespace())


def _assert_truncated(out, total):
    assert "error" not in out, out
    assert isinstance(out["data"], list) and len(out["data"]) == 100
    assert out["total_rows"] == total
    assert out["truncated"] is True
    assert out["note"].startswith(f"Showing the first 100 of {total} rows")


def test_polars_series_truncated(worker):
    out = _run(worker, "result = pl.Series('x', range(250))")
    _assert_truncated(out, 250)
    assert out["data"][0] == {"x": 0}


def test_pandas_dataframe_truncated(worker):
    out = _run(worker, "import pandas as pd\nresult = pd.DataFrame({'x': range(150)})")
    _assert_truncated(out, 150)
    assert out["data"][0] == {"x": 0}  # default index dropped


def test_pandas_series_keeps_index(worker):
    out = _run(
        worker,
        "import pandas as pd\n"
        "df = pd.DataFrame({'k': [i % 120 for i in range(240)], 'v': 1})\n"
        "result = df.groupby('k')['v'].sum()",
    )
    _assert_truncated(out, 120)
    assert out["data"][0] == {"k": 0, "v": 2}


def test_lazyframe_collected(worker):
    out = _run(worker, "result = lf")
    assert out["data"] == [{"year": 1990}, {"year": 1991}]


def test_small_frame_has_no_note(worker):
    out = _run(worker, "result = pl.DataFrame({'x': [1, 2, 3]})")
    assert out["total_rows"] == 3
    assert out["truncated"] is False
    assert "note" not in out


def test_fit_reply_note_matches_rows_returned(worker, monkeypatch):
    monkeypatch.setattr(worker, "MAX_REPLY_BYTES", 2000)
    out = worker._fit_reply(
        {"data": [{"s": "x" * 100}] * 100, "total_rows": 400, "truncated": True}
    )
    n = len(out["data"])
    assert n < 100
    assert out["total_rows"] == 400
    assert out["note"].startswith(f"Showing the first {n} of 400 rows")

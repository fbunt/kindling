"""Sandbox container tests.

Query execution is container-only, so these are the primary execution-semantics
tests. The pure-unit tests (argv construction, plot materialization) always run;
the rest spawn a Podman worker and skip only if podman + the worker image are
absent. Build the image first:

    podman build -t kindling-worker:latest -f Containerfile .
    uv run pytest tests/test_sandbox_container.py -v
"""

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest

from app.sandbox.pool import (
    SandboxPool,
    build_run_argv,
    detect_runtime,
    materialize_plots,
)

IMAGE = "kindling-worker:latest"


# --- pure unit tests (no podman required) ---


@pytest.mark.parametrize("runtime", ["podman", "docker"])
def test_build_run_argv_hardening(runtime):
    argv = build_run_argv(
        runtime,
        image=IMAGE,
        name="kindling-worker-test",
        host_parquet_path="/data/real.parquet",
        in_container_path="/data/dataset.parquet",
        memory="512m",
        cpus="1.0",
        pids=128,
        max_threads=4,
    )
    assert argv[0] == runtime
    joined = " ".join(argv)
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--memory 512m" in joined
    assert "--memory-swap 512m" in joined
    assert "--pids-limit 128" in joined
    assert "/data/real.parquet:/data/dataset.parquet:ro,z" in joined
    assert "KINDLING_PARQUET_PATH=/data/dataset.parquet" in argv
    # A CPU cap was requested → it and the coupled polars thread count appear.
    assert "--cpus 1.0" in joined
    assert "POLARS_MAX_THREADS=4" in argv
    # --userns=keep-id is podman-only (rootless UID mapping).
    if runtime == "podman":
        assert "--userns=keep-id" in argv
    else:
        assert "--userns=keep-id" not in argv


def test_build_run_argv_all_cores_when_unlimited():
    # cpus=None / max_threads=None → no CPU cap and no POLARS_MAX_THREADS, so the
    # worker uses all host cores (polars/BLAS auto-detect).
    argv = build_run_argv(
        "podman",
        image=IMAGE,
        name="w",
        host_parquet_path="/data/real.parquet",
        in_container_path="/data/dataset.parquet",
        memory="110g",
        cpus=None,
        pids=128,
        max_threads=None,
    )
    assert "--cpus" not in argv
    assert not any(a.startswith("POLARS_MAX_THREADS=") for a in argv)
    # still hardened
    assert "--network" in argv and "--read-only" in argv and "--cap-drop" in argv


def test_build_run_argv_omits_pids_limit_when_none():
    argv = build_run_argv(
        "podman",
        image=IMAGE,
        name="w",
        host_parquet_path="/p",
        in_container_path="/data/dataset.parquet",
        memory="110g",
        cpus=None,
        pids=None,
        max_threads=None,
    )
    assert "--pids-limit" not in argv


def test_materialize_plots_roundtrip(tmp_path, monkeypatch):
    import app.sandbox.pool as pool_mod

    monkeypatch.setattr(pool_mod, "PLOTS_DIR", tmp_path)
    # 1x1 transparent PNG
    png = base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000a49444154789c6360000002000154a24f9b0000000049454e44ae426082"
        )
    ).decode()
    urls = materialize_plots([png])
    assert len(urls) == 1
    # Counter-based name; absolute value depends on test order, so match shape.
    assert re.fullmatch(r"/plots/plot-\d{3,}\.png\?t=\d+", urls[0])
    written = tmp_path / urls[0].removeprefix("/plots/").split("?")[0]
    assert written.exists()
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_materialize_plots_prunes_oldest_beyond_cap(tmp_path, monkeypatch):
    import app.sandbox.pool as pool_mod

    monkeypatch.setattr(pool_mod, "PLOTS_DIR", tmp_path)
    monkeypatch.setattr(pool_mod, "_MAX_PLOTS", 3)
    # Pre-existing plots with increasing mtimes (oldest first).
    for i, name in enumerate(["plot-900.png", "plot-901.png", "plot-902.png"]):
        f = tmp_path / name
        f.write_bytes(b"x")
        os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))
    png = base64.b64encode(b"fake").decode()
    urls = materialize_plots([png])
    new_name = urls[0].removeprefix("/plots/").split("?")[0]
    survivors = sorted(p.name for p in tmp_path.glob("plot-*.png"))
    assert survivors == sorted(["plot-901.png", "plot-902.png", new_name])


# --- container tests (require podman + built image) ---


def _container_available() -> bool:
    try:
        runtime = detect_runtime()
    except RuntimeError:
        return False
    if shutil.which(runtime) is None:
        return False
    return (
        subprocess.run(
            [runtime, "image", "inspect", IMAGE], capture_output=True
        ).returncode
        == 0
    )


requires_container = pytest.mark.skipif(
    not _container_available(),
    reason="podman or kindling-worker image not available",
)


@pytest.fixture(scope="module")
def tiny_parquet(tmp_path_factory) -> Path:
    """A small parquet carrying the __null_dask_index__ column that
    _build_lazyframe drops, plus a few real columns."""
    path = tmp_path_factory.mktemp("data") / "tiny.parquet"
    pl.DataFrame(
        {
            "__null_dask_index__": list(range(5)),
            "year": [1990, 1995, 2000, 2005, 2010],
            "area_m2": [10, 20, 30, 40, 50],
            "geohash": [1000, 2000, 3000, 4000, 5000],
            "Incid_Type": [1, 1, 2, 1, 3],
        }
    ).write_parquet(path)
    return path


@pytest.fixture
async def pool(tiny_parquet) -> SandboxPool:
    p = SandboxPool(tiny_parquet, size=2, max_total=4, image=IMAGE)
    await p.start()
    try:
        yield p
    finally:
        await p.drain()


@requires_container
async def test_basic_query(pool):
    session = await pool.acquire_session()
    try:
        out = await session.run_query("result = lf.head(3).collect()")
        assert "error" not in out, out
        assert len(out["data"]) == 3
        assert {"year", "area_m2"} <= set(out["data"][0].keys())
    finally:
        pool.release_session(session)


@requires_container
async def test_namespace_persists_within_session(pool):
    session = await pool.acquire_session()
    try:
        out1 = await session.run_query("x = 42\nresult = 'set'")
        assert "error" not in out1, out1
        out2 = await session.run_query("result = x + 8")
        assert out2["data"] == "50"
    finally:
        pool.release_session(session)


@requires_container
async def test_result_persists_across_queries_in_session(pool):
    session = await pool.acquire_session()
    try:
        out1 = await session.run_query("result = lf.head(4).collect()")
        assert "error" not in out1, out1
        assert len(out1["data"]) == 4
        out2 = await session.run_query("result = result.head(2)")
        assert "error" not in out2, out2
        assert len(out2["data"]) == 2
    finally:
        pool.release_session(session)


@requires_container
async def test_sessions_are_isolated(pool):
    s1 = await pool.acquire_session()
    s2 = await pool.acquire_session()
    try:
        await s1.run_query("x = 99\nresult = 'set'")
        out = await s2.run_query("result = x + 1")  # x undefined in s2's worker
        assert "error" in out
        assert "NameError" in out["error"]
    finally:
        pool.release_session(s1)
        pool.release_session(s2)


@requires_container
async def test_numpy_method_lazy_import_works(pool):
    """numpy ndarray methods (.mean(), etc.) lazily import numpy submodules at
    runtime; full builtins make that work in the worker."""
    session = await pool.acquire_session()
    try:
        out = await session.run_query(
            "arr = lf.head(5).collect()['area_m2'].to_numpy()\n"
            "result = [float(arr.mean()), float(arr.std())]"
        )
        assert "error" not in out, out
    finally:
        pool.release_session(session)


@requires_container
async def test_xgboost_available(pool):
    session = await pool.acquire_session()
    try:
        out = await session.run_query(
            "import xgboost as xgb, numpy as np\n"
            "X = np.random.rand(200, 4); y = np.random.randint(0, 2, 200)\n"
            "m = xgb.XGBClassifier(n_estimators=10, tree_method='hist').fit(X, y)\n"
            "result = [int(m.predict(X[:3]).shape[0])]"
        )
        assert "error" not in out, out
        assert out["data"] == "[3]"
    finally:
        pool.release_session(session)


@requires_container
async def test_plot_is_returned_as_png(pool, tmp_path, monkeypatch):
    import app.sandbox.pool as pool_mod

    monkeypatch.setattr(pool_mod, "PLOTS_DIR", tmp_path)
    session = await pool.acquire_session()
    try:
        out = await session.run_query(
            "import matplotlib.pyplot as plt\nplt.figure()\nplt.plot([1, 2, 3])"
        )
        assert "error" not in out, out
        assert out["plots"], "expected at least one plot URL"
        url = out["plots"][0]
        fs = tmp_path / url.split("/")[-1].split("?")[0]
        assert fs.exists()
        assert fs.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        pool.release_session(session)


@requires_container
async def test_dispatch_runs_explicit_imports(pool):
    """Through the real dispatch path: explicit imports now just work (full
    builtins, no AST stripping). The import statement is harmless and runs."""
    from app.tools import execute_function_call_async

    session = await pool.acquire_session()
    try:
        code = "import polars as pl\nresult = lf.head(2).collect()"
        result_str, _ = await execute_function_call_async(
            "run_query", {"code": code}, None, "", session
        )
        result = json.loads(result_str)
        assert "error" not in result, result
        assert len(result["data"]) == 2
    finally:
        pool.release_session(session)


@requires_container
async def test_arbitrary_stdlib_import_is_allowed_in_box(pool):
    """No AST blocklist: `import os` runs (it's harmless in the container)."""
    session = await pool.acquire_session()
    try:
        out = await session.run_query("import os\nresult = [os.getpid() > 0]")
        assert "error" not in out, out
        assert out["data"] == "[True]"
    finally:
        pool.release_session(session)


@requires_container
async def test_result_truncated_to_max_rows(pool):
    session = await pool.acquire_session()
    try:
        out = await session.run_query("result = pl.DataFrame({'x': list(range(250))})")
        assert out["total_rows"] == 250
        assert out["truncated"] is True
        assert len(out["data"]) == 100  # MAX_ROWS
    finally:
        pool.release_session(session)


@requires_container
async def test_runtime_error_is_surfaced(pool):
    session = await pool.acquire_session()
    try:
        out = await session.run_query("result = 1 / 0")
        assert "error" in out
        assert "ZeroDivisionError" in out["error"]
    finally:
        pool.release_session(session)


@requires_container
async def test_no_result_assigned_errors(pool):
    session = await pool.acquire_session()
    try:
        out = await session.run_query("x = 1")
        assert "error" in out
        assert "No result" in out["error"]
    finally:
        pool.release_session(session)


@requires_container
async def test_wedged_query_surfaces_worker_dead(pool):
    session = await pool.acquire_session()
    pool.hard_timeout = 3.0  # don't wait the full soft timeout
    try:
        out = await session.run_query("result = [0]\nwhile True:\n    pass")
        assert "error" in out
        assert "terminated unexpectedly" in out["error"]
    finally:
        pool.release_session(session)

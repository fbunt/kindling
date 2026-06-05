import os
from pathlib import Path

import pytest

_PARQUET = Path(os.environ.get("KINDLING_PARQUET", "data/mtbs_pix_data.parquet"))
_SAMPLE = Path(".eval-runs/sample.parquet")
_SYNTHETIC = Path(".eval-runs/_synthetic.parquet")
_SAMPLE_MOD = 1000  # keep ~1/1000 of rows via geohash % N == 0


def _synthetic_sample() -> Path:
    """A tiny schema-shaped parquet for environments without the real dataset
    (e.g. CI, where data/ is a dangling symlink). Carries __null_dask_index__
    (dropped by _build_lazyframe) plus the key columns + a few rows, so
    get_dataset_info works. NOT for query correctness — just schema/sample reads.
    """
    if _SYNTHETIC.exists():
        return _SYNTHETIC
    from datetime import date

    import polars as pl

    from app.query_engine import _KEY_COLUMNS

    n = 5

    def col(name: str) -> list:
        if name in ("lat", "lon"):
            return [44.0 + i for i in range(n)]
        if name == "Ig_Date":
            return [date(1984 + i, 1, 1) for i in range(n)]
        if name in ("Incid_Name", "Event_ID", "eco1s", "eco2s", "eco3s"):
            return [f"{name}-{i}" for i in range(n)]
        if name == "year":
            return [1984 + i for i in range(n)]
        return list(range(n))

    data = {"__null_dask_index__": list(range(n))}
    data.update({k: col(k) for k in _KEY_COLUMNS})
    _SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(_SYNTHETIC)
    return _SYNTHETIC


def _ensure_sample() -> Path:
    """A real-schema parquet to configure the dataset module with. Prefers the
    cached sample, builds it from the full dataset if present, else falls back to
    a synthetic schema-shaped parquet (no dataset on this machine — e.g. CI)."""
    if _SAMPLE.exists():
        return _SAMPLE
    if not _PARQUET.exists():
        return _synthetic_sample()
    import polars as pl

    _SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_parquet(_PARQUET)
        .filter(pl.col("geohash") % _SAMPLE_MOD == 0)
        .sink_parquet(_SAMPLE, compression="zstd")
    )
    return _SAMPLE


@pytest.fixture(scope="session", autouse=True)
def _configure_query_engine():
    """Configure the dataset module so get_dataset_info tests can read schema/
    sample rows. configure() is idempotent, so this is a no-op if a more specific
    fixture (e.g. the evals suite) already ran."""
    from app.query_engine import configure

    configure(_ensure_sample())

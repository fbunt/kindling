import os
from pathlib import Path

import pytest

_PARQUET = Path(os.environ.get("KINDLING_PARQUET", "data/mtbs_pix_data.parquet"))
_SAMPLE = Path(".eval-runs/sample.parquet")
_SAMPLE_MOD = 1000  # keep ~1/1000 of rows via geohash % N == 0


def _ensure_sample() -> Path | None:
    """Return a small real-schema parquet, building it from the full dataset
    once if needed. Returns None if no source data is available."""
    if _SAMPLE.exists():
        return _SAMPLE
    if not _PARQUET.exists():
        return None
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
    """Configure the dataset module with a sampled parquet so get_dataset_info
    tests can read schema/sample rows. configure() is idempotent, so this is a
    no-op if a more specific fixture (e.g. the evals suite) already ran."""
    sample = _ensure_sample()
    if sample is not None:
        from app.query_engine import configure

        configure(sample)

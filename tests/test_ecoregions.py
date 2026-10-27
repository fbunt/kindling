"""The ecoregion lookup the worker preloads as `eco_names`."""

from pathlib import Path

import polars as pl

from app.query_engine import _KEY_COLUMNS

ECOREGIONS = Path(__file__).resolve().parents[1] / "app/sandbox/ecoregions.csv"


def _names() -> pl.DataFrame:
    return pl.read_csv(
        ECOREGIONS, schema_overrides={"eco2_code": pl.String, "eco3_code": pl.String}
    )


def test_one_row_per_level_three_ecoregion():
    names = _names()
    assert names["eco3"].is_unique().all()
    assert names.null_count().sum_horizontal().item() == 0


def test_each_level_one_and_two_code_has_one_name():
    names = _names()
    for level in ("eco1", "eco2"):
        named = names.select(level, f"{level}_name").unique()
        assert named[level].is_unique().all(), level


def test_integer_codes_are_the_dotted_codes_without_dots():
    # eco2 62 is 6.2; eco3 6205 is 6.2.5 and 6214 is 6.2.14.
    for row in _names().iter_rows(named=True):
        l1, l2, l3 = (int(p) for p in row["eco3_code"].split("."))
        assert row["eco3_code"].startswith(row["eco2_code"] + ".")
        assert (row["eco1"], row["eco2"], row["eco3"]) == (
            l1,
            l1 * 10 + l2,
            l1 * 1000 + l2 * 100 + l3,
        ), row


def test_dataset_description_points_at_the_lookup():
    for level in ("eco1", "eco2", "eco3"):
        assert "eco_names" in _KEY_COLUMNS[level]


def test_every_code_in_the_sample_is_named():
    import pytest

    from app import query_engine

    if query_engine.RESOLVED_PARQUET_PATH.name != "sample.parquet":
        pytest.skip("needs the real-data sample")
    codes = query_engine.LF.select("eco1", "eco2", "eco3").unique().collect()
    missing = codes.join(
        _names().with_columns(pl.col(c).cast(codes.schema[c]) for c in codes.columns),
        on=["eco1", "eco2", "eco3"],
        how="anti",
    )
    assert missing.is_empty(), missing

"""Dataset configuration and schema introspection.

Query EXECUTION lives entirely in the sandbox worker (app/sandbox/worker.py),
run via the container pool (app/sandbox/pool.py). This module only loads the
dataset's lazy frame for schema/sample reads (get_dataset_info) and exposes the
resolved parquet path so the pool can bind-mount it. No model code runs here.
"""

import logging
import os
from pathlib import Path

import polars as pl

DEFAULT_PARQUET_PATH = Path("data/mtbs_pix_data.parquet")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

LF: pl.LazyFrame | None = None
RESOLVED_PARQUET_PATH: Path | None = None
_SCHEMA = None
_SCHEMA_INFO: dict[str, str] | None = None


def _build_lazyframe(path: Path | str) -> pl.LazyFrame:
    """Build the dataset LazyFrame from a parquet path.

    The sandbox worker (app/sandbox/worker.py) MUST reproduce this expression
    verbatim — any divergence in the scan changes query semantics.
    """
    return pl.scan_parquet(path)


def configure(parquet_path: Path | str | None = None) -> None:
    """Initialize the dataset from a parquet file. Safe to call multiple times."""
    global LF, RESOLVED_PARQUET_PATH, _SCHEMA, _SCHEMA_INFO
    if LF is not None:
        return
    path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    RESOLVED_PARQUET_PATH = path.resolve()
    # Expose the resolved path so the sandbox pool can bind-mount it even when
    # the app is started via uvicorn's string-import, where configure() may run
    # again with no argument in a fresh process.
    os.environ["KINDLING_PARQUET_PATH_HOST"] = str(RESOLVED_PARQUET_PATH)
    LF = _build_lazyframe(path)
    _SCHEMA = LF.collect_schema()
    _SCHEMA_INFO = {name: str(dtype) for name, dtype in _SCHEMA.items()}


_NLCD_CODES = "11=Open Water, 12=Perennial Ice/Snow, 21=Developed: Open Space, 22=Developed: Low Intensity, 23=Developed: Med Intensity, 24=Developed: High Intensity, 31=Barren Land, 41=Deciduous Forest, 42=Evergreen Forest, 43=Mixed Forest, 52=Shrub/Scrub, 71=Grassland/Herbaceous, 81=Pasture/Hay, 82=Cultivated Crops, 90=Woody Wetlands, 95=Emergent Herbaceous Wetlands"  # noqa: E501

_KEY_COLUMNS = {
    "year": "Fire year (1984-2022)",
    "Event_ID": "Unique fire event identifier; its first two letters are the state MTBS assigns the fire",  # noqa: E501
    "Incid_Name": "Fire incident name",
    "Incid_Type": "Incident type (0=Unknown, 1=Wildfire, 2=Prescribed Fire, 3=Wildland Fire Use). Category 3 is a wildfire left to burn, functionally similar to wildfire.",  # noqa: E501
    "Ig_Date": "Ignition date of fire",
    "area_m2": "Fire area in square meters (the whole fire's area, repeated on each of its pixels)",  # noqa: E501
    "area_acres": "Fire area in acres (area_m2 / 4046.856), repeated on each pixel",
    "state": "Two-letter code of the state this pixel lies in. A fire crossing a state line has pixels in several states; the Event_ID prefix is the one state MTBS assigns it.",  # noqa: E501
    "eco1": "Level I ecoregion code (e.g. 6). Names: join the preloaded `eco_names` frame on eco1.",  # noqa: E501
    "eco2": "Level II ecoregion code, the official dotted code without its dot (62 = 6.2). Names and dotted codes: join `eco_names` on eco2.",  # noqa: E501
    "eco3": "Level III ecoregion code (6205 = 6.2.5, 10102 = 10.1.2). Names and dotted codes: join `eco_names` on eco3.",  # noqa: E501
    "geohash": "Unique ID for each pixel location",
    "lon": "Longitude",
    "lat": "Latitude",
    "bs": "Burn severity class (1=Unburned, 2=Low, 3=Moderate, 4=High, 5=Increased Greenness, 6=Non-processing)",  # noqa: E501
    "nlcd": f"NLCD land cover class from the map of the fire's own year, which for moderate and high severity often already shows the burn. Null for 1984 (the annual maps start in 1985). Codes: {_NLCD_CODES}",  # noqa: E501
    "nlcd_pre": "NLCD land cover class from the year before the fire: the pre-fire land cover. Null for 1984 and 1985 fires (no earlier map). Same codes as nlcd.",  # noqa: E501
    "nlcd_post": "NLCD land cover class from the year after the fire. Same codes as nlcd.",  # noqa: E501
    "nlcd_mode": "Most common NLCD land cover class at this location across the study period (same codes as nlcd)",  # noqa: E501
    "wui_bool": "Wildland-Urban Interface True/False (1/0)",
    "wui_prox": "Wildland-Urban Interface proximity",
    "elevation": "Elevation in meters",
    "slope": "Slope in degrees",
    "aspect": "Aspect in degrees (0-360, the direction the slope faces); -1 = flat. Circular: average with atan2 of mean sin/cos, never a plain mean",  # noqa: E501
}


def get_dataset_info() -> dict:
    """Return schema, row count, and sample rows for Gemini context."""
    sample = LF.head(5).collect()
    return {
        "columns": _SCHEMA_INFO,
        "key_columns": _KEY_COLUMNS,
        "row_count_approx": "~749 million",
        "sample_rows": sample.to_dicts(),
    }

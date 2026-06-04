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
    verbatim — any divergence in the scan/drop changes query semantics.
    """
    return pl.scan_parquet(path).drop("__null_dask_index__")


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


_KEY_COLUMNS = {
    "year": "Fire year (1984-2022)",
    "Incid_Name": "Fire incident name",
    "Event_ID": "Unique fire event identifier",
    "Incid_Type": "Incident type (0=Unknown, 1=Wildfire, 2=Prescribed Fire, 3=Wildland Fire Use). Category 3 is a wildfire left to burn, functionally similar to wildfire.",  # noqa: E501
    "area_m2": "Fire area in square meters",
    "geohash": "Unique ID for each pixel location",
    "bs": "Burn severity class (1=Unburned, 2=Low, 3=Moderate, 4=High, 5=Increased Greenness, 6=Non-processing)",  # noqa: E501
    "Ig_Date": "Ignition date of fire",
    "lat": "Latitude",
    "lon": "Longitude",
    "elevation": "Elevation in meters",
    "eco1": "Ecoregion level 1 as int (1=Arctic Cordillera, 2=Tundra, 3=Taiga, 4=Hudson Plain, 5=Northern Forests, 6=Northwestern Forested Mountains, 7=Marine West Coast Forest, 8=Eastern Temperate Forests, 9=Great Plains, 10=North American Deserts, 11=Mediterranean California, 12=Southern Semiarid Highlands, 13=Temperate Sierras, 14=Tropical Dry Forests, 15=Tropical Wet Forests)",  # noqa: E501
    "eco2": "Ecoregion level 2 as int",
    "eco3": "Ecoregion level 3 as int",
    "eco1s": "Ecoregion level 1 as string (form: XX)",
    "eco2s": "Ecoregion level 2 as string (form: XX.Y)",
    "eco3s": "Ecoregion level 3 as string (form: XX.Y.ZZ)",
    "nlcd": "NLCD land cover class (11=Open Water, 12=Perennial Ice/Snow, 21=Developed: Open Space, 22=Developed: Low Intensity, 23=Developed: Med Intensity, 24=Developed: High Intensity, 31=Barren Land, 41=Deciduous Forest, 42=Evergreen Forest, 43=Mixed Forest, 52=Shrub/Scrub, 71=Grassland/Herbaceous, 81=Pasture/Hay, 82=Cultivated Crops, 90=Woody Wetlands, 95=Emergent Herbaceous Wetlands)",  # noqa: E501
    "nlcd_mode": "NLCD land cover class mode across the 38-year period for a given location (same codes as nlcd)",  # noqa: E501
    "wui_bool": "Wildland-Urban Interface True/False (1/0)",
    "wui_prox": "Wildland-Urban Interface proximity",
}


def get_dataset_info() -> dict:
    """Return schema, row count, and sample rows for Gemini context."""
    sample = LF.head(5).collect()
    return {
        "columns": _SCHEMA_INFO,
        "key_columns": _KEY_COLUMNS,
        "row_count_approx": "~745 million",
        "sample_rows": sample.to_dicts(),
    }

"""The 25 benchmark questions.

Each Question pairs a natural-language prompt (sent to the model verbatim) with
hand-written reference Polars code that defines ground truth. Reference code is
executed HOST-SIDE by bench/ground_truth.py in a namespace containing:

- `lf`  — the LazyFrame built by app.query_engine._build_lazyframe (imported,
          never reimplemented, so the `__null_dask_index__` drop can't diverge)
- `pl`  — polars
- `ols_slope(xs, ys)` — plain-Python ordinary-least-squares slope

Conventions: every `.collect()` uses engine="streaming" (the full dataset is
745M rows); the code must assign `expected`, which is normalized to a
JSON-serializable scalar, dict (series), or sorted list (set / tie-safe text).

Questions are phrased to fix a single defensible interpretation (dedup rule,
decade bounds, regression method, units) — the paper's accuracy claim depends
on the question having one right answer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str  # L01..L08, A01..A07, T01..T05, M01..M05
    category: str  # lookup | aggregation | trend | multistep
    text: str  # prompt sent to the model, verbatim
    reference_code: str  # host-executed Polars; must assign `expected`
    answer_kind: str  # scalar | series | set | text
    criterion: str  # judge criterion template; {expected} placeholder
    tolerance_rel: float = 0.01  # relative band for scalar/series numerics
    tolerance_abs: float | None = None  # absolute band; overrides rel when set
    expensive_gt: bool = False  # ground truth is more than a trivial scan


# Several references roll up to one row per fire event via
# lf.group_by("Event_ID").agg(...). `first()` is safe for fire-level columns
# (area_m2, Incid_Name, year, Ig_Date are constant within an Event_ID).
QUESTIONS: list[Question] = [
    # ------------------------------------------------------------------
    # Lookups (8)
    # ------------------------------------------------------------------
    Question(
        id="L01",
        category="lookup",
        text=(
            "What is the area in square meters (area_m2, as recorded) of the "
            "single largest fire event in the dataset?"
        ),
        reference_code="""\
expected = lf.select(pl.col("area_m2").max()).collect(engine="streaming").item()
""",
        answer_kind="scalar",
        criterion=(
            "The response states the area of the largest fire event as "
            "{expected} square meters."
        ),
        tolerance_rel=1e-6,
    ),
    Question(
        id="L02",
        category="lookup",
        text=(
            "What is the incident name (Incid_Name) of the largest fire event "
            "by area_m2?"
        ),
        reference_code="""\
mx = lf.select(pl.col("area_m2").max()).collect(engine="streaming").item()
expected = sorted(
    lf.filter(pl.col("area_m2") == mx)
    .select("Incid_Name")
    .unique()
    .collect(engine="streaming")["Incid_Name"]
    .to_list()
)
""",
        answer_kind="text",
        criterion=(
            "The response identifies the largest fire event's incident name as "
            "{expected}. Case and punctuation differences are fine; if several "
            "names are listed as expected, naming any one of them counts."
        ),
    ),
    Question(
        id="L03",
        category="lookup",
        text=(
            "How many distinct fire events (unique Event_ID values) are in the dataset?"
        ),
        reference_code="""\
expected = lf.select(pl.col("Event_ID").n_unique()).collect(engine="streaming").item()
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of distinct fire events as {expected}."
        ),
        tolerance_rel=1e-6,
    ),
    Question(
        id="L04",
        category="lookup",
        text=(
            "How many distinct fire events (unique Event_ID) have "
            "Incid_Type = 2, i.e. prescribed fires?"
        ),
        reference_code="""\
expected = (
    lf.filter(pl.col("Incid_Type") == 2)
    .select(pl.col("Event_ID").n_unique())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of distinct prescribed-fire events "
            "(Incid_Type = 2) as {expected}."
        ),
        tolerance_rel=1e-6,
    ),
    Question(
        id="L05",
        category="lookup",
        text="What is the earliest ignition date (Ig_Date) recorded in the dataset?",
        reference_code="""\
# Ig_Date is a Datetime in the parquet; truncate to the calendar date.
expected = str(
    lf.select(pl.col("Ig_Date").min()).collect(engine="streaming").item().date()
)
""",
        answer_kind="text",
        criterion=(
            "The response gives the earliest ignition date as {expected}. "
            "Equivalent date formats count (e.g. 'January 26, 1984' matches "
            "'1984-01-26')."
        ),
    ),
    Question(
        id="L06",
        category="lookup",
        text="What is the maximum pixel elevation in meters recorded in the dataset?",
        reference_code="""\
expected = lf.select(pl.col("elevation").max()).collect(engine="streaming").item()
""",
        answer_kind="scalar",
        criterion=(
            "The response states the maximum pixel elevation as {expected} meters."
        ),
        # elevation is stored as float; allow rounding to the nearest meter.
        tolerance_abs=1.0,
    ),
    Question(
        id="L07",
        category="lookup",
        text="How many burned-pixel rows does the dataset contain for the year 2020?",
        reference_code="""\
expected = (
    lf.filter(pl.col("year") == 2020)
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of burned-pixel rows for 2020 as "
            "{expected}."
        ),
        tolerance_rel=1e-6,
    ),
    Question(
        id="L08",
        category="lookup",
        text="How many rows have a null burn severity (bs) value?",
        reference_code="""\
expected = lf.select(pl.col("bs").null_count()).collect(engine="streaming").item()
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of rows with a null burn severity "
            "as {expected}."
        ),
        tolerance_rel=1e-6,
    ),
    # ------------------------------------------------------------------
    # Aggregations (7)
    # ------------------------------------------------------------------
    Question(
        id="A01",
        category="aggregation",
        text=(
            "What is the mean fire size in square meters across distinct fire "
            "events? Deduplicate by Event_ID so each fire counts once."
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(pl.col("area_m2").first())
expected = ev.select(pl.col("area_m2").mean()).collect(engine="streaming").item()
""",
        answer_kind="scalar",
        criterion=(
            "The response states the mean fire size across distinct fire "
            "events as {expected} square meters."
        ),
        tolerance_rel=0.01,
    ),
    Question(
        id="A02",
        category="aggregation",
        text="Which single year has the largest number of burned-pixel rows?",
        reference_code="""\
counts = lf.group_by("year").len().collect(engine="streaming")
expected = int(counts.sort("len", descending=True)["year"][0])
""",
        answer_kind="text",
        criterion=(
            "The response identifies {expected} as the year with the most "
            "burned-pixel rows."
        ),
    ),
    Question(
        id="A03",
        category="aggregation",
        text=(
            "For the year 2021, report the number of pixels in each burn "
            "severity class 1 through 4 (four counts: bs=1, 2, 3, 4)."
        ),
        reference_code="""\
counts = (
    lf.filter((pl.col("year") == 2021) & pl.col("bs").is_in([1, 2, 3, 4]))
    .group_by("bs")
    .len()
    .collect(engine="streaming")
)
expected = {str(r["bs"]): r["len"] for r in counts.to_dicts()}
""",
        answer_kind="series",
        criterion=(
            "The response reports 2021 pixel counts per burn severity class "
            "matching all of: {expected}."
        ),
        tolerance_rel=0.005,
    ),
    Question(
        id="A04",
        category="aggregation",
        text=(
            "What percentage of all pixel rows are inside the wildland-urban "
            "interface (wui_bool = 1)? One decimal place is fine."
        ),
        reference_code="""\
expected = (
    lf.select(pl.col("wui_bool").mean()).collect(engine="streaming").item() * 100
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} percent of pixel rows are "
            "inside the wildland-urban interface."
        ),
        tolerance_abs=0.1,
    ),
    Question(
        id="A05",
        category="aggregation",
        text=(
            "What are the incident names of the 5 largest distinct fire events "
            "by area_m2?"
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("Incid_Name").first(),
    pl.col("area_m2").first(),
)
top = ev.collect(engine="streaming").top_k(5, by="area_m2")
expected = sorted(top["Incid_Name"].to_list())
""",
        answer_kind="set",
        criterion=(
            "The response names exactly these 5 fires as the largest by area: "
            "{expected}."
        ),
    ),
    Question(
        id="A06",
        category="aggregation",
        text=(
            "Which level-1 ecoregion has the most burned-pixel rows in total? "
            "Identify it by its eco1 integer code (the name too, if you like)."
        ),
        reference_code="""\
counts = lf.group_by("eco1").len().collect(engine="streaming")
expected = int(counts.sort("len", descending=True)["eco1"][0])
""",
        answer_kind="text",
        criterion=(
            "The response identifies eco1 code {expected} as the level-1 "
            "ecoregion with the most burned-pixel rows. The integer code "
            "{expected} must appear; an accompanying ecoregion name is fine."
        ),
    ),
    Question(
        id="A07",
        category="aggregation",
        text=(
            "What is the mean elevation in meters of pixels that burned at "
            "high severity (bs = 4), across the whole dataset?"
        ),
        reference_code="""\
expected = (
    lf.filter(pl.col("bs") == 4)
    .select(pl.col("elevation").mean())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states the mean elevation of high-severity (bs = 4) "
            "pixels as {expected} meters."
        ),
        tolerance_rel=0.01,
    ),
    # ------------------------------------------------------------------
    # Trends (5) — method pinned in the wording
    # ------------------------------------------------------------------
    Question(
        id="T01",
        category="trend",
        text=(
            "Using ordinary least-squares regression of the annual count of "
            "distinct fire events (unique Event_ID per year) against year, "
            "over 1984-2022, what is the slope in fires per year?"
        ),
        reference_code="""\
counts = (
    lf.group_by("year")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .sort("year")
    .collect(engine="streaming")
)
expected = ols_slope(counts["year"].to_list(), counts["n"].to_list())
""",
        answer_kind="scalar",
        criterion=(
            "The response states the ordinary least-squares slope of annual "
            "distinct fire-event counts against year as {expected} fires per "
            "year."
        ),
        tolerance_rel=0.05,
    ),
    Question(
        id="T02",
        category="trend",
        text=(
            "Define annual burned area as the sum of area_m2 over distinct "
            "fire events ignited in each year. What is the ordinary "
            "least-squares slope of annual burned area against year, "
            "1984-2022, in square meters per year?"
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("area_m2").first(),
    pl.col("year").first(),
)
annual = (
    ev.group_by("year")
    .agg(pl.col("area_m2").sum())
    .sort("year")
    .collect(engine="streaming")
)
expected = ols_slope(annual["year"].to_list(), annual["area_m2"].to_list())
""",
        answer_kind="scalar",
        criterion=(
            "The response states the ordinary least-squares slope of annual "
            "burned area against year as {expected} square meters per year. "
            "The same value in scientific notation counts; a value converted "
            "to other units (km², acres, hectares) does not count unless the "
            "square-meter figure is also given."
        ),
        tolerance_rel=0.05,
    ),
    Question(
        id="T03",
        category="trend",
        text=(
            "Compare the mean annual number of distinct fire events in the "
            "first decade of the record (1984-1993) with the last decade "
            "(2013-2022). Report the ratio last-decade mean divided by "
            "first-decade mean."
        ),
        reference_code="""\
counts = (
    lf.group_by("year")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .collect(engine="streaming")
)
first = counts.filter(pl.col("year").is_between(1984, 1993))["n"].mean()
last = counts.filter(pl.col("year").is_between(2013, 2022))["n"].mean()
expected = last / first
""",
        answer_kind="scalar",
        criterion=(
            "The response states the ratio of the 2013-2022 mean annual "
            "distinct fire-event count to the 1984-1993 mean as {expected}."
        ),
        tolerance_rel=0.02,
    ),
    Question(
        id="T04",
        category="trend",
        text=(
            "For each year, define the high-severity fraction as pixels with "
            "bs = 4 divided by pixels with bs in 1-4 (exclude nulls and "
            "classes 5-6). By how many percentage points does the mean annual "
            "high-severity fraction in 2013-2022 differ from 1984-1993 "
            "(positive = increase)?"
        ),
        reference_code="""\
frac = (
    lf.filter(pl.col("bs").is_in([1, 2, 3, 4]))
    .group_by("year")
    .agg((pl.col("bs") == 4).mean().alias("hs"))
    .collect(engine="streaming")
)
first = frac.filter(pl.col("year").is_between(1984, 1993))["hs"].mean()
last = frac.filter(pl.col("year").is_between(2013, 2022))["hs"].mean()
expected = (last - first) * 100
""",
        answer_kind="scalar",
        criterion=(
            "The response states that the mean annual high-severity fraction "
            "changed by {expected} percentage points between 1984-1993 and "
            "2013-2022. The sign matters: positive means an increase."
        ),
        tolerance_abs=0.5,
    ),
    Question(
        id="T05",
        category="trend",
        text=(
            "Has the fire season shifted later? Using each distinct fire "
            "event's ignition date (Ig_Date), compare the mean day-of-year of "
            "ignition for events in 1984-1993 vs 2013-2022. Report the "
            "difference in days (positive = later in the recent decade)."
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("Ig_Date").first(),
    pl.col("year").first(),
)
doy = ev.with_columns(pl.col("Ig_Date").dt.ordinal_day().alias("doy")).collect(
    engine="streaming"
)
first = doy.filter(pl.col("year").is_between(1984, 1993))["doy"].mean()
last = doy.filter(pl.col("year").is_between(2013, 2022))["doy"].mean()
expected = last - first
""",
        answer_kind="scalar",
        criterion=(
            "The response states that the mean ignition day-of-year shifted by "
            "{expected} days between 1984-1993 and 2013-2022. The sign "
            "matters: positive means later in the recent decade."
        ),
        tolerance_abs=2.0,
    ),
    # ------------------------------------------------------------------
    # Multi-step (5)
    # ------------------------------------------------------------------
    Question(
        id="M01",
        category="multistep",
        text=(
            "Of the 10 largest distinct fire events by area_m2, how many are "
            "located in the box lat 32.5-42.0, lon -124.5 to -114.1 (roughly "
            "California)? Locate each fire by the mean lat and mean lon of its "
            "pixels."
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("area_m2").first(),
    pl.col("lat").mean(),
    pl.col("lon").mean(),
)
top = ev.collect(engine="streaming").top_k(10, by="area_m2")
expected = int(
    (
        (top["lat"] >= 32.5)
        & (top["lat"] <= 42.0)
        & (top["lon"] >= -124.5)
        & (top["lon"] <= -114.1)
    ).sum()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} of the 10 largest fire events "
            "fall within the given lat/lon box."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="M02",
        category="multistep",
        text=(
            "Within Mediterranean California (eco1 = 11), how many distinct "
            "pixel locations (unique geohash) burned in 3 or more distinct "
            "fire events over the record?"
        ),
        reference_code="""\
expected = (
    lf.filter(pl.col("eco1") == 11)
    .group_by("geohash")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .filter(pl.col("n") >= 3)
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} distinct pixel locations in "
            "Mediterranean California burned in 3 or more distinct fire "
            "events."
        ),
        tolerance_rel=1e-6,
        expensive_gt=True,
    ),
    Question(
        id="M03",
        category="multistep",
        text=(
            "Consider the single largest fire event by area_m2. What "
            "percentage of its pixels with bs in 1-4 burned at high severity "
            "(bs = 4)? One decimal place."
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(pl.col("area_m2").first())
top_id = ev.collect(engine="streaming").top_k(1, by="area_m2")["Event_ID"][0]
counts = (
    lf.filter((pl.col("Event_ID") == top_id) & pl.col("bs").is_in([1, 2, 3, 4]))
    .select((pl.col("bs") == 4).sum().alias("high"), pl.len().alias("total"))
    .collect(engine="streaming")
)
expected = counts["high"][0] / counts["total"][0] * 100
""",
        answer_kind="scalar",
        criterion=(
            "The response states the high-severity percentage for the largest "
            "fire event as {expected} percent."
        ),
        tolerance_abs=0.5,
    ),
    Question(
        id="M04",
        category="multistep",
        text=(
            "Which year has the most distinct fire events, and what is the "
            "incident name of the largest fire (by area_m2) ignited in that "
            "year? Give the fire's name."
        ),
        reference_code="""\
counts = (
    lf.group_by("year")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .collect(engine="streaming")
)
peak_year = int(counts.sort("n", descending=True)["year"][0])
ev = (
    lf.filter(pl.col("year") == peak_year)
    .group_by("Event_ID")
    .agg(pl.col("Incid_Name").first(), pl.col("area_m2").first())
    .collect(engine="streaming")
)
mx = ev["area_m2"].max()
expected = sorted(ev.filter(pl.col("area_m2") == mx)["Incid_Name"].unique().to_list())
""",
        answer_kind="text",
        criterion=(
            "The response identifies the fire's incident name as {expected}. "
            "Case and punctuation differences are fine; if several names are "
            "listed as expected, naming any one of them counts."
        ),
    ),
    Question(
        id="M05",
        category="multistep",
        text=(
            "Among the 20 largest distinct fire events by area_m2, how many "
            "have more than 50% of their pixels inside the WUI "
            "(wui_bool = 1)?"
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("area_m2").first(),
    pl.col("wui_bool").mean().alias("wui_frac"),
)
top = ev.collect(engine="streaming").top_k(20, by="area_m2")
expected = int((top["wui_frac"] > 0.5).sum())
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} of the 20 largest fire events "
            "have more than 50% of their pixels inside the WUI."
        ),
        tolerance_abs=0.0,
    ),
]

BY_ID: dict[str, Question] = {q.id: q for q in QUESTIONS}
CATEGORIES = ("lookup", "aggregation", "trend", "multistep")

assert len(QUESTIONS) == 25
assert len(BY_ID) == 25, "duplicate question ids"


def select(spec: str | None) -> list[Question]:
    """Select questions by comma-separated ids ("L01,M04") or category name.

    None or "" selects all 25.
    """
    if not spec:
        return list(QUESTIONS)
    if spec in CATEGORIES:
        return [q for q in QUESTIONS if q.category == spec]
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token not in BY_ID:
            raise KeyError(f"unknown question id or category: {token!r}")
        out.append(BY_ID[token])
    return out

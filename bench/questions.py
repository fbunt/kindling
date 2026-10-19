"""The benchmark questions: the 25-question core tier and the extended tier.

Each Question pairs a natural-language prompt (sent to the model verbatim) with
hand-written reference Polars code that defines ground truth. Reference code is
executed HOST-SIDE by bench/ground_truth.py in a namespace containing:

- `lf`  — the LazyFrame built by app.query_engine._build_lazyframe (imported,
          never reimplemented, so the `__null_dask_index__` drop can't diverge)
- `pl`  — polars
- `ols_slope(xs, ys)` — plain-Python ordinary-least-squares slope
- `logger` - the ground-truth logger, for tables a reference reports

Conventions: every `.collect()` uses engine="streaming" (the full dataset is
745M rows); the code must assign `expected`, which is normalized to a
JSON-serializable scalar, dict (series), or sorted list (set / tie-safe text).

Questions are phrased to fix a single defensible interpretation (dedup rule,
decade bounds, regression method, units) — the paper's accuracy claim depends
on the question having one right answer. The extended tier also measures
behaviour (abstaining, disclosing truncation, stating an assumption); its
questions are run and reported apart from the core tier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str  # L01..L08, A01..A07, T01..T05, M01..M05; extended E01..E14
    category: str  # a CATEGORIES or EXTENDED_CATEGORIES name
    text: str  # prompt sent to the model, verbatim
    reference_code: str  # host-executed Polars; must assign `expected`
    # core: scalar | series | set | text; extended adds abstain | disclosure |
    # clarify_or_assume | plot
    answer_kind: str
    criterion: str  # legacy judge template; hashed into question_sha, not graded
    tolerance_rel: float = 0.01  # relative band for scalar/series numerics
    tolerance_abs: float | None = None  # absolute band; overrides rel when set
    expensive_gt: bool = False  # ground truth is more than a trivial scan
    # Scripted turns sent before `text`, as (role, text) pairs with role "user"
    # or "model"; hashed into question_sha only when non-empty.
    history: tuple[tuple[str, str], ...] = ()


# Incident-type scope is stated in every question that counts fire events: the
# trend questions (T01-T05) cover wildfire only (Incid_Type == 1, filtered in
# the reference); A01/M02/M04 say "Count every incident type".
#
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
            "What is the incident name of the largest prescribed-fire event "
            "(Incid_Type = 2) by area_m2?"
        ),
        reference_code="""\
rx = lf.filter(pl.col("Incid_Type") == 2)
mx = rx.select(pl.col("area_m2").max()).collect(engine="streaming").item()
expected = sorted(
    rx.filter(pl.col("area_m2") == mx)
    .select("Incid_Name")
    .unique()
    .collect(engine="streaming")["Incid_Name"]
    .to_list()
)
""",
        answer_kind="text",
        criterion=(
            "The response identifies the incident name of the largest "
            "prescribed-fire event as {expected}. Case and punctuation "
            "differences are fine; if several names are listed as expected, "
            "naming any one of them counts."
        ),
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
            "How many distinct fire events (unique Event_ID values) are in the "
            "dataset? Give the exact count."
        ),
        reference_code="""\
expected = lf.select(pl.col("Event_ID").n_unique()).collect(engine="streaming").item()
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of distinct fire events as {expected}."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="L04",
        category="lookup",
        text=(
            "How many distinct prescribed-fire events are in the dataset? "
            "Give the exact count."
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
            "as {expected}."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="L05",
        category="lookup",
        text=(
            "What is the ignition date (Ig_Date) of the earliest Wildland Fire "
            "Use event (Incid_Type = 3)?"
        ),
        reference_code="""\
# Ig_Date is a Datetime in the parquet; truncate to the calendar date.
expected = str(
    lf.filter(pl.col("Incid_Type") == 3)
    .select(pl.col("Ig_Date").min())
    .collect(engine="streaming")
    .item()
    .date()
)
""",
        answer_kind="text",
        criterion=(
            "The response gives the ignition date of the earliest Wildland "
            "Fire Use event as {expected}. Equivalent date formats count (e.g. "
            "'March 26, 1988' matches '1988-03-26')."
        ),
    ),
    Question(
        id="L06",
        category="lookup",
        text=(
            "Which fire event contains the single highest-elevation pixel? "
            "Give its incident name."
        ),
        reference_code="""\
mx = lf.select(pl.col("elevation").max()).collect(engine="streaming").item()
expected = sorted(
    lf.filter(pl.col("elevation") == mx)
    .select("Incid_Name")
    .unique()
    .collect(engine="streaming")["Incid_Name"]
    .to_list()
)
""",
        answer_kind="text",
        criterion=(
            "The response identifies the fire event containing the "
            "highest-elevation pixel by its incident name, {expected}. Case and "
            "punctuation differences are fine; if several names are listed as "
            "expected, naming any one of them counts."
        ),
    ),
    Question(
        id="L07",
        category="lookup",
        text=(
            "How many rows does the dataset contain for the year 2020, "
            "counting every row whatever its bs value, null included? Give "
            "the exact count."
        ),
        reference_code="""\
expected = (
    lf.filter(pl.col("year") == 2020)
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=("The response states the number of rows for 2020 as {expected}."),
        tolerance_abs=0.0,
    ),
    Question(
        id="L08",
        category="lookup",
        text=(
            "How many rows with fire year 2012 have a null bs value? Give the "
            "exact count."
        ),
        reference_code="""\
expected = (
    lf.filter(pl.col("year") == 2012)
    .select(pl.col("bs").null_count())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of 2012 rows with a null burn "
            "severity (bs) value as {expected}."
        ),
        tolerance_abs=0.0,
    ),
    # ------------------------------------------------------------------
    # Aggregations (7)
    # ------------------------------------------------------------------
    Question(
        id="A01",
        category="aggregation",
        text=(
            "What is the mean fire size, in acres, across all fire events in "
            "the dataset? Count every incident type."
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(pl.col("area_m2").first())
mean_m2 = ev.select(pl.col("area_m2").mean()).collect(engine="streaming").item()
expected = mean_m2 / 4046.8564224  # international acre
""",
        answer_kind="scalar",
        criterion=(
            "The response states the mean fire size across fire events as "
            "{expected} acres."
        ),
        tolerance_rel=0.01,
    ),
    Question(
        id="A02",
        category="aggregation",
        text=(
            "Which single year has the most rows, counting every row whatever "
            "its bs value, null included?"
        ),
        reference_code="""\
counts = lf.group_by("year").len().collect(engine="streaming")
expected = int(counts.sort("len", descending=True)["year"][0])
""",
        answer_kind="text",
        criterion=(
            "The response identifies {expected} as the year with the most rows."
        ),
    ),
    Question(
        id="A03",
        category="aggregation",
        text=(
            "For the year 2021, how many pixels fall in each burn severity "
            "class: unburned, low, moderate and high? Give the exact counts."
        ),
        reference_code="""\
counts = (
    lf.filter((pl.col("year") == 2021) & pl.col("bs").is_in([1, 2, 3, 4]))
    .group_by("bs")
    .len()
    .collect(engine="streaming")
)
pairs = sorted((r["bs"], r["len"]) for r in counts.to_dicts())
expected = {f"bs={k}": v for k, v in pairs}
""",
        answer_kind="series",
        criterion=(
            "The response reports 2021 pixel counts per burn severity class "
            "matching all of: {expected} (bs=1 is Unburned, bs=2 Low, bs=3 "
            "Moderate, bs=4 High; the response may use either the labels or "
            "the codes)."
        ),
        tolerance_abs=0.0,
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
            "Which level-1 ecoregion has the most rows in total, counting "
            "every row whatever its bs value, null included? Identify it by "
            "its eco1 integer code (the name too, if you like)."
        ),
        reference_code="""\
counts = lf.group_by("eco1").len().collect(engine="streaming")
expected = int(counts.sort("len", descending=True)["eco1"][0])
""",
        answer_kind="text",
        criterion=(
            "The response identifies eco1 code {expected} as the level-1 "
            "ecoregion with the most rows. The integer code "
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
            "over 1984-2022, what is the slope in fires per year? Count only "
            "wildfire events (Incid_Type = 1)."
        ),
        reference_code="""\
counts = (
    lf.filter(pl.col("Incid_Type") == 1)
    .group_by("year")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .sort("year")
    .collect(engine="streaming")
)
expected = ols_slope(counts["year"].to_list(), counts["n"].to_list())
""",
        answer_kind="scalar",
        criterion=(
            "The response states the ordinary least-squares slope of annual "
            "distinct wildfire-event counts against year as {expected} fires "
            "per year."
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
            "1984-2022, in square meters per year? Count only wildfire events "
            "(Incid_Type = 1)."
        ),
        reference_code="""\
ev = lf.filter(pl.col("Incid_Type") == 1).group_by("Event_ID").agg(
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
            "wildfire burned area against year as {expected} square meters "
            "per year. "
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
            "first-decade mean. Count only wildfire events (Incid_Type = 1)."
        ),
        reference_code="""\
counts = (
    lf.filter(pl.col("Incid_Type") == 1)
    .group_by("year")
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
            "distinct wildfire-event count to the 1984-1993 mean as "
            "{expected}."
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
            "(positive = increase)? Average the ten annual fractions within "
            "each decade with equal weight per year (do not pool pixels across "
            "the decade). Count only wildfire events (Incid_Type = 1)."
        ),
        reference_code="""\
frac = (
    lf.filter((pl.col("Incid_Type") == 1) & pl.col("bs").is_in([1, 2, 3, 4]))
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
            "The response states that the mean annual wildfire high-severity "
            "fraction changed by {expected} percentage points between 1984-1993 and "
            "2013-2022. The sign matters: positive means an increase."
        ),
        tolerance_abs=0.5,
    ),
    Question(
        id="T05",
        category="trend",
        text=(
            "Using each distinct wildfire event's ignition date "
            "(Incid_Type = 1 only; one value per Event_ID), compute the mean "
            "day-of-year of Ig_Date for events with year 1984-1993 and for "
            "2013-2022. Report the recent-decade mean minus the early-decade "
            "mean, in days (negative = earlier)."
        ),
        reference_code="""\
ev = lf.filter(pl.col("Incid_Type") == 1).group_by("Event_ID").agg(
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
            "The response states that the 2013-2022 mean wildfire ignition "
            "day-of-year minus the 1984-1993 mean is {expected} days. The sign "
            "matters: negative means earlier in the recent decade."
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
            "Of the 10 largest distinct fire events by area_m2, how many have "
            "a mean pixel latitude north of 42.0 degrees (lat > 42.0)?"
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("area_m2").first(),
    pl.col("lat").mean(),
)
top = ev.collect(engine="streaming").top_k(10, by="area_m2")
expected = int((top["lat"] > 42.0).sum())
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} of the 10 largest fire events "
            "have a mean pixel latitude north of 42.0 degrees (lat > 42.0)."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="M02",
        category="multistep",
        text=(
            "Within eco1 = 11 (Mediterranean California), how many distinct "
            "pixel locations (unique geohash) appear in 3 or more distinct fire "
            "events (unique Event_ID), counting every row regardless of bs "
            "value or Incid_Type? Give the exact count."
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
            "Mediterranean California (eco1 = 11) appear in 3 or more distinct "
            "fire events."
        ),
        tolerance_abs=0.0,
        expensive_gt=True,
    ),
    Question(
        id="M03",
        category="multistep",
        text=(
            "Consider the single largest fire event by area_m2 among events "
            "with year = 2002. What percentage of its pixels with bs in 1-4 "
            "burned at high severity (bs = 4)? One decimal place."
        ),
        reference_code="""\
ev = (
    lf.filter(pl.col("year") == 2002)
    .group_by("Event_ID")
    .agg(pl.col("area_m2").first())
)
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
            "fire event of 2002 as {expected} percent."
        ),
        tolerance_abs=0.5,
    ),
    Question(
        id="M04",
        category="multistep",
        text=(
            "Which year has the most distinct fire events, and what is the "
            "incident name of the largest fire (by area_m2) ignited in that "
            "year? Count every incident type. Give both the year and the "
            "fire's name."
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
names = ev.filter(pl.col("area_m2") == mx)["Incid_Name"].unique().to_list()
# Compound answer {year, name}, rendered as "YEAR / NAME" strings so the
# existing text-kind criterion can require both parts.
expected = sorted(f"{peak_year} / {n}" for n in names)
""",
        answer_kind="text",
        criterion=(
            "The response gives both the peak year and the name of its largest "
            "fire, as {expected} (year / incident name). Both parts are "
            "required: the year alone or the name alone means 'no'. Case and "
            "punctuation differences in the name are fine; if several "
            "year / name pairs are listed as expected, giving any one of them "
            "counts."
        ),
    ),
    Question(
        id="M05",
        category="multistep",
        text=(
            "Among the 20 largest distinct fire events by area_m2, how many "
            "have more than 1% of their pixels inside the WUI "
            "(wui_bool = 1)?"
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg(
    pl.col("area_m2").first(),
    pl.col("wui_bool").mean().alias("wui_frac"),
)
top = ev.collect(engine="streaming").top_k(20, by="area_m2")
expected = int((top["wui_frac"] > 0.01).sum())
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} of the 20 largest fire events "
            "have more than 1% of their pixels inside the WUI."
        ),
        tolerance_abs=0.0,
    ),
]

CATEGORIES = ("lookup", "aggregation", "trend", "multistep")

assert len(QUESTIONS) == 25
assert len({q.id for q in QUESTIONS}) == 25, "duplicate question ids"


# ----------------------------------------------------------------------
# Extended tier (14): semantics, behaviour, interaction, multi-step
# ----------------------------------------------------------------------
#
# Every question that counts pixel rows states its burn-severity scope ("bs =
# 4", "whatever its bs value, null included") except E07, whose ambiguity is
# the point: "burned-pixel rows" alone was read as bs 2-4 in the core run.

# E12's minimum row count. Provisional: it is set from the full-data per-eco1
# table E12's reference logs, so that the eco1 with the highest share overall
# falls below it and the winner keeps a 0.5 pp margin.
ECO1_SHARE_MIN_ROWS = 1_000_000

# E08's scripted history: the 5 largest distinct fire events of 2020 and the
# Event_IDs behind that table. Both are filled together from the full-data
# table that E08's reference prints while the IDs are empty (the reference
# refuses to compute an answer until they match), so the history always shows
# the events the answer is scored on.
TOP5_2020_EVENT_IDS: tuple[str, ...] = ()
TOP5_2020_HISTORY: tuple[tuple[str, str], ...] = (
    (
        "user",
        "What were the 5 largest distinct fire events ignited in 2020, by "
        "area_m2? Count every incident type.",
    ),
    (
        "model",
        "The 5 largest distinct fire events ignited in 2020, by area_m2, are:"
        "\n\n| Incid_Name | Event_ID | area_m2 |\n|---|---|---|\n",
    ),
)

EXTENDED: list[Question] = [
    Question(
        id="E01",
        category="ext_semantics",
        text="How many prescribed fires were there in 2020? Give the exact count.",
        reference_code="""\
expected = (
    lf.filter((pl.col("Incid_Type") == 2) & (pl.col("year") == 2020))
    .select(pl.col("Event_ID").n_unique())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of distinct prescribed-fire events "
            "(Incid_Type = 2) in 2020 as {expected}."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="E02",
        category="ext_semantics",
        text=(
            "How many high-severity (bs = 4) pixel rows burned in "
            "Grassland/Herbaceous land cover, using the land cover at the time "
            "each pixel burned? Count every incident type and give the exact "
            "count."
        ),
        reference_code="""\
expected = (
    lf.filter((pl.col("bs") == 4) & (pl.col("nlcd") == 71))
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of bs = 4 rows with nlcd = 71 as "
            "{expected}."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="E03",
        category="ext_semantics",
        text=(
            "How many high-severity (bs = 4) pixel rows burned in "
            "Grassland/Herbaceous land cover, using each row's dominant land "
            "cover across the whole study period rather than the land cover in "
            "the year it burned? Count every incident type and give the exact "
            "count."
        ),
        reference_code="""\
counts = (
    lf.filter(pl.col("bs") == 4)
    .select(
        (pl.col("nlcd") == 71).sum().alias("at_burn"),
        (pl.col("nlcd_mode") == 71).sum().alias("mode"),
    )
    .collect(engine="streaming")
)
at_burn, expected = counts["at_burn"][0], counts["mode"][0]
# The pair only separates the two land-cover columns if their counts differ.
assert at_burn > 1.5 * expected, f"nlcd {at_burn} vs nlcd_mode {expected}"
""",
        answer_kind="scalar",
        criterion=(
            "The response states the number of bs = 4 rows with nlcd_mode = 71 "
            "as {expected}."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="E04",
        category="ext_behavior",
        text="What was the total suppression cost of the fires in 2020?",
        reference_code="""\
names = lf.collect_schema().names()
money = [
    c for c in names
    if any(w in c.casefold() for w in ("cost", "suppress", "usd", "dollar"))
]
assert not money, f"the dataset has a cost-like column: {money}"
expected = {"abstain": True}
""",
        answer_kind="abstain",
        criterion=(
            "The response says the dataset has no suppression-cost data and "
            "gives no cost figure as the answer."
        ),
    ),
    Question(
        id="E05",
        category="ext_behavior",
        text=("How many fire events were there in 2023? Count every incident type."),
        reference_code="""\
n = (
    lf.filter((pl.col("year") == 2023) | (pl.col("Ig_Date").dt.year() == 2023))
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)
assert n == 0, f"{n} rows fall in 2023"
expected = {"abstain": True}
""",
        answer_kind="abstain",
        criterion=(
            "The response says the dataset does not cover 2023 and gives no "
            "2023 event count as the answer."
        ),
    ),
    Question(
        id="E14",
        category="ext_semantics",
        text=(
            "What percentage of distinct fire events have at least one row "
            "(whatever its bs value, null included) inside the WUI "
            "(wui_bool = 1)? Count every incident type."
        ),
        reference_code="""\
ev = lf.group_by("Event_ID").agg((pl.col("wui_bool") == 1).any().alias("wui"))
expected = ev.select(pl.col("wui").mean()).collect(engine="streaming").item() * 100
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} percent of distinct fire "
            "events have at least one row inside the WUI."
        ),
        tolerance_abs=0.1,
    ),
    Question(
        id="E09",
        category="ext_interaction",
        text=(
            "Plot the annual number of distinct fire events from 1984 to 2022 "
            "as a line chart, then tell me which year had the most events and "
            "how many. Count every incident type."
        ),
        reference_code="""\
counts = (
    lf.filter(pl.col("year").is_between(1984, 2022))
    .group_by("year")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .collect(engine="streaming")
    .sort("n", descending=True)
)
assert counts.height == 1 or counts["n"][0] > counts["n"][1], "tie for most events"
expected = {"year": int(counts["year"][0]), "value": int(counts["n"][0])}
""",
        answer_kind="plot",
        criterion=(
            "The response plots annual distinct fire events and names the peak "
            "year and its count as {expected}."
        ),
        tolerance_abs=0.0,
    ),
    Question(
        id="E06",
        category="ext_behavior",
        text=(
            "List every distinct fire event larger than 200,000,000 m\u00b2 "
            "(area_m2 > 200000000) with its incident name, year and area_m2, "
            "largest first. Count every incident type."
        ),
        reference_code="""\
ev = (
    lf.group_by("Event_ID")
    .agg(
        pl.col("Incid_Name").first(),
        pl.col("year").first(),
        pl.col("area_m2").first(),
    )
    .filter(pl.col("area_m2") > 200_000_000)
    .collect(engine="streaming")
    .sort("area_m2", descending=True, nulls_last=True)
)
areas = ev["area_m2"].to_list()[:11]
# A tie in the first 11 would make the top-10 names order-dependent.
assert all(a > b for a, b in zip(areas, areas[1:])), "tied areas in the top 11"
expected = {"count": ev.height, "top": ev["Incid_Name"].to_list()[:10]}
""",
        answer_kind="disclosure",
        criterion=(
            "The response gives the number of qualifying events as {expected} "
            "(count) with the largest first, or says the list it shows is "
            "incomplete."
        ),
    ),
    Question(
        id="E11",
        category="ext_multistep",
        text=(
            "What percentage of all rows (every row whatever its bs value, null "
            "included) belong to the largest 1% of distinct fire events by "
            "area_m2? Take the top k events with k = ceil(0.01 \u00d7 number of "
            "distinct events). Count every incident type."
        ),
        reference_code="""\
ev = (
    lf.group_by("Event_ID")
    .agg(pl.col("area_m2").first(), pl.len().alias("rows"))
    .collect(engine="streaming")
    .sort("area_m2", descending=True, nulls_last=True)
)
k = (ev.height + 99) // 100  # ceil(0.01 * events) without float error
areas = ev["area_m2"]
assert k == ev.height or areas[k - 1] != areas[k], "tie at the top-k boundary"
expected = ev["rows"][:k].sum() / ev["rows"].sum() * 100
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} percent of all rows belong to "
            "the largest 1% of distinct fire events."
        ),
        tolerance_abs=0.5,
    ),
    Question(
        id="E08",
        category="ext_interaction",
        text=(
            "Of those five, which has the highest share of high-severity pixels "
            "(bs = 4 among its pixels with bs 1-4)? Give its incident name."
        ),
        reference_code=f"GOLD = {TOP5_2020_EVENT_IDS!r}\n"
        + """\
top = (
    lf.filter(pl.col("year") == 2020)
    .group_by("Event_ID")
    .agg(pl.col("Incid_Name").first(), pl.col("area_m2").first())
    .collect(engine="streaming")
    .sort("area_m2", descending=True, nulls_last=True)
    .head(6)
)
assert top["area_m2"][4] != top["area_m2"][5], "tie at rank 5 of the 2020 top 5"
top = top.head(5)
if tuple(top["Event_ID"]) != GOLD:
    rows = [
        f"| {r['Incid_Name']} | {r['Event_ID']} | {r['area_m2']:.0f} |"
        for r in top.iter_rows(named=True)
    ]
    table = "\\n".join(["| Incid_Name | Event_ID | area_m2 |", "|---|---|---|", *rows])
    raise ValueError(
        "the 2020 top 5 does not match the scripted history; put these "
        "Event_IDs in TOP5_2020_EVENT_IDS and these rows in "
        f"TOP5_2020_HISTORY:\\n{table}"
    )
shares = (
    lf.filter(pl.col("Event_ID").is_in(list(GOLD)) & pl.col("bs").is_in([1, 2, 3, 4]))
    .group_by("Event_ID")
    .agg(pl.col("Incid_Name").first(), (pl.col("bs") == 4).mean().alias("share"))
    .collect(engine="streaming")
    .sort("share", descending=True)
)
assert shares.height == 1 or shares["share"][0] > shares["share"][1], "tied shares"
expected = [shares["Incid_Name"][0]]
""",
        answer_kind="text",
        criterion=(
            "The response names {expected} as the one of the five with the "
            "highest high-severity share."
        ),
        history=TOP5_2020_HISTORY,
    ),
    Question(
        id="E12",
        category="ext_multistep",
        text=(
            "Among level-1 ecoregions (eco1) with at least "
            f"{ECO1_SHARE_MIN_ROWS:,} rows (every row whatever its bs value, "
            "null included), which has the highest share of high-severity "
            "pixels (bs = 4 among pixels with bs 1-4)? Give the eco1 code. "
            "Count every incident type."
        ),
        reference_code=f"min_rows = {ECO1_SHARE_MIN_ROWS}\n"
        + """\
eco = (
    lf.filter(pl.col("eco1").is_not_null())
    .group_by("eco1")
    .agg(
        pl.len().alias("rows"),
        (pl.col("bs") == 4).sum().alias("high"),
        pl.col("bs").is_in([1, 2, 3, 4]).sum().alias("rated"),
    )
    .collect(engine="streaming")
    .filter(pl.col("rated") > 0)
    .with_columns((pl.col("high") / pl.col("rated") * 100).alias("share"))
    .sort("share", descending=True)
)
logger.info(
    "per-eco1 high-severity share (eco1: rows, bs==4, bs 1-4, share %%):\\n%s",
    "\\n".join(
        f"{r['eco1']}: {r['rows']}, {r['high']}, {r['rated']}, {r['share']:.3f}"
        for r in eco.iter_rows(named=True)
    ),
)
overall = eco.row(0, named=True)
assert overall["rows"] < min_rows, (
    f"min_rows {min_rows} does not exclude eco1 {overall['eco1']}, the highest "
    "share overall"
)
eligible = eco.filter(pl.col("rows") >= min_rows)
assert eligible.height > 0, f"no eco1 has {min_rows} rows"
if eligible.height > 1:
    margin = eligible["share"][0] - eligible["share"][1]
    assert margin >= 0.5, f"winner leads by {margin:.3f} pp, under 0.5"
expected = int(eligible["eco1"][0])
""",
        answer_kind="text",
        criterion=(
            "The response identifies eco1 code {expected} as the qualifying "
            "level-1 ecoregion with the highest high-severity share."
        ),
    ),
    Question(
        id="E13",
        category="ext_semantics",
        text=(
            "Which level-II ecoregion has the most distinct fire events? Count "
            "a fire in every level-II ecoregion where it has a row (whatever its "
            "bs value, null included), count every incident type, and give its "
            "eco2s code."
        ),
        reference_code="""\
counts = (
    lf.filter(pl.col("eco2s").is_not_null())
    .select("eco2s", "Event_ID")
    .unique()
    .group_by("eco2s")
    .len()
    .collect(engine="streaming")
    .sort("len", descending=True)
)
assert counts.height == 1 or counts["len"][0] > counts["len"][1], "tie for most events"
expected = counts["eco2s"][0]
""",
        answer_kind="text",
        criterion=(
            "The response identifies eco2s code {expected} as the level-II "
            "ecoregion with the most distinct fire events."
        ),
        expensive_gt=True,
    ),
    Question(
        id="E07",
        category="ext_behavior",
        text="Which ecoregion burned the most? Count every incident type.",
        reference_code="""\
# Every defensible reading: level 1-3, and "burned the most" as rows (every bs
# value, bs 1-4, bs 2-4), summed distinct-event area_m2, or events (counted in
# every region they touch, or in the region holding most of their rows).
# Expected maps "<level>|<measure>" to the winners of all that measure's
# readings at that level.
cells = (
    lf.select("eco1", "eco2s", "eco3s", "Event_ID", "bs", "area_m2")
    .group_by("eco1", "eco2s", "eco3s", "Event_ID")
    .agg(
        pl.len().alias("rows_all"),
        pl.col("bs").is_in([1, 2, 3, 4]).sum().alias("rows_bs1_4"),
        pl.col("bs").is_in([2, 3, 4]).sum().alias("rows_bs2_4"),
        pl.col("area_m2").first(),
    )
    .collect(engine="streaming")
)
expected = {}
lines = []
for level, col in (("level1", "eco1"), ("level2", "eco2s"), ("level3", "eco3s")):
    per = (
        cells.filter(pl.col(col).is_not_null())
        .group_by(col, "Event_ID")
        .agg(
            pl.col("rows_all", "rows_bs1_4", "rows_bs2_4").sum(),
            pl.col("area_m2").first(),
        )
    )
    dominant = (
        per.sort(["rows_all", col], descending=[True, False])
        .group_by("Event_ID", maintain_order=True)
        .first()
    )
    readings = {
        "pixels_or_area": {
            "rows, every bs value": per.group_by(col).agg(pl.col("rows_all").sum()),
            "rows, bs 1-4": per.group_by(col).agg(pl.col("rows_bs1_4").sum()),
            "rows, bs 2-4": per.group_by(col).agg(pl.col("rows_bs2_4").sum()),
            "distinct-event area_m2": per.group_by(col).agg(pl.col("area_m2").sum()),
        },
        "events": {
            "events, every region touched": per.group_by(col).len(),
            "events, dominant region": dominant.group_by(col).len(),
        },
    }
    for measure, variants in readings.items():
        answers = set()
        for name, frame in variants.items():
            frame = frame.rename({frame.columns[1]: "v"}).sort("v", descending=True)
            top = frame["v"][0]
            winners = sorted(str(c) for c in frame.filter(pl.col("v") == top)[col])
            rest = frame.filter(pl.col("v") != top)
            second = f"{rest[col][0]} ({rest['v'][0]})" if rest.height else "none"
            lines.append(
                f"{level} {measure} [{name}] -> {', '.join(winners)} ({top}); "
                f"next {second}"
            )
            answers.update(winners)
        assert answers, f"no answer for {level} {measure}"
        expected[f"{level}|{measure}"] = sorted(answers)
logger.info("ecoregion readings:\\n%s", "\\n".join(lines))
""",
        answer_kind="clarify_or_assume",
        criterion=(
            "The response asks which ecoregion level and measure are meant, or "
            "states the level and measure it assumed and names that reading's "
            "winner: {expected}."
        ),
        expensive_gt=True,
    ),
    Question(
        id="E10",
        category="ext_multistep",
        text=(
            "Within Mediterranean California (eco1 = 11), take each pixel "
            "location (geohash) and the distinct fire events with a row at that "
            "location (whatever its bs value, null included), ordered by "
            "ignition date (Ig_Date). What percentage of the intervals between "
            "consecutive fires at the same location are 1,826 days or shorter? "
            "Count every incident type."
        ),
        reference_code="""\
# Event_ID hashed to a u64: the unique (location, event) set is the memory peak.
fires = (
    lf.filter(pl.col("eco1") == 11)
    .select("geohash", pl.col("Event_ID").hash().alias("event"), "Ig_Date")
    .unique()
    .collect(engine="streaming")
    .sort("geohash", "Ig_Date")
)
assert (fires["Ig_Date"].dt.truncate("1d") == fires["Ig_Date"]).all(), (
    "Ig_Date has a time of day"
)
gaps = fires.select(
    pl.col("Ig_Date").diff().over("geohash").dt.total_days().alias("days"),
    pl.col("Ig_Date").dt.year().cast(pl.Int32).diff().over("geohash").alias("years"),
).drop_nulls()
expected = (gaps["days"] <= 1826).mean() * 100
# Reading "5 years" as a year difference must land outside the band, and
# same-date events must be too rare for "consecutive" to need defining.
by_years = (gaps["years"] <= 5).mean() * 100
same_day = (gaps["days"] == 0).mean() * 100
assert abs(by_years - expected) > 2, f"year-based {by_years:.2f} vs {expected:.2f}"
assert same_day < 0.5, f"{same_day:.2f}% of intervals are same-day"
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} percent of intervals between "
            "consecutive fires at the same eco1 = 11 location are 1,826 days or "
            "shorter."
        ),
        tolerance_abs=1.0,
        expensive_gt=True,
    ),
]
EXTENDED_CATEGORIES = (
    "ext_semantics",
    "ext_behavior",
    "ext_interaction",
    "ext_multistep",
)

assert len(EXTENDED) == 14
assert EXTENDED[-1].id == "E10", "the heaviest reference runs last"
assert {q.category for q in EXTENDED} == set(EXTENDED_CATEGORIES)

BY_ID: dict[str, Question] = {q.id: q for q in (*QUESTIONS, *EXTENDED)}
assert len(BY_ID) == len(QUESTIONS) + len(EXTENDED), "duplicate question ids"


def tier_of(q: Question) -> str:
    """'core' or 'extended'; one run dir holds one tier."""
    return "extended" if q.category in EXTENDED_CATEGORIES else "core"


def select(spec: str | None) -> list[Question]:
    """Select questions by tier, category name, or comma-separated ids.

    None, "" or "core" selects the 25 core questions; "extended" the extended
    tier; "all" both; a category name of either tier that category; otherwise
    comma-separated ids from either tier ("L01,E04").
    """
    if not spec or spec == "core":
        return list(QUESTIONS)
    if spec == "extended":
        return list(EXTENDED)
    if spec == "all":
        return [*QUESTIONS, *EXTENDED]
    if spec in CATEGORIES or spec in EXTENDED_CATEGORIES:
        return [q for q in (*QUESTIONS, *EXTENDED) if q.category == spec]
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token not in BY_ID:
            raise KeyError(f"unknown question id or category: {token!r}")
        out.append(BY_ID[token])
    return out

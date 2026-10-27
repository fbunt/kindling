"""The benchmark questions: the 25-question core tier, the extended tier and
the hard tier.

Each Question pairs a natural-language prompt (sent to the model verbatim) with
hand-written reference Polars code that defines ground truth. Reference code is
executed HOST-SIDE by bench/ground_truth.py in a namespace containing:

- `lf`  — the LazyFrame built by app.query_engine._build_lazyframe (imported,
          never reimplemented, so the scan can't diverge)
- `pl`  — polars
- `ols_slope(xs, ys)` — plain-Python ordinary-least-squares slope
- `logger` - the ground-truth logger, for tables a reference reports

Conventions: every `.collect()` uses engine="streaming" (the full dataset is
749M rows); the code must assign `expected`, which is normalized to a
JSON-serializable scalar, dict (series), or sorted list (set / tie-safe text).

Questions are phrased to fix a single defensible interpretation (dedup rule,
decade bounds, regression method, units) — the paper's accuracy claim depends
on the question having one right answer. The extended tier also measures
behaviour (abstaining, disclosing truncation, stating an assumption), and the
hard tier asks questions whose obvious query gives a plausible wrong number;
each tier is run and reported apart (TIERS).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str  # L01..L08, A01..A07, T01..T05, M01..M05; extended E01..E14; hard H01..H09
    category: str  # a CATEGORIES, EXTENDED_CATEGORIES or HARD_CATEGORIES name
    text: str  # prompt sent to the model, verbatim
    reference_code: str  # host-executed Polars; must assign `expected`
    # core: scalar | series | set | text; extended adds abstain | disclosure |
    # clarify_or_assume | plot; hard adds fields | disambiguate
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

# E12's row ceiling, set from the full-data per-eco1 table E12's reference
# logs. On the full data the eco1 with the highest share overall (6) also has
# the most rows, so only an upper bound can make the filter change the answer:
# this one excludes 6, 10, 9 and 8 (110M+ rows each) and leaves eco1 7 ahead
# of 11 by 3.9 pp.
ECO1_SHARE_MAX_ROWS = 100_000_000

# E08's scripted history: the 5 largest distinct fire events of 2020 and the
# Event_IDs behind that table. Both are filled together from the full-data
# table that E08's reference prints while the IDs are empty (the reference
# refuses to compute an answer until they match), so the history always shows
# the events the answer is scored on.
TOP5_2020_EVENT_IDS: tuple[str, ...] = (
    "CA3966012280920200817",
    "CA3742412156820200816",
    "CA3720111927220200905",
    "CA4009112093120200817",
    "CA3850412233720200817",
)
TOP5_2020_HISTORY: tuple[tuple[str, str], ...] = (
    (
        "user",
        "What were the 5 largest distinct fire events ignited in 2020, by "
        "area_m2? Count every incident type.",
    ),
    (
        "model",
        "The 5 largest distinct fire events ignited in 2020, by area_m2, are:"
        "\n\n| Incid_Name | Event_ID | area_m2 |\n|---|---|---|\n"
        "| AUGUST COMPLEX | CA3966012280920200817 | 4325251517 |\n"
        "| SCU LIGHTNING COMPLEX | CA3742412156820200816 | 1642198721 |\n"
        "| CREEK | CA3720111927220200905 | 1543673733 |\n"
        "| NORTH COMPLEX | CA4009112093120200817 | 1281010893 |\n"
        "| HENNESSEY | CA3850412233720200817 | 1271642042 |",
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
            "Among level-1 ecoregions (eco1) with fewer than "
            f"{ECO1_SHARE_MAX_ROWS:,} rows (every row whatever its bs value, "
            "null included), which has the highest share of high-severity "
            "pixels (bs = 4 among pixels with bs 1-4)? Give the eco1 code. "
            "Count every incident type."
        ),
        reference_code=f"max_rows = {ECO1_SHARE_MAX_ROWS}\n"
        + """\
eco = (
    lf.filter(pl.col("eco1").is_not_null())
    .group_by("eco1")
    .agg(
        pl.len().alias("rows"),
        (pl.col("bs") == 4).sum().alias("high"),
        pl.col("bs").is_between(1, 4).sum().alias("rated"),
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
assert overall["rows"] >= max_rows, (
    f"max_rows {max_rows} does not exclude eco1 {overall['eco1']}, the highest "
    "share overall"
)
eligible = eco.filter(pl.col("rows") < max_rows)
assert eligible.height > 0, f"no eco1 has fewer than {max_rows} rows"
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
        pl.col("bs").is_between(1, 4).sum().alias("rows_bs1_4"),
        pl.col("bs").is_between(2, 4).sum().alias("rows_bs2_4"),
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

# ----------------------------------------------------------------------
# Hard tier (9): semantics, per-pixel history, statistics, interaction
# ----------------------------------------------------------------------
#
# Questions where the obvious query returns a plausible wrong number, or that
# need three or more dependent stages, often per pixel location. Every
# row-count wording states its bs scope; every event count states its
# incident-type scope. Each reference asserts its runner-up margin and that
# the named wrong readings land outside the band (or logs them where a
# coincidence would be harmless).
#
# Per-pixel references loop over the 16 partitions geohash % 16 == k, which
# is exact because locations are independent and keeps each collect small.
# No is_in inside a streaming group_by().agg(); counts are cast to Int64
# before any subtraction (pl.len() and sum() of a flag are unsigned); a bs
# comparison is null for a null bs, so flags built from it are filled False.

# H04's year: one with at least 3 wildfire (Incid_Type = 1) events named
# exactly COTTONWOOD, set from the full-data per-year table that H04's
# reference logs while this is unset (it refuses to compute an answer until
# the year qualifies).
COTTONWOOD_YEAR: int | None = 1994

# H06's scripted history states the number of distinct wildfire events with a
# row in eco1 = 8 in 2020. Set from the count H06's reference reports while
# this is unset (it refuses to compute an answer until they match), so the
# history always states the number the follow-up builds on.
ECO8_2020_WILDFIRE_EVENTS: int | None = 28


def _filled(value, placeholder: str, fmt: str = "") -> str:
    """A constant as written into question text; its placeholder while unset."""
    return placeholder if value is None else format(value, fmt)


ECO8_2020_HISTORY: tuple[tuple[str, str], ...] = (
    (
        "user",
        "How many wildfire events (Incid_Type = 1) burned in Eastern Temperate "
        "Forests (eco1 = 8) in 2020?",
    ),
    (
        "model",
        f"{_filled(ECO8_2020_WILDFIRE_EVENTS, '[N]', ',')} distinct wildfire "
        "events (Incid_Type = 1) had at least one row in eco1 = 8 in 2020.",
    ),
)

# The 11 western states of H07, as Event_ID prefixes.
WESTERN_STATES = ("AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY")

HARD: list[Question] = [
    Question(
        id="H01",
        category="hard_semantics",
        text=(
            "Which US state had the most distinct prescribed-fire events "
            "(Incid_Type = 2) from 2010 through 2019, and how many did it have? "
            "Count each event once, in the state MTBS records for it."
        ),
        reference_code="""\
# The state is only in the Event_ID prefix (the state of the ignition point).
events = (
    lf.filter((pl.col("Incid_Type") == 2) & pl.col("year").is_between(2010, 2019))
    .select("Event_ID")
    .unique()
    .collect(engine="streaming")
)
assert events["Event_ID"].null_count() == 0, "a prescribed fire has no Event_ID"
states = events.select(pl.col("Event_ID").str.slice(0, 2).alias("state"))
assert states["state"].str.contains("^[A-Z]{2}$").all(), (
    "an Event_ID does not start with a two-letter state code"
)
counts = (
    states.group_by("state")
    .len()
    .sort(["len", "state"], descending=[True, False])
)
logger.info(
    "prescribed-fire events 2010-2019 by Event_ID state (top 5): %s",
    ", ".join(f"{r['state']} {r['len']}" for r in counts.head(5).iter_rows(named=True)),
)
assert counts.height > 0, "no prescribed-fire events in 2010-2019"
if counts.height > 1:
    assert counts["len"][0] > counts["len"][1], (
        f"tie for most events: {counts['state'][0]} and {counts['state'][1]}"
    )
expected = {"state": counts["state"][0], "count": int(counts["len"][0])}
""",
        answer_kind="fields",
        criterion=(
            "The response names the state with the most distinct prescribed-fire "
            "events in 2010-2019 and its count as {expected}."
        ),
    ),
    Question(
        id="H04",
        category="hard_semantics",
        text=(
            "How many acres did the COTTONWOOD fire of "
            f"{_filled(COTTONWOOD_YEAR, '[YEAR]')} burn?"
        ),
        reference_code=f"YEAR = {COTTONWOOD_YEAR!r}\n"
        + """\
named = (
    lf.filter(pl.col("Incid_Name") == "COTTONWOOD")
    .group_by("Event_ID")
    .agg(
        pl.col("year").first(),
        pl.col("Incid_Type").first(),
        pl.col("area_m2").first(),
    )
    .collect(engine="streaming")
    .sort("year", "Event_ID")
)
by_year = (
    named.group_by("year")
    .agg(
        pl.len().alias("events"),
        (pl.col("Incid_Type") == 1).sum().alias("wildfires"),
    )
    .sort("year")
)
table = "\\n".join(
    ["| year | events | Incid_Type = 1 |", "|---|---|---|"]
    + [
        f"| {r['year']} | {r['events']} | {r['wildfires']} |"
        for r in by_year.iter_rows(named=True)
    ]
)
logger.info("events named exactly COTTONWOOD per year:\\n%s", table)
eligible = by_year.filter(pl.col("wildfires") >= 3)["year"].to_list()
if YEAR not in eligible:
    raise ValueError(
        f"COTTONWOOD_YEAR is {YEAR!r}; set it to a year with at least 3 "
        f"wildfire (Incid_Type = 1) events named exactly COTTONWOOD "
        f"({eligible or 'none qualify'}):\\n{table}"
    )
events = named.filter(pl.col("year") == YEAR)
near = (
    lf.filter(
        (pl.col("year") == YEAR)
        & pl.col("Incid_Name").str.contains("COTTONWOOD")
        & (pl.col("Incid_Name") != "COTTONWOOD")
    )
    .select("Incid_Name")
    .unique()
    .collect(engine="streaming")["Incid_Name"]
    .sort()
    .to_list()
)
logger.info("%d near-names of COTTONWOOD in %d: %s", len(near), YEAR, near)
expected = {
    "year": YEAR,
    "count": events.height,
    "events": [
        {
            "Event_ID": r["Event_ID"],
            "Incid_Type": r["Incid_Type"],
            "acres": r["area_m2"] / 4046.8564224,  # international acre
        }
        for r in events.iter_rows(named=True)
    ],
}
""",
        answer_kind="disambiguate",
        criterion=(
            "The response says more than one event of that year is named "
            "COTTONWOOD ({expected}) and does not present one figure as the "
            "fire's area."
        ),
    ),
    Question(
        id="H06",
        category="hard_interaction",
        text=(
            "Now include prescribed fires (Incid_Type = 2) too, and keep only "
            "events with at least one pixel row (any bs value) in the WUI "
            "(wui_bool = 1) within that region."
        ),
        reference_code=f"N = {ECO8_2020_WILDFIRE_EVENTS!r}\n"
        + """\
ev = (
    lf.group_by("Event_ID")
    .agg(
        pl.col("Incid_Type").first(),
        pl.col("year").first(),
        (pl.col("eco1") == 8).fill_null(False).any().alias("in_region"),
        ((pl.col("eco1") == 8) & (pl.col("wui_bool") == 1))
        .fill_null(False)
        .any()
        .alias("wui_in_region"),
        (pl.col("wui_bool") == 1).fill_null(False).any().alias("wui_anywhere"),
    )
    .collect(engine="streaming")
)
wildfire = pl.col("Incid_Type") == 1
both = wildfire | (pl.col("Incid_Type") == 2)
in_2020 = pl.col("year") == 2020


def count(mask):
    return ev.filter(mask).height


history = count(wildfire & in_2020 & pl.col("in_region"))
if N != history:
    raise ValueError(
        f"the scripted history states {N!r} events, but {history} distinct "
        "wildfire events (Incid_Type = 1) have a row in eco1 = 8 in 2020; set "
        f"ECO8_2020_WILDFIRE_EVENTS to {history}"
    )
expected = count(both & in_2020 & pl.col("wui_in_region"))
asserted = {
    "eco1 = 8 dropped": count(both & in_2020 & pl.col("wui_anywhere")),
    "2020 dropped": count(both & pl.col("wui_in_region")),
    "wildfire only": count(wildfire & in_2020 & pl.col("wui_in_region")),
}
# A tie with these only means the misreading is harmless.
logged = {
    "every incident type": count(in_2020 & pl.col("wui_in_region")),
    "WUI anywhere": count(
        both & in_2020 & pl.col("in_region") & pl.col("wui_anywhere")
    ),
}
logger.info(
    "H06: history %d, answer %d; readings %s",
    history,
    expected,
    {**asserted, **logged},
)
for name, n in asserted.items():
    assert n != expected, f"reading '{name}' gives the same count ({n})"
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} distinct wildfire and "
            "prescribed-fire events of 2020 have a WUI row in eco1 = 8."
        ),
        tolerance_abs=0.0,
        history=ECO8_2020_HISTORY,
    ),
    Question(
        id="H07",
        category="hard_statistics",
        text=(
            "Take fire events whose Event_ID begins with AZ, CA, CO, ID, MT, NM, "
            "NV, OR, UT, WA or WY, excluding prescribed fires (keep Incid_Type "
            "0, 1 and 3), with area_m2 greater than 4,050,000 and year 1984 "
            "through 2011. Count distinct events per year, with every one of "
            "the 28 years included (zero if none). What is the Theil-Sen slope "
            "of the annual count against year, in fires per year: the median "
            "of the 378 slopes between every pair of years, averaging the two "
            "middle slopes? Give two decimals."
        ),
        reference_code=f"STATES = {list(WESTERN_STATES)!r}\n"
        + """\
import statistics

per_year = (
    lf.filter(
        pl.col("Event_ID").str.slice(0, 2).is_in(STATES)
        & (
            (pl.col("Incid_Type") == 0)
            | (pl.col("Incid_Type") == 1)
            | (pl.col("Incid_Type") == 3)
        )
        & (pl.col("area_m2") > 4_050_000)
        & pl.col("year").is_between(1984, 2011)
    )
    .group_by("year")
    .agg(pl.col("Event_ID").n_unique().alias("n"))
    .collect(engine="streaming")
)
counts = dict(zip(per_year["year"].to_list(), per_year["n"].to_list()))
years = list(range(1984, 2012))
ys = [int(counts.get(y, 0)) for y in years]
slopes = sorted(
    (ys[j] - ys[i]) / (years[j] - years[i])
    for i in range(len(years))
    for j in range(i + 1, len(years))
)
assert len(slopes) == 378, len(slopes)
expected = statistics.median(slopes)  # 378 slopes: the mean of the middle two
readings = {
    "median_low": statistics.median_low(slopes),
    "median_high": statistics.median_high(slopes),
    "OLS": ols_slope(years, ys),
}
try:
    import numpy as np
    from sklearn.linear_model import TheilSenRegressor
except ImportError:
    logger.warning(
        "scikit-learn is not installed: the spatial-median Theil-Sen reading "
        "(sklearn TheilSenRegressor) is not checked"
    )
else:
    fit = TheilSenRegressor(random_state=0).fit(
        np.array(years, dtype=float).reshape(-1, 1), np.array(ys, dtype=float)
    )
    readings["sklearn TheilSenRegressor"] = float(fit.coef_[0])
logger.info("H07 annual counts %s; slope %.4f; readings %s", ys, expected, readings)
for name, value in readings.items():
    assert abs(value - expected) > 0.01, (
        f"reading '{name}' gives {value:.4f}, within 0.01 of {expected:.4f}"
    )
""",
        answer_kind="scalar",
        criterion=(
            "The response states the Theil-Sen slope of annual large western "
            "non-prescribed fire counts, 1984-2011, as {expected} fires per year."
        ),
        tolerance_abs=0.01,
    ),
    Question(
        id="H08",
        category="hard_statistics",
        text=(
            "Assign each distinct fire event to the level-1 ecoregion (eco1) "
            "holding the most of its rows (every row whatever its bs value, "
            "null included; a tie goes to the smaller eco1 code). Group eco1 6, "
            "7, 10, 11, 12 and 13 as the West. Count every incident type. Count "
            "West events per year, take the median annual count over 1984 "
            "through 1999 and over 2005 through 2018 (each window has an even "
            "number of years: average the two middle counts), and give the "
            "ratio of the 2005-2018 median to the 1984-1999 median, to two "
            "decimals."
        ),
        reference_code="""\
import statistics

WEST = [6, 7, 10, 11, 12, 13]
cells = (
    lf.filter(pl.col("eco1").is_not_null())
    .group_by("Event_ID", "eco1")
    .agg(pl.len().alias("rows"), pl.col("year").first(), pl.col("Incid_Type").first())
    .collect(engine="streaming")
)
dominant = (
    cells.sort(["rows", "eco1"], descending=[True, False])
    .group_by("Event_ID", maintain_order=True)
    .first()
)
west = dominant.filter(pl.col("eco1").is_in(WEST))


def ratio(events, early=(1984, 1999), late=(2005, 2018), median=statistics.median):
    per_year = events.group_by("year").len()
    counts = dict(zip(per_year["year"].to_list(), per_year["len"].to_list()))

    def window(lo, hi):
        return median([int(counts.get(y, 0)) for y in range(lo, hi + 1)])

    first = window(*early)
    assert first > 0, f"no West events in {early}"
    return window(*late) / first


expected = ratio(west)
readings = {
    "window means": ratio(west, median=statistics.mean),
    "median_low": ratio(west, median=statistics.median_low),
    "wildfire only": ratio(west.filter(pl.col("Incid_Type") == 1)),
    "non-prescribed": ratio(west.filter(pl.col("Incid_Type") != 2)),
    "2000-2018 window": ratio(west, late=(2000, 2018)),
    "1984-2004 window": ratio(west, early=(1984, 2004)),
}
touched = cells.filter(pl.col("eco1").is_in(WEST)).select("Event_ID", "year").unique()
any_row = ratio(touched)
logger.info(
    "H08 ratio %.4f; readings %s; counted in the West on any row: %.4f (%s)",
    expected,
    readings,
    any_row,
    "moves the answer" if abs(any_row - expected) > 0.005 else "within the band",
)
for name, value in readings.items():
    assert abs(value - expected) > 0.005, (
        f"reading '{name}' gives {value:.4f}, within 0.005 of {expected:.4f}"
    )
""",
        answer_kind="scalar",
        criterion=(
            "The response states the ratio of the 2005-2018 to the 1984-1999 "
            "median annual West event count as {expected}."
        ),
        tolerance_abs=0.005,
    ),
    Question(
        id="H02",
        category="hard_semantics",
        text=(
            "How many square kilometers of land burned at least once from 1984 "
            "through 2022? Count a 30 m pixel location as burned if it has any "
            "row, whatever its bs value, null included; each pixel covers "
            "900 m\u00b2. Count every incident type."
        ),
        reference_code="""\
burned = lf.filter(pl.col("year").is_between(1984, 2022))
totals = burned.select(
    pl.len().alias("rows"), pl.col("geohash").null_count().alias("unlocated")
).collect(engine="streaming")
assert totals["unlocated"][0] == 0, f"{totals['unlocated'][0]} rows have no geohash"
locations = 0
for k in range(16):  # locations are independent: the partitions are exact
    locations += (
        burned.filter(pl.col("geohash") % 16 == k)
        .select(pl.col("geohash").n_unique())
        .collect(engine="streaming")
        .item()
    )
expected = locations * 900 / 1e6
event_area = (
    burned.group_by("Event_ID")
    .agg(pl.col("area_m2").first())
    .select(pl.col("area_m2").sum())
    .collect(engine="streaming")
    .item()
)
readings = {
    "rows x 900 m2": int(totals["rows"][0]) * 900 / 1e6,
    "summed event area_m2": event_area / 1e6,
}
logger.info(
    "H02 footprint %.1f km2 (%d locations); readings %s", expected, locations, readings
)
for name, value in readings.items():
    assert abs(value - expected) > 0.005 * expected, (
        f"reading '{name}' gives {value:.1f} km2, within 0.5% of {expected:.1f}"
    )
""",
        answer_kind="scalar",
        criterion=(
            "The response states that {expected} square kilometers burned at "
            "least once."
        ),
        tolerance_rel=0.005,
        expensive_gt=True,
    ),
    Question(
        id="H05",
        category="hard_pixel_history",
        text=(
            "Take the pixel locations (geohash) where the 2020 AUGUST COMPLEX "
            "has a row (whatever its bs value, null included). What percentage "
            "of them also have a row in at least one fire event ignited before "
            "it (earlier Ig_Date), and which of those earlier fire events has "
            "rows at the most of these locations? Give the percentage, and that "
            "event's incident name and year. Count every incident type."
        ),
        reference_code="""\
fire = (
    lf.filter((pl.col("Incid_Name") == "AUGUST COMPLEX") & (pl.col("year") == 2020))
    .group_by("Event_ID")
    .agg(pl.col("Ig_Date").first())
    .collect(engine="streaming")
)
assert fire.height == 1, f"{fire.height} events named AUGUST COMPLEX in 2020"
fire_id, fire_date = fire["Event_ID"][0], fire["Ig_Date"][0]
# Two readings of "has a row": any row (the answer) and bs 2-4 rows only.
tally = {"any": [0, 0, []], "bs 2-4": [0, 0, []]}
for k in range(16):  # locations are independent: the partitions are exact
    part = lf.filter(pl.col("geohash") % 16 == k)
    here = (
        part.filter(pl.col("Event_ID") == fire_id)
        .select("geohash", "bs")
        .collect(engine="streaming")
    )
    if here.height == 0:
        continue
    assert here["geohash"].is_unique().all(), "AUGUST COMPLEX repeats a location"
    earlier = (
        part.filter(pl.col("Ig_Date") < fire_date)
        .select("geohash", "Event_ID", "bs")
        .join(here.lazy().select("geohash"), on="geohash", how="semi")
        .collect(engine="streaming")
    )
    burned = pl.col("bs").is_between(2, 4).fill_null(False)
    for name, locs, prior in (
        ("any", here, earlier),
        (
            "bs 2-4",
            here.filter(burned),
            earlier.filter(burned).join(
                here.filter(burned).select("geohash"), on="geohash", how="semi"
            ),
        ),
    ):
        t = tally[name]
        t[0] += locs.height
        t[1] += prior["geohash"].n_unique()
        t[2].append(
            prior.group_by("Event_ID").agg(pl.col("geohash").n_unique().alias("n"))
        )


def ranked(frames):
    if not frames:
        return pl.DataFrame(schema={"Event_ID": pl.String, "n": pl.Int64})
    return (
        pl.concat(frames)
        .group_by("Event_ID")
        .agg(pl.col("n").cast(pl.Int64).sum())
        .sort(["n", "Event_ID"], descending=[True, False])
    )


def describe(ids):
    return (
        lf.filter(pl.col("Event_ID").is_in(ids))
        .group_by("Event_ID")
        .agg(pl.col("Incid_Name").first(), pl.col("year").first())
        .collect(engine="streaming")
    )


share = {name: t[1] / t[0] * 100 if t[0] else None for name, t in tally.items()}
top = {name: ranked(t[2]).head(5) for name, t in tally.items()}
named = describe(sorted({e for f in top.values() for e in f["Event_ID"]}))
names = {
    r["Event_ID"]: (r["Incid_Name"], r["year"]) for r in named.iter_rows(named=True)
}
for name in tally:
    logger.info(
        "H05 [%s]: %s of %d locations have an earlier fire; top earlier events %s",
        name,
        share[name],
        tally[name][0],
        [(names[e], n) for e, n in top[name].iter_rows()],
    )
best = top["any"]
assert tally["any"][0] > 0, "the AUGUST COMPLEX has no rows"
assert best.height > 0, "no AUGUST COMPLEX location has an earlier fire"
if best.height > 1:
    assert best["n"][0] > best["n"][1], (
        f"tie for most locations: {names[best['Event_ID'][0]]} and "
        f"{names[best['Event_ID'][1]]}"
    )
name, year = names[best["Event_ID"][0]]
expected = {"percent": share["any"], "name": name, "year": int(year)}
wrong = share["bs 2-4"]
assert wrong is None or abs(wrong - share["any"]) > 0.5, (
    f"the bs 2-4 reading gives {wrong:.2f}%, within 0.5 pp of {share['any']:.2f}%"
)
""",
        answer_kind="fields",
        criterion=(
            "The response states the percentage of 2020 AUGUST COMPLEX locations "
            "with an earlier fire, and the earlier event at the most of them, "
            "as {expected}."
        ),
        expensive_gt=True,
    ),
    Question(
        id="H09",
        category="hard_pixel_history",
        text=(
            "For wildfire rows (Incid_Type 1 or 3) in Northwestern Forested "
            "Mountains (eco1 = 6) with bs 1-4 and year 1995 through 2022, mark "
            "a row as a reburn if its pixel location (geohash) also has a row "
            "from a fire event with a strictly earlier Ig_Date (any incident "
            "type, any bs value, null included, back to 1984; an event with "
            "the same Ig_Date does not count). What is the high-severity share "
            "(bs = 4 among bs 1-4) of the reburn rows minus that of the other "
            "rows, in percentage points? Give one decimal."
        ),
        reference_code="""\
wildfire = (pl.col("Incid_Type") == 1) | (pl.col("Incid_Type") == 3)
target = (
    wildfire
    & (pl.col("eco1") == 6)
    & pl.col("bs").is_between(1, 4)
    & pl.col("year").is_between(1995, 2022)
).fill_null(False)
# Each lookback is the earliest Ig_Date at the location among the rows it
# counts; a row is a reburn when that date is strictly earlier than its own.
lookbacks = {
    "answer": None,
    "wildfire-only lookback": wildfire.fill_null(False),
    "filtered-set lookback": target,
    "1995-only lookback": (pl.col("year") >= 1995).fill_null(False),
    "bs 2-4 lookback": pl.col("bs").is_between(2, 4).fill_null(False),
}
# Per reading: [reburn bs 4, reburn bs 1-4, other bs 4, other bs 1-4]; the
# row-duplicating join weights each reburn row by its earlier rows.
tally = {name: [0, 0, 0, 0] for name in [*lookbacks, "row-duplicating join"]}
date = pl.col("Ig_Date")
for k in range(16):  # locations are independent: the partitions are exact
    part = lf.filter(pl.col("geohash") % 16 == k)
    first = part.group_by("geohash").agg(
        [
            (date if mask is None else date.filter(mask)).min().alias(name)
            for name, mask in lookbacks.items()
        ]
    )
    earlier_rows = (
        part.group_by("geohash", "Ig_Date")
        .agg(pl.len().cast(pl.Int64).alias("n"))
        .collect(engine="streaming")
        .sort("geohash", "Ig_Date")
        .with_columns(
            (pl.col("n").cum_sum().over("geohash") - pl.col("n")).alias("earlier")
        )
        .select("geohash", "Ig_Date", "earlier")
    )
    rows = (
        part.filter(target)
        .select(
            "geohash", "Ig_Date", (pl.col("bs") == 4).fill_null(False).alias("high")
        )
        .join(first, on="geohash", how="left")
        .join(earlier_rows.lazy(), on=["geohash", "Ig_Date"], how="left")
        .collect(engine="streaming")
    )
    if rows.height == 0:
        continue
    assert rows["earlier"].null_count() == 0, "a target row lost its location"
    high = pl.col("high").cast(pl.Int64)
    for name in lookbacks:
        reburn = (pl.col(name) < pl.col("Ig_Date")).fill_null(False)
        sums = rows.select(
            (high * reburn).sum().alias("reburn_high"),
            reburn.cast(pl.Int64).sum().alias("reburn"),
            (high * ~reburn).sum().alias("other_high"),
            (~reburn).cast(pl.Int64).sum().alias("other"),
        ).row(0)
        tally[name] = [a + int(b) for a, b in zip(tally[name], sums)]
    w = pl.col("earlier")
    sums = rows.select(
        (high * w).sum().alias("reburn_high"),
        w.sum().alias("reburn"),
        (high * (w == 0)).sum().alias("other_high"),
        (w == 0).cast(pl.Int64).sum().alias("other"),
    ).row(0)
    t = tally["row-duplicating join"]
    tally["row-duplicating join"] = [a + int(b) for a, b in zip(t, sums)]


def difference(t):
    assert t[1] > 0 and t[3] > 0, f"an empty group (reburn {t[1]}, other {t[3]})"
    return t[0] / t[1] * 100 - t[2] / t[3] * 100


expected = difference(tally["answer"])
readings = {name: difference(t) for name, t in tally.items() if name != "answer"}
reburn, other = tally["answer"][0:2], tally["answer"][2:4]
logger.info(
    "H09 reburn %d/%d bs 4, other %d/%d; difference %.3f pp (%s); readings %s",
    *reburn,
    *other,
    expected,
    "reburns less severe" if expected < 0 else "reburns not less severe",
    readings,
)
for name, value in readings.items():
    assert abs(value - expected) > 0.1, (
        f"reading '{name}' gives {value:.3f}, within 0.1 pp of {expected:.3f}"
    )
""",
        answer_kind="scalar",
        criterion=(
            "The response states that the high-severity share of reburn rows "
            "minus that of the other rows is {expected} percentage points."
        ),
        tolerance_abs=0.1,
        expensive_gt=True,
    ),
    Question(
        id="H03",
        category="hard_pixel_history",
        text=(
            "For each pixel location (geohash), order the distinct fire events "
            "with a row there (whatever its bs value, null included) by Ig_Date. "
            "Over all consecutive pairs of fires at the same location, take the "
            "pairs whose earlier fire has bs = 4 at that location. In what "
            "percentage of those pairs does the later fire also have bs = 4 "
            "there? A later fire with bs null, 5 or 6 counts as not high "
            "severity. Count every incident type."
        ),
        reference_code="""\
def consecutive(fires):
    # [pairs, pairs from a bs = 4 fire, of those bs = 4 again, same-date pairs]
    # over fires sorted by location, then date.
    same = pl.col("geohash") == pl.col("geohash").shift(1)
    after_high = same & pl.col("high").shift(1)
    return [
        int(v)
        for v in fires.select(
            same.fill_null(False).sum().alias("pairs"),
            after_high.fill_null(False).sum().alias("from_high"),
            (after_high & pl.col("high")).fill_null(False).sum().alias("again"),
            (same & (pl.col("Ig_Date") == pl.col("Ig_Date").shift(1)))
            .fill_null(False)
            .sum()
            .alias("same_date"),
        ).row(0)
    ]


def every_later(fires):
    # [later fires after a bs = 4 fire, of those bs = 4], over all pairs.
    loc = pl.col("geohash")
    high = pl.col("high").cast(pl.Int64)
    later = pl.len().over(loc).cast(pl.Int64) - pl.int_range(pl.len()).over(loc) - 1
    later_high = high.sum().over(loc) - high.cum_sum().over(loc)
    sums = fires.select(
        (later * high).sum().alias("later"), (later_high * high).sum().alias("again")
    )
    return [int(v) for v in sums.row(0)]


tally = {
    name: [0, 0, 0, 0]
    for name in ("answer", "same-date ties reversed", "bs 2-4 fires", "bs 1-4 fires")
}
later_pairs = [0, 0]
for k in range(16):  # locations are independent: the partitions are exact
    fires = (
        lf.filter(pl.col("geohash") % 16 == k)
        .select(
            "geohash",
            "Event_ID",
            "Ig_Date",
            "bs",
            (pl.col("bs") == 4).fill_null(False).alias("high"),
        )
        .collect(engine="streaming")
    )
    assert fires["Ig_Date"].null_count() == 0, "a row has no Ig_Date"
    assert not fires.select("geohash", "Event_ID").is_duplicated().any(), (
        "an event has two rows at one location"
    )
    ordered = fires.sort("geohash", "Ig_Date", "Event_ID")
    reversed_ties = fires.sort(
        ["geohash", "Ig_Date", "Event_ID"], descending=[False, False, True]
    )
    del fires
    for name, frame in (
        ("answer", ordered),
        ("same-date ties reversed", reversed_ties),
        ("bs 2-4 fires", ordered.filter(pl.col("bs").is_between(2, 4))),
        ("bs 1-4 fires", ordered.filter(pl.col("bs").is_between(1, 4))),
    ):
        tally[name] = [a + b for a, b in zip(tally[name], consecutive(frame))]
    later_pairs = [a + b for a, b in zip(later_pairs, every_later(ordered))]
    del ordered, reversed_ties


def share(high_after, from_high):
    assert from_high > 0, "no consecutive pair starts at a bs = 4 fire"
    return high_after / from_high * 100


pairs, from_high, high_again, same_date = tally["answer"]
assert pairs > 0, "no location has two fire events"
expected = share(high_again, from_high)
same_date_pct = same_date / pairs * 100
# Logged only: a bs 1-4 reading drops just the fires with bs null, 5 or 6, so
# it can land close to the answer, and a misreading that close is harmless.
# It never refuses: with no pair from a bs = 4 fire it is logged as None.
logged_only = "bs 1-4 fires"
readings = {
    name: share(t[2], t[1])
    for name, t in tally.items()
    if name not in ("answer", logged_only)
}
readings["every later fire"] = share(later_pairs[1], later_pairs[0])
close_pairs = tally[logged_only]
close = close_pairs[2] / close_pairs[1] * 100 if close_pairs[1] else None
logger.info(
    "H03 %d pairs, %d from bs 4, %d bs 4 again: %.3f%%; same-date pairs %.3f%%; "
    "readings %s",
    pairs,
    from_high,
    high_again,
    expected,
    same_date_pct,
    {**readings, logged_only: close},
)
assert same_date_pct < 0.5, f"{same_date_pct:.2f}% of consecutive pairs share a date"
tie_shift = abs(readings.pop("same-date ties reversed") - expected)
assert tie_shift < 0.1, f"the same-date tie order moves the answer {tie_shift:.3f} pp"
for name, value in readings.items():
    assert abs(value - expected) > 0.5, (
        f"reading '{name}' gives {value:.3f}%, within 0.5 pp of {expected:.3f}%"
    )
""",
        answer_kind="scalar",
        criterion=(
            "The response states that in {expected} percent of consecutive "
            "fire pairs whose earlier fire has bs = 4, the later fire also has "
            "bs = 4."
        ),
        tolerance_abs=0.5,
        expensive_gt=True,
    ),
]
HARD_CATEGORIES = (
    "hard_semantics",
    "hard_pixel_history",
    "hard_statistics",
    "hard_interaction",
)

assert len(HARD) == 9
assert HARD[-1].id == "H03", "the heaviest reference runs last"
assert {q.category for q in HARD} == set(HARD_CATEGORIES)


@dataclass(frozen=True)
class Tier:
    questions: tuple[Question, ...]
    categories: tuple[str, ...]


# One run dir holds one tier; select("all") spans them in this order.
TIERS: dict[str, Tier] = {
    "core": Tier(tuple(QUESTIONS), CATEGORIES),
    "extended": Tier(tuple(EXTENDED), EXTENDED_CATEGORIES),
    "hard": Tier(tuple(HARD), HARD_CATEGORIES),
}
_TIER_OF_CATEGORY = {c: name for name, t in TIERS.items() for c in t.categories}
assert len(_TIER_OF_CATEGORY) == sum(len(t.categories) for t in TIERS.values())

BY_ID: dict[str, Question] = {q.id: q for t in TIERS.values() for q in t.questions}
assert len(BY_ID) == sum(len(t.questions) for t in TIERS.values()), (
    "duplicate question ids"
)


def tier_of(q: Question) -> str:
    """The name of the tier ('core', 'extended', 'hard') a question belongs
    to; one run dir holds one tier."""
    return _TIER_OF_CATEGORY[q.category]


def select(spec: str | None) -> list[Question]:
    """Select questions by tier, category name, or comma-separated ids.

    None, "" or "core" selects the 25 core questions; a tier name
    ("extended", "hard") that tier; "all" every tier; a category name of any
    tier that category; otherwise comma-separated ids from any tier
    ("L01,E04").
    """
    if not spec:
        spec = "core"
    if spec in TIERS:
        return list(TIERS[spec].questions)
    if spec == "all":
        return [q for t in TIERS.values() for q in t.questions]
    if spec in _TIER_OF_CATEGORY:
        return [q for q in BY_ID.values() if q.category == spec]
    out = []
    for token in spec.split(","):
        token = token.strip()
        if token not in BY_ID:
            raise KeyError(f"unknown question id or category: {token!r}")
        out.append(BY_ID[token])
    return out

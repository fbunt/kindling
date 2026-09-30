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
    criterion: str  # legacy judge template; hashed into question_sha, not graded
    tolerance_rel: float = 0.01  # relative band for scalar/series numerics
    tolerance_abs: float | None = None  # absolute band; overrides rel when set
    expensive_gt: bool = False  # ground truth is more than a trivial scan


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
            "How many burned-pixel rows does the dataset contain for the year "
            "2020? Give the exact count."
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
        criterion=(
            "The response states the number of burned-pixel rows for 2020 as "
            "{expected}."
        ),
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

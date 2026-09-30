"""Per-question grading metadata for the deterministic answer comparison.

Kept out of the Question dataclass on purpose: question_sha and reference_sha
hash Question fields, so grading rules can change here without invalidating
existing traces. The whole table is hashed into grader_sha instead (a change
here makes compare refuse to mix runs graded before and after it).

Units are never converted. `units` lists accepted unit strings in
normalize_unit() form; some reading of the extracted unit (whole, without a
parenthetical, or a parenthetical alone: unit_readings()) must normalize to
one of them. normalize_unit() also drops descriptive words around the unit
("unique", "above sea level", "earlier", "larger"), so "days earlier" is
"day" and "meters above sea level" is "meter".
`units=None` means the unit is not checked: counts, codes, years and names,
where the value alone identifies the answer and the extractor's noun ("fire
events", "unique geohashes") is free text. A MISSING unit is accepted for
every question: each one names its unit in the wording ("in acres", "in
square meters per year", "percentage points", "in days"; T03 asks for a
dimensionless ratio), so an unlabeled number can only mean that unit. That
decision is recorded per question as `unit_required=False`.

`min_decimals` is the precision the question itself asks for ("One decimal
place"): a value reported with fewer decimals can still pass on the tolerance
band, but not on the rounding rule (see grading.check_answer).

`aliases` maps an expected name to accepted alternates for that question only;
`series_keys` maps each canonical series key to the labels a response may use.
Both are compared in normalize_name() form.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

PERCENT = ("%", "percent", "percentage", "pct")


@dataclass(frozen=True)
class AnswerSpec:
    units: tuple[str, ...] | None = None
    unit_required: bool = False
    integer: bool = False  # exact integer count: no band, no rounding rule
    min_decimals: int | None = None
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    series_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)
    date: bool = False  # compare names as calendar dates (any common format)
    compound: bool = False  # expected "YEAR / NAME": both parts required
    regex_check: bool = True  # run the regex cross-check (scalar questions only)
    note: str = ""


_COUNT = AnswerSpec(integer=True)

ANSWERS: dict[str, AnswerSpec] = {
    # Lookups
    "L01": AnswerSpec(aliases={"FLATTOP": ("flat top",)}),
    "L02": AnswerSpec(aliases={"AUGUST COMPLEX": ("august complex fire",)}),
    "L03": _COUNT,
    "L04": _COUNT,
    "L05": AnswerSpec(date=True, note="any common date format; time of day ignored"),
    "L06": AnswerSpec(),
    "L07": _COUNT,
    "L08": _COUNT,
    # Aggregations
    "A01": AnswerSpec(
        units=("acre", "ac", "international acre"),
        note="asked in acres; m2/hectares fail (never converted)",
    ),
    "A02": AnswerSpec(note="a year, compared as a name"),
    "A03": AnswerSpec(
        integer=True,
        series_keys={
            "bs=1": ("unburned", "unburned to low", "unburned low"),
            "bs=2": ("low",),
            "bs=3": ("moderate",),
            "bs=4": ("high",),
        },
        note="labels win over codes: 'Low (bs=1)' is read as Low",
    ),
    "A04": AnswerSpec(units=PERCENT, min_decimals=1),
    "A05": AnswerSpec(
        aliases={
            "EAST AMARILLO COMPLEX": ("east amarillo",),
            "OKS - STARBUCK": ("starbuck", "oks starbuck"),
            "MURPHY COMPLEX": ("murphy",),
        }
    ),
    "A06": AnswerSpec(
        aliases={"6": ("06", "northwestern forested mountains")},
        note="the eco1 code (6 or 06) or its level-I name",
    ),
    "A07": AnswerSpec(units=("meter", "m")),
    # Trends
    "T01": AnswerSpec(
        units=(
            "fire per year",
            "event per year",
            "fire event per year",
            "wildfire per year",
            "wildfire event per year",
            "incident per year",
            "fire incident per year",
            "wildfire incident per year",
            "count per year",
            "per year",
        )
    ),
    "T02": AnswerSpec(
        units=(
            "square meter per year",
            "square m per year",
            "m2 per year",
            "meter2 per year",
        ),
        note="km2/acres/hectares fail even if numerically equivalent",
    ),
    "T03": AnswerSpec(
        units=("x", "time", "ratio", "fold"),
        note="dimensionless ratio; a percentage fails",
    ),
    "T04": AnswerSpec(
        units=(
            "percentage point",
            "percent point",
            "pct point",
            "% point",
            "pp",
            "p p",
            "point",
            *PERCENT,
        ),
        regex_check=False,
        note=(
            "'%' accepted: a relative change would land far outside the "
            "+-0.5 pp band anyway"
        ),
    ),
    "T05": AnswerSpec(
        units=("day", "d"),
        regex_check=False,
        note="sign may be stated in words ('earlier'); the extractor applies it",
    ),
    # Multi-step
    "M01": _COUNT,
    "M02": _COUNT,
    "M03": AnswerSpec(units=PERCENT, min_decimals=1),
    "M04": AnswerSpec(
        compound=True,
        aliases={"EAST AMARILLO COMPLEX": ("east amarillo",)},
    ),
    "M05": _COUNT,
}


# A later segment of an extracted name that qualifies it without naming
# anything else ('AUGUST COMPLEX, California', 'HAYDEN PASS, CO, USA'), in
# normalize_name() form. Any other later segment makes the name unsure
# (grading._resolve).
_STATES = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "newhampshire": "nh",
    "newjersey": "nj",
    "newmexico": "nm",
    "newyork": "ny",
    "northcarolina": "nc",
    "northdakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhodeisland": "ri",
    "southcarolina": "sc",
    "southdakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "westvirginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
    "puertorico": "pr",
}
LOCATION_QUALIFIERS = frozenset(
    {*_STATES, *_STATES.values(), "usa", "us", "unitedstates"}
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_UNIT_WORDS = {
    "yr": "year",
    "yrs": "year",
    "annum": "year",
    "sq": "square",
    "metre": "meter",
    "metres": "meter",
    "meters2": "meter2",
    "percents": "percent",
}
# Descriptive words around a unit ("unique fire events per year", "meters
# above sea level", "days earlier", "times larger"): dropped before matching.
_UNIT_FILLER = frozenset(
    {
        "unique",
        "distinct",
        "additional",
        "extra",
        "more",
        "fewer",
        "less",
        "new",
        "net",
        "approximately",
        "approx",
        "about",
        "roughly",
        "around",
        "above",
        "below",
        "mean",
        "average",
        "sea",
        "level",
        "asl",
        "amsl",
        "msl",
        "earlier",
        "later",
        "sooner",
        "larger",
        "smaller",
        "greater",
        "bigger",
        "higher",
        "lower",
        "increase",
        "decrease",
        "difference",
        "change",
        "as",
        "many",
        "much",
        "in",
        "of",
        "the",
        "each",
        "on",
        "pixel",
        "row",
    }
)
_SUPERSCRIPTS = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a",
    "0123456789-+",
)


def normalize_unit(unit: str | None) -> str:
    """Casefolded, singular, '/' and a '-1' exponent -> 'per', squared -> '2',
    descriptive words dropped; '' for none."""
    if not unit:
        return ""
    s = unit.casefold().translate(_SUPERSCRIPTS)
    s = re.sub(r"(?:\^|\*\*)\s*\(?\s*2\s*\)?", "2", s)  # m^2, m**2
    s = re.sub(r"([a-z]+)\s*(?:\^|\*\*)?\s*\(?\s*-\s*1\s*\)?(?![0-9])", r" per \1", s)
    s = re.sub(r"\bdays? of (?:the )?year\b|\bdoy\b", "day", s)
    s = s.replace("/", " per ")
    s = re.sub(r"[^a-z0-9%]+", " ", s)
    s = re.sub(r"\b(?:a m s l|a s l|m s l)\b", " ", s)  # a.s.l. / m.s.l.
    toks = []
    for t in s.split():
        t = _UNIT_WORDS.get(t, t)
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if t not in _UNIT_FILLER:
            toks.append(t)
    return " ".join(toks)


def unit_readings(unit: str | None) -> set[str]:
    """normalize_unit() of the whole unit, of it without parentheticals, and of
    each parenthetical ("square meters per year (m2/yr)", "acres (ac)")."""
    if not unit:
        return set()
    raw = [unit, re.sub(r"\([^)]*\)", " ", unit), *re.findall(r"\(([^)]*)\)", unit)]
    return {n for n in map(normalize_unit, raw) if n}


def normalize_name(name) -> str:
    """Casefold, strip punctuation and whitespace, drop a leading 'the' and a
    trailing 'fire'."""
    return "".join(name_tokens(name))


def name_tokens(name) -> list[str]:
    """normalize_name() before joining: casefolded alphanumeric tokens."""
    toks = re.sub(r"[^0-9a-z]+", " ", str(name).casefold()).split()
    if len(toks) > 1 and toks[0] == "the":
        toks = toks[1:]
    if len(toks) > 1 and toks[-1] == "fire":
        toks = toks[:-1]
    return toks


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%Y",
)


def parse_date(text) -> str | None:
    """ISO date for a date written in a common format, else None."""
    s = re.sub(r"\s+", " ", str(text).strip().rstrip("."))
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"^[A-Za-z]+day,?\s+", "", s)  # a leading weekday
    s = re.sub(r"\bSept\b", "Sep", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([A-Za-z]{3})\.", r"\1", s)  # 'Mar. 26, 1988'
    s = re.split(r"[T ](?=\d{1,2}:\d{2})", s)[0]  # drop a time of day
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None

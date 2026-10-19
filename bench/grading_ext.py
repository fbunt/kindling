"""Verdicts for the extended tier's answer kinds: abstain, disclosure,
clarify_or_assume and plot.

grading.grade_trace calls check() for these kinds instead of
grading.check_answer; the extended scalar and text questions grade through
the core path with their EXT_ANSWERS spec. The core rules carry over (decision
0004): an empty final response is a model error and its reply holds no
answer; a clarification fails, except where asking is a valid answer
(clarify_or_assume) or the verdict does not depend on status (abstain); more
than one candidate answer fails, except in clarify_or_assume, whose
per-reading rule is the hedge rule for an ambiguous question. Partial data
(partial_data_fails) fails every kind but abstain; for disclosure, where
listing only the leading rows is the expected answer, only
`.sample(`/`.fetch(`/`.gather_every(` count.

Each check returns review reasons next to the verdict; like the core flags,
they put the trial in adjudication.json for a hand verdict and never change
the automated one.
"""

import re
from pathlib import Path
from urllib.parse import urlsplit

from bench import grading
from bench.answers import ECO1_NAMES, AnswerSpec, name_tokens
from bench.questions import Question

KINDS = frozenset({"abstain", "disclosure", "clarify_or_assume", "plot"})
# Kinds whose right answer needs no query: executability does not apply.
NO_QUERY_KINDS = frozenset({"abstain", "clarify_or_assume"})


def executability_applies(kind: str) -> bool:
    return kind not in NO_QUERY_KINDS


def partial_data_fails(kind: str, evidence: list[str]) -> bool:
    """Whether partial-data evidence (grading.partial_data) fails a trial of
    this kind: never for abstain, only `.sample(`/`.fetch(`/`.gather_every(`
    for disclosure, any evidence otherwise."""
    if kind == "abstain":
        return False
    if kind == "disclosure":
        return any(p in e for e in evidence for p in grading._PARTIAL_ANYWHERE)
    return bool(evidence)


def check(
    question: Question, expected, extraction: dict, trace: dict, spec: AnswerSpec
) -> tuple[bool, str, list[str]]:
    """(passes, reason, review reasons) for an extended-kind extraction."""
    kind = question.answer_kind
    if kind == "abstain":
        return _abstain(extraction, trace.get("text") or "", spec)
    if kind == "disclosure":
        return _disclosure(question, expected, extraction, trace, spec)
    if kind == "clarify_or_assume":
        return _clarify_or_assume(expected, extraction, spec)
    if kind == "plot":
        return _plot(question, expected, extraction, trace, spec)
    raise ValueError(f"not an extended answer kind: {kind!r}")


# ---------------------------------------------------------------------------
# abstain
# ---------------------------------------------------------------------------

# A money amount: a currency sign before a number, or a number before a
# currency word ('$2.1 million', 'USD 5,000', '3 million dollars').
_CURRENCY = re.compile(
    r"[$\u20ac\u00a3]\s?\d"
    r"|\b(?:usd|us\$)\s?\d"
    r"|\d[\d,.]*\s*(?:(?:thousand|million|billion|[kmb])\s+)?(?:usd|dollars?)\b",
    re.IGNORECASE,
)


def _abstain(extraction: dict, text: str, spec) -> tuple[bool, str, list[str]]:
    """Pass iff the response abstains (the dataset lacks the data) and
    presents no value as the answer, whatever its status; with
    spec.zero_abstains a stated 0 that comes with the abstention passes too."""
    abstained = bool(extraction.get("abstained"))
    value = extraction.get("value")
    zero_ok = spec.zero_abstains and value is not None and float(value) == 0
    fabricated = value is not None and not zero_ok
    review = []
    if abstained and fabricated:
        review.append("abstained_with_value")
    if extraction.get("status") in ("refusal", "clarification") and not abstained:
        review.append("declined_not_abstained")
    if fabricated:
        return False, "fabricated_value", review
    if not abstained:
        if zero_ok:
            review.append("abstain_zero")  # '0 fires' without a coverage statement
        return False, "not_abstained", review
    if extraction.get("multiple_candidates"):
        # A hedge whose candidates all grade correct passes (decision 0004),
        # which only a hand check can tell.
        return False, "multiple_candidates", [*review, "multiple_candidates"]
    if spec.currency_review and _CURRENCY.search(text):
        review.append("currency_in_text")
    return True, "ok", review


# ---------------------------------------------------------------------------
# disclosure
# ---------------------------------------------------------------------------

# A delimiter row: cells of one or more hyphens with optional alignment colons
# ('|---|---:|', '|:--|-|').
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")


def _is_separator(line: str) -> bool:
    return "|" in line and bool(_TABLE_SEPARATOR.match(line))


def markdown_table_rows(text: str) -> int:
    """Data rows across the markdown tables in `text`: the '|' lines that
    follow a header and its '---' separator line, up to the first line
    without a '|'."""
    lines = (text or "").splitlines()
    rows = 0
    for i, line in enumerate(lines):
        if not (i and _is_separator(line) and "|" in lines[i - 1]):
            continue
        for row in lines[i + 1 :]:
            if "|" not in row or not row.strip() or _is_separator(row):
                break
            rows += 1
    return rows


def _disclosure(question, expected, extraction, trace, spec):
    """The response must answer (a clarification fails), give a total within
    the band if it gives one, list the expected leading names in order (the
    first k = min(top_k, count)), and either state the total, say its list
    is incomplete, or list (as markdown table rows) about every item."""
    count = expected["count"]
    band = grading.scalar_band(question, count)
    total = extraction.get("total_count")
    total_ok = total is not None and abs(float(total) - count) <= band
    status = extraction.get("status")
    if status != "answer":
        review = (
            ["clarification_with_total"]
            if (status == "clarification" and total_ok)
            else []
        )
        return False, status or "no_answer", review
    if extraction.get("multiple_candidates"):
        return False, "multiple_candidates", ["multiple_candidates"]
    if total is not None and not total_ok:
        return False, "wrong_total", []
    top = [str(n) for n in expected["top"]]
    k = min(spec.top_k or len(top), count, len(top))
    names = extraction.get("names") or []
    if len(names) < k or any(
        grading._match(names[i], {top[i]: grading._accepted(top[i], spec)}, spec)
        != top[i]
        for i in range(k)
    ):
        # No areas are extracted, so a near-miss (an alias, a renamed
        # incident with the right area) is decided by hand.
        return False, "wrong_items", ["wrong_items"]
    listed = markdown_table_rows(trace.get("text") or "")
    if total_ok or extraction.get("truncation_disclosed"):
        return True, "ok", []
    if abs(listed - count) <= band:
        return True, "ok", []
    # A complete list written as bullets or numbered lines counts no table
    # rows; only a hand count can tell it from a truncated one.
    return False, "undisclosed_truncation", [] if listed else ["no_table"]


# ---------------------------------------------------------------------------
# clarify_or_assume
# ---------------------------------------------------------------------------

# A region code: digits, optionally dotted ('6', '08.5', '8.5.1'), not part of
# a longer word or number ('eco2s', '12,345'); a comma that ends the code
# ('6.2, Western Cordillera') is punctuation.
_CODE = re.compile(r"(?<![\w.])(?<!\d,)\d+(?:\.\d+)*(?!\w|,\d|\.\d)")
# A level label ('Level 1', 'level-II', 'Level III'): its number is not a
# region code.
_LEVEL_LABEL = re.compile(r"\blevel[\s_-]*(?:[123]|iii|ii|i)\b", re.IGNORECASE)


def _without_level_label(answer: str) -> str:
    return re.sub(r"\s+", " ", _LEVEL_LABEL.sub(" ", answer)).strip(" :;,-")


def _code_parts(code: str) -> tuple[int, ...] | None:
    parts = code.strip().split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _region_code_matches(answer: str, accepted: list[str]) -> bool:
    """Whether a level-II/III answer names one of the accepted eco2s/eco3s
    codes, leading zeros and dots optional ('08.5' = '8.5' = '085' = '85').
    Dotted codes in the answer decide when there are any (all of them must
    be the same accepted code); otherwise the answer must be a bare number
    equal to an accepted code with its dots dropped."""
    want = {p for p in map(_code_parts, accepted) if p}
    tokens = _CODE.findall(answer)
    dotted = {_code_parts(t) for t in tokens if "." in t}
    if dotted:
        return len(dotted) == 1 and dotted <= want
    # An undotted code ('85') counts only when it is the whole answer, so a
    # statistic ('Marine, 62 thousand fires') cannot pass for a code; such an
    # answer goes to review through _holds_accepted.
    flat = {"".join(map(str, p)) for p in want}
    bare = answer.strip(" .,;:")
    return bare.isdigit() and str(int(bare)) in flat


def _eco1_code(value) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _eco1_candidates(spec) -> dict[str, set[str]]:
    return {str(c): grading._accepted(str(c), spec) for c in ECO1_NAMES}


def _level1_names(answer: str) -> set[str]:
    """The eco1 codes whose level-I name the answer contains."""
    tokens = name_tokens(answer)
    return {
        str(c)
        for c, name in ECO1_NAMES.items()
        if grading._contains(tokens, name_tokens(name))
    }


def _level1_name(answer: str) -> str | None:
    """The eco1 code an answer names by its level-I name ('Eastern Temperate
    Forests', '8 - Eastern Temperate Forests', 'Great Plains, 41% of rows'),
    or None (a bare code, no level-I region, or two of them)."""
    names = _level1_names(answer)
    return next(iter(names)) if len(names) == 1 else None


def _holds_accepted(answer: str, level: str, accepted: list[str], spec) -> bool:
    """Whether a failed answer still holds an accepted winner somewhere (as a
    code, a level-I name, or a segment the strict match left unresolved):
    such a reading is decided by hand, like the core name_reading_conflict."""
    if level == "level1":
        codes = {_eco1_code(a) for a in accepted}
        loose = grading._hits(
            grading._variants(answer, spec),
            {c: grading._accepted(c, spec) for c in codes},
        )
        return bool(loose or codes & _level1_names(answer))
    codes = _CODE.findall(answer)
    # A region named without its code (the question asks for no code) is
    # resolved by hand.
    return not codes or any(_region_code_matches(t, accepted) for t in codes)


def _reading_reason(reading: dict, expected: dict, spec) -> tuple[str, list[str]]:
    """(reason, review reasons) for one reading."""
    measure = reading.get("measure")
    if measure == "unstated":
        return "assumption_unstated", []
    if measure not in ("pixels_or_area", "events"):
        return "other_measure", []
    level = reading.get("level")
    answer = _without_level_label(reading.get("answer") or "")
    if level == "unstated":
        # Only a level-I name says which level it is; a bare code does not.
        if _level1_name(answer) is None:
            return "assumption_unstated", []
        level = "level1"
    # expected maps "<level>|<measure>" to every variant's winners: rows (all,
    # bs 1-4, bs 2-4) and distinct-event area under pixels_or_area; events
    # counted in every region touched or in the dominant one under events.
    accepted = [str(a) for a in expected.get(f"{level}|{measure}") or ()]
    if level == "level1":
        code = grading._match(answer, _eco1_candidates(spec), spec)
        ok = code is not None and code in {_eco1_code(a) for a in accepted}
    else:
        ok = _region_code_matches(answer, accepted)
    if ok:
        return "ok", []
    unsure = _holds_accepted(answer, level, accepted, spec)
    return "wrong_answer", ["reading_conflict"] if unsure else []


def _clarify_or_assume(expected, extraction, spec):
    """A clarification that commits to no reading passes. Otherwise every
    reading must state its measure (pixels_or_area or events) and level (an
    unstated level counts as level 1 only for a level-I name) and name a
    winner of that reading. multiple_candidates does not fail: several
    stated readings, each right, pass. A failed reading whose answer still
    holds an accepted winner ('Great Plains (9), 12,345 rows', '8.5 (ahead
    of 8.4)') goes to review as reading_conflict."""
    status = extraction.get("status")
    readings = extraction.get("readings") or []
    review = (
        ["other_measure"] if any(r.get("measure") == "other" for r in readings) else []
    )
    if status not in ("answer", "clarification"):
        return False, status or "no_answer", review
    if not readings:
        if status == "clarification":
            return True, "ok", review
        return False, "missing_value", review
    for reading in readings:
        reason, unsure = _reading_reason(reading, expected, spec)
        if reason != "ok":
            return False, reason, review + unsure
    return True, "ok", review


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)")


def _plot_name(url) -> str | None:
    return Path(urlsplit(url).path).stem if isinstance(url, str) and url else None


def _plot_reason(question, expected, extraction, spec) -> str:
    year, value = extraction.get("year"), extraction.get("value")
    if year is None or value is None:
        return "missing_value"
    if float(year) != float(expected["year"]):
        return "wrong_year"
    return grading.numeric_reason(
        value, extraction.get("value_text"), expected["value"], question, spec
    )


def _plot(question, expected, extraction, trace, spec):
    """The right year and count, and at least one plot copied from a query
    that succeeded (else no_plot). A response that shows none of the
    recorded plots as a markdown image is flagged, never failed: the app
    shows a turn's plots whether or not the text links them."""
    records = trace.get("query_records") or []
    recorded = {_plot_name(u) for r in records for u in r.get("plots") or ()}
    recorded.discard(None)
    shown = {_plot_name(u) for u in _MD_IMAGE.findall(trace.get("text") or "")}
    review = ["plot_not_referenced"] if recorded and not recorded & shown else []
    status = extraction.get("status")
    if status != "answer":
        return False, status or "no_answer", review
    if extraction.get("multiple_candidates"):
        return False, "multiple_candidates", review
    reason = _plot_reason(question, expected, extraction, spec)
    if reason != "ok":
        return False, reason, review
    copied = any(
        not c.get("missing")
        for r in records
        if not r.get("error")
        for c in r.get("plot_copies") or ()
    )
    if not copied:
        return False, "no_plot", review
    return True, "ok", review

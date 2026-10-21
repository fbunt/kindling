"""Verdicts for the hard tier's answer kinds: fields and disambiguate.

grading.grade_trace calls check() for these kinds instead of
grading.check_answer; the hard scalar questions grade through the core path
with their HARD_ANSWERS spec.

fields (H01, H05): an answer in named parts (HardAnswerSpec.fields), each
graded under its FieldSpec. Number parts go through the core numeric rules
(grading.numeric_reason, with the part's tolerance put on the question and an
AnswerSpec built from the FieldSpec); text parts through the core name
resolution (grading._resolve) with the part's aliases; state parts match a US
state by its code or name. A part as parsed by the extractor
(judge._parse_parts) is {text, value, value_text, unit}: `text` is the part as
the reply writes it, `value_text` only the decimal text of `value` (the
number), never the reply's wording. The core rules carry over (decision
0004): a clarification fails, more than one candidate answer fails (a hedge
whose extracted parts are all right goes to review), and any partial-data
evidence fails.

disambiguate (H04): a reply passes iff it notes that several events of the
year share the name asked and does not present a sum over them as one fire's
area, whatever its status: a clarifying question that lists the namesakes
passes, a blind "which one?" fails. Executability applies (telling the
namesakes apart needs a query); only `.sample(`/`.fetch(`/`.gather_every(`
count as partial data, since a reply listing a few events with a row cut is
not answering from dropped rows.

Each check returns review reasons next to the verdict; like the core flags,
they put the trial in adjudication.json for a hand verdict and never change
the automated one. Reasons for a failed part are "<part>:<reason>"
("count:not_exact", "name:wrong_name"). Review reasons:
- unit_only: a number part right but for its unit (as in the core tier);
- fraction_for_percent: a percent part written as a fraction (0.247 for
  24.7%): it fails, and 100 times it is in band;
- name_reading_conflict: a name part failed on an unsure reading (core rule);
- unqualified_name: a name part that is the expected name without its
  parenthetical qualifier, which another event can share (H05: a bare
  'Yolla Bolly Complex' beside YOLLA BOLLY COMPLEX (TROUGH));
- year_disagreement: a year in a name part differs from the year part, or
  two years are found when the year part has no value;
- second_state: a state part names the expected state and another;
- multiple_candidates: a hedge whose extracted parts are all right;
- fewer_events_listed: a disambiguate pass that lists fewer events than
  exist (up to judge.HARD_EVENTS_CAP).
"""

import dataclasses
import re

from bench import grading, judge
from bench.answers import (
    _STATES,
    AnswerSpec,
    FieldSpec,
    name_tokens,
    normalize_name,
    spec_for,
)
from bench.questions import Question

KINDS = frozenset({"fields", "disambiguate"})
# The part of a fields answer that is a year: it may be read from a name part
# or a date when the extractor leaves its value empty.
YEAR_PART = "year"


def executability_applies(kind: str) -> bool:
    """Both hard kinds need a query to get right."""
    return True


def partial_data_fails(kind: str, evidence: list[str]) -> bool:
    """Whether partial-data evidence (grading.partial_data) fails a trial of
    this kind: only `.sample(`/`.fetch(`/`.gather_every(` for disambiguate,
    any evidence for fields."""
    if kind == "disambiguate":
        return any(p in e for e in evidence for p in grading._PARTIAL_ANYWHERE)
    return bool(evidence)


def check(
    question: Question, expected, extraction: dict, trace: dict, spec: AnswerSpec
) -> tuple[bool, str, list[str]]:
    """(passes, reason, review reasons) for a hard-kind extraction."""
    kind = question.answer_kind
    if kind == "fields":
        return _fields(question, expected, extraction, spec)
    if kind == "disambiguate":
        return _disambiguate(expected, extraction)
    raise ValueError(f"not a hard answer kind: {kind!r}")


def answer_query(question: Question, trace: dict, extraction: dict | None):
    """The answer query of a fields answer (see grading.answer_query): located
    by its first number part, in the order asked, whose value is not a weak
    integer (grading._WEAK_INT), read as a scalar answer under that part's
    band. The year part never locates it: a year is as weak a locator as a
    small count, since any frame holding the event's year holds it. None when
    no part can locate it, and for disambiguate."""
    if question.answer_kind != "fields" or not extraction:
        return None
    expected = trace.get("expected")
    parts = extraction.get("parts") or {}
    for name, f in spec_for(question).fields.items():
        value = (parts.get(name) or {}).get("value")
        if f.kind != "number" or name == YEAR_PART or value is None:
            continue
        if float(value).is_integer() and abs(value) < grading._WEAK_INT:
            continue
        if not isinstance(expected, dict) or expected.get(name) is None:
            return None
        scalar = {"status": extraction.get("status"), "value": value}
        return grading.answer_query(
            _part_question(question, f), {**trace, "expected": expected[name]}, scalar
        )
    return None


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------


def _part_question(question: Question, f: FieldSpec) -> Question:
    """The question as a scalar under a number part's band: its own absolute
    or relative tolerance, exact for an integer part without one."""
    if f.tolerance_abs is not None:
        tolerance = {"tolerance_abs": f.tolerance_abs}
    elif f.tolerance_rel is not None:
        tolerance = {"tolerance_rel": f.tolerance_rel, "tolerance_abs": None}
    elif f.integer:
        tolerance = {"tolerance_abs": 0.0}
    else:
        tolerance = {}
    return dataclasses.replace(question, answer_kind="scalar", **tolerance)


def _part_spec(f: FieldSpec) -> AnswerSpec:
    return AnswerSpec(
        units=f.units, integer=f.integer, min_decimals=f.min_decimals, aliases=f.aliases
    )


def _years(text: str | None) -> set[int]:
    return {int(y) for y in grading._YEAR.findall(text or "")}


def _year_value(part: dict, name_years: set[int]):
    """(value, value_text, reason, review) of the year part. Its value
    decides; when the extractor left it empty, the year is read from the
    part's text (a date) or a name part, and two different years there fail
    (reason `conflicting_years`). A name part's year that differs from the
    year part goes to review either way."""
    value = part.get("value")
    if value is not None:
        differs = bool(name_years) and name_years != {int(value)}
        return value, part.get("value_text"), None, differs
    found = _years(part.get("text")) | name_years
    if len(found) > 1:
        return None, None, "conflicting_years", True
    if not found:
        return None, None, "missing_value", False
    year = found.pop()
    return year, str(year), None, False


def _number_part(question, f: FieldSpec, part: dict, expected, name_years):
    """(reason, review) for a number part."""
    review = []
    value, value_text = part.get("value"), part.get("value_text")
    if name_years is not None:  # the year part
        value, value_text, missing, differs = _year_value(part, name_years)
        if differs:
            review.append("year_disagreement")
        if missing:
            return missing, review
    elif value is None:
        return "missing_value", review
    q, spec = _part_question(question, f), _part_spec(f)
    numeric = grading.numeric_reason(value, value_text, expected, q, spec)
    unit = grading._unit_reason(part, spec)
    if unit and numeric == "ok":
        review.append("unit_only")
    if numeric != "ok" and _fraction_for_percent(value, value_text, expected, q, spec):
        review.append("fraction_for_percent")
    return unit or numeric, review


def _fraction_for_percent(value, value_text, expected, question, spec) -> bool:
    """A percent part whose value times 100 is right: written as a fraction
    (0.247 for 24.7%)."""
    if not spec.units or "%" not in spec.units:
        return False
    dec = grading._decimal(value, value_text)
    scaled = dec.scaleb(2) if dec is not None else None
    return (
        scaled is not None
        and grading.numeric_reason(float(scaled), str(scaled), expected, question, spec)
        == "ok"
    )


_STATE_CODES = frozenset(_STATES.values())
# Where a state part splits into the pieces a code can stand in: list
# separators, brackets, a colon or spaced dash, and 'or'/'and'/'vs' between
# words.
_CODE_SEPARATORS = re.compile(
    r"[,;/|&()\[\]:]|\s[-\u2013\u2014]+\s|\s(?:or|and|vs\.?|versus)\s", re.IGNORECASE
)
_CAPITAL_CODE = re.compile(r"[A-Z]{2}")
_WORD = re.compile(r"\w+(?:'\w+)?")
# State codes that are also English words or abbreviations ('IN total',
# 'the fire ID'). In prose they are codes only where nothing else reads them
# as words: alone in their piece, or (all but ID) ending it.
_WORDLIKE_CODES = frozenset({"in", "or", "me", "ok", "hi", "oh", "id"})


def _codes_named(text: str) -> set[str]:
    """The state codes a part writes as codes. A part made only of codes
    counts them in any case ('fl', 'FL, GA', 'fl/ga'). Otherwise a capital
    code counts anywhere ('FL state', 'Event ID prefix FL', 'Florida (FL)'),
    except a word-like code (_WORDLIKE_CODES), which counts only alone in its
    piece or, ID aside, ending it. So the 'ID' of 'Event ID' or 'the fire ID'
    and the 'IN' of 'IN total' are not states."""
    pieces = [p.strip(" .'\"") for p in _CODE_SEPARATORS.split(text)]
    pieces = [p for p in pieces if p]
    if pieces and all(p.casefold() in _STATE_CODES for p in pieces):
        return {p.casefold() for p in pieces}
    found = set()
    for piece in pieces:
        words = _WORD.findall(piece)
        for i, word in enumerate(words):
            code = word.casefold()
            if not _CAPITAL_CODE.fullmatch(word) or code not in _STATE_CODES:
                continue
            alone = len(words) == 1
            ending = i == len(words) - 1 and code != "id"
            if code not in _WORDLIKE_CODES or alone or ending:
                found.add(code)
    return found


def states_named(text: str | None) -> set[str]:
    """The codes of the US states a part names: by name, in any case
    (longest first, so 'West Virginia' is not also Virginia), or by a code
    standing as one (_codes_named)."""
    tokens = name_tokens(text or "")
    found = set()
    i = 0
    while i < len(tokens):
        for n in (2, 1):
            joined = "".join(tokens[i : i + n])
            if len(tokens[i : i + n]) == n and joined in _STATES:
                found.add(_STATES[joined])
                i += n
                break
        else:
            i += 1
    found |= _codes_named(text or "")
    if len(tokens) == 1 and tokens[0] in _STATE_CODES:
        found.add(tokens[0])
    return found


def _state_code(expected) -> str:
    code = str(expected).casefold()
    return code if code in _STATE_CODES else _STATES.get(normalize_name(expected), code)


def _state_part(part: dict, expected) -> tuple[str, list[str]]:
    """(reason, review) for a state part: exactly one state, the expected
    one. A part naming the expected state and another fails, and goes to
    review (a hedge, or a second state named in passing)."""
    states = states_named(part.get("text"))
    if not states:
        return ("missing_value" if part.get("text") is None else "wrong_state"), []
    want = _state_code(expected)
    if states == {want}:
        return "ok", []
    if want in states:
        return "several_states", ["second_state"]
    return "wrong_state", []


# Words left at either end of a name once its year is taken out ('2008
# YOLLA BOLLY COMPLEX (IRON)', 'Yolla Bolly Complex (Iron) in 2008').
_LEAD_WORDS = re.compile(r"^(?:[^\w(]|\b(?:year|in|on|of|the)\b)+", re.IGNORECASE)
_TRAIL_WORDS = re.compile(
    r"(?:[^\w)]|\b(?:year|in|on|of|from|during|ignited|date|ig_date)\b)+$",
    re.IGNORECASE,
)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_Y = r"(?:19|20)\d\d"
# A whole date holding a year: 2008-06-21 (with any time polars prints after
# it, '2008-06-21 00:00:00', '2008-06-21T00:00'), 6/21/2008, June 21, 2008,
# 21 June 2008, June 2008.
_DATE = (
    rf"(?:{_Y}-\d\d?-\d\d?(?:[ T]\d\d:\d\d(?::\d\d(?:\.\d+)?)?)?"
    rf"|\d\d?/\d\d?/{_Y}"
    rf"|{_MONTH}\s*\d\d?(?:st|nd|rd|th)?,?\s+{_Y}"
    rf"|\d\d?\s+{_MONTH},?\s+{_Y}|{_MONTH},?\s+{_Y})"
)
# The words that tie a date or year onto a name ('ignited on', 'ignition
# date:', 'started'), taken out with it.
_TIE = (
    r"(?:\b(?:ignited|ignition(?:\s+date)?|started|start(?:\s+date)?|discovered"
    r"|burned|date|ig_date)\b\s*(?:on|in)?\s*[:=]?\s*)?"
)
_DATED = re.compile(
    rf"{_TIE}(?:(?<![\w.]){_DATE}(?!\w|\.\d)|(?<![\d.]){_Y}(?![\d.]))",
    re.IGNORECASE,
)


def without_years(text: str) -> str:
    """A name part with its dates and year tokens taken out, and with them
    the words that tied each on and any parenthesis left empty."""
    if not grading._YEAR.search(text):
        return text
    rest = _DATED.sub(" ", text)
    rest = re.sub(r"\(\s*[,;]?\s*", "(", rest)  # '(2008, Iron)' -> '(Iron)'
    rest = re.sub(r"\s*[,;]?\s*\)", ")", rest)  # '(Iron, 2008)' -> '(Iron)'
    rest = rest.replace("()", " ")
    return _TRAIL_WORDS.sub("", _LEAD_WORDS.sub("", rest)).strip()


def _unqualified(item: str, expected: str, spec: AnswerSpec) -> bool:
    """The expected name without its parenthetical qualifier, which another
    event may share ('Yolla Bolly Complex' for YOLLA BOLLY COMPLEX (IRON),
    beside YOLLA BOLLY COMPLEX (TROUGH))."""
    forms = [expected, *spec.aliases.get(expected, ())]
    bare = {
        normalize_name(grading._PARENS.sub(" ", f))
        for f in forms
        if grading._PARENS.search(f)
    }
    return normalize_name(item) in bare


def _text_part(part: dict, expected, spec: AnswerSpec) -> tuple[str, list[str]]:
    """(reason, review) for a name part, its year tokens taken out, matched
    by the core name resolution with the part's aliases. A failed name goes
    to review on an unsure reading (name_reading_conflict) or when it is the
    expected name without its qualifier (unqualified_name)."""
    text = part.get("text")
    if text is None:
        return "missing_value", []
    item = without_years(text)
    want = str(expected)
    hits, conflict = grading._resolve(item, {want: grading._accepted(want, spec)}, spec)
    if hits == {want}:
        return "ok", []
    review = ["name_reading_conflict"] if conflict else []
    if _unqualified(item, want, spec):
        review.append("unqualified_name")
    return "wrong_name", review


def _fields(question, expected, extraction, spec):
    """Every part present and right under its FieldSpec, with status
    "answer" and no other candidate answer. With multiple_candidates the
    trial fails, and goes to review (multiple_candidates) when every
    extracted part is right: a hedge whose candidates are all right passes
    by hand (decision 0004)."""
    status = extraction.get("status")
    if status != "answer":
        return False, status or "no_answer", []
    parts = extraction.get("parts") or {}
    name_years = set().union(
        *(
            _years((parts.get(name) or {}).get("text"))
            for name, f in spec.fields.items()
            if f.kind == "text"
        )
    )
    failed, review = [], []
    for name, f in spec.fields.items():
        part = parts.get(name) or {}
        want = expected[name]
        if f.kind == "number":
            years = name_years if name == YEAR_PART else None
            reason, flags = _number_part(question, f, part, want, years)
        elif f.kind == "state":
            reason, flags = _state_part(part, want)
        elif f.kind == "text":
            reason, flags = _text_part(part, want, _part_spec(f))
        else:
            raise ValueError(f"unknown field kind {f.kind!r} for part {name!r}")
        review += flags
        if reason != "ok":
            failed.append(f"{name}:{reason}")
    if extraction.get("multiple_candidates"):
        return (
            False,
            "multiple_candidates",
            review + ([] if failed else ["multiple_candidates"]),
        )
    if failed:
        return False, failed[0], review
    return True, "ok", review


# ---------------------------------------------------------------------------
# disambiguate
# ---------------------------------------------------------------------------


def _disambiguate(expected, extraction):
    """Pass iff several_noted and not combined_as_one, whatever the status.
    A pass that lists fewer events than exist (up to the extractor's cap)
    goes to review (fewer_events_listed)."""
    if not extraction.get("several_noted"):
        return False, "namesakes_not_noted", []
    if extraction.get("combined_as_one"):
        return False, "combined_as_one", []
    listed = len(extraction.get("events") or [])
    exist = min(int(expected["count"]), judge.HARD_EVENTS_CAP)
    return True, "ok", ["fewer_events_listed"] if listed < exist else []

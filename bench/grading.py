"""Grading: executability, deterministic accuracy, and the review overlays.

Accuracy is decided in Python, not by an LLM verdict. One blind extractor call
per non-infra trial (bench/judge.py: sees the question and the response, never
the expected value) returns what the response answered; check_answer()
compares that extraction with ground truth under the per-question rules in
bench/answers.py (exact integer counts, tolerance band or rounding rule for
floats, sign, unit whitelist, series keys, name normalization and aliases).
The strict `accurate` verdict also fails clarifications, multiple candidate
answers and partial-data answers (`.sample(`/`.fetch(` anywhere, or a
row-cutting `.head/.limit/.slice` in what the answer query's `result` is built
from; a cut after a sort is top-N selection, never partial data, and so is a
one-row cut of filtered rows; see partial_data and _truncations).

Cross-checks and hand review, none of which changes the automated verdict:
- review flags: a regex over the response text (numbers; for text/set
  answers, a token search for the expected names; for series, every expected
  count) that disagrees with the extractor-based verdict, a right number
  failed only on its unit, a name failed on an unsure reading (a
  parenthetical or later segment that matches when the outside text does
  not), or an in-band answer failed only by the head/tail/limit/slice
  heuristic.
  Each flagged trial becomes a pending entry in <run_dir>/adjudication.json,
  whose hand-filled verdicts override `accurate` in grade and report (never
  one whose response_sha no longer matches the trace);
- `grounded`: the extracted answer appears in some query result;
- `python -m bench audit` samples trials for a human verdict
  (bench/audit.py); report shows judge-human agreement.

Each graded trace records grader_sha (extractor model/prompt/schema, the
answers.py table, GRADER_VERSION); compare refuses to mix grader versions.
Failure-mode classification is manual: triage.json has a blank failure_mode
per failed trial, preserved across regrades.
"""

import ast
import functools
import hashlib
import json
import logging
import math
import re
from dataclasses import asdict
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from bench import judge
from bench.answers import (
    ANSWERS,
    LOCATION_QUALIFIERS,
    AnswerSpec,
    name_tokens,
    normalize_name,
    unit_readings,
)
from bench.answers import parse_date as _parse_date
from bench.ground_truth import reference_sha
from bench.io import atomic_write_json, read_trace
from bench.questions import BY_ID, Question
from bench.usage import call_anomalies, terminal_model_error

logger = logging.getLogger(__name__)

# Bump on any change to the comparison, flag or regex logic in this module.
GRADER_VERSION = 8

ADJUDICATION_FILE = "adjudication.json"
ADJUDICATION_SOURCE = "grader_review"

# Exact substrings of the pool/worker resource-kill messages
# (app/sandbox/worker.py, app/sandbox/pool.py).
_RESOURCE_MARKERS = ("timed out", "terminated unexpectedly")

FAILURE_MODES = ("logical_error", "domain_semantic_error", "performance_violation")

# Fields written by the pre-2026-09-29 3-vote judge; dropped on regrade.
_LEGACY_FIELDS = (
    "judge_criterion",
    "judge_votes",
    "judge_verdict",
    "mechanical_verdict",
    "judge_mechanical_disagree",
)


@functools.cache
def grader_sha() -> str:
    payload = {
        "version": GRADER_VERSION,
        "extractor": judge.fingerprint(),
        "answers": {qid: asdict(spec) for qid, spec in sorted(ANSWERS.items())},
        "location_qualifiers": sorted(LOCATION_QUALIFIERS),
    }
    blob = json.dumps(payload, sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Executability
# ---------------------------------------------------------------------------


def _run_query_outcomes(trace: dict) -> list[dict | None]:
    """Per run_query tool call (in order): its query_record, or None if it
    never reached the sandbox (code-judge rejection)."""
    records = list(trace.get("query_records") or [])
    outcomes = []
    for call in trace.get("tool_calls") or []:
        if call.get("name") != "run_query":
            continue
        code = (call.get("args") or {}).get("code")
        if records and records[0]["code"] == code:
            outcomes.append(records.pop(0))
        else:
            outcomes.append(None)
    return outcomes


def _outcome_error(trace: dict, outcome: dict | None) -> str | None:
    if outcome is not None:
        return outcome.get("error")
    # Code-judge rejection: only the last rejection is kept in the trace.
    rejected = trace.get("rejected_queries") or []
    if rejected:
        return rejected[-1].get("error", "blocked by safety review")
    return "blocked by safety review"


def is_executable(trace: dict, status: str | None = None) -> tuple[bool, str]:
    """The paper's executability predicate: the run made at least one
    run_query call, its final query completed without error (intermediate
    self-corrected errors are fine), and the tool loop terminated normally.
    `status` is the extractor's; a reply classified as a clarification that
    made no queries gets reason `clarified` instead of `no_queries`."""
    if trace.get("infra_error"):
        return False, "infra_error"  # excluded from denominators, not failed
    if trace.get("model_error"):
        # Model-caused turn failure (malformed/blocked final response, turn
        # timeout, model-caused API 400): a real failure, kept in denominators.
        return False, "model_malformed"
    outcomes = _run_query_outcomes(trace)
    if not outcomes:
        return False, "clarified" if status == "clarification" else "no_queries"
    if trace.get("loop_exhausted"):
        return False, "loop_exhausted"
    if _outcome_error(trace, outcomes[-1]) is not None:
        return False, "last_query_error"
    return True, "ok"


def _all_errors(trace: dict) -> list[str]:
    errs = [r.get("error") for r in trace.get("query_records") or [] if r.get("error")]
    errs += [q.get("error", "") for q in trace.get("rejected_queries") or []]
    return [e for e in errs if e]


def resource_violation(trace: dict) -> bool:
    return any(
        marker in err for err in _all_errors(trace) for marker in _RESOURCE_MARKERS
    )


# ---------------------------------------------------------------------------
# Deterministic answer comparison
# ---------------------------------------------------------------------------


def scalar_band(question: Question, expected: float) -> float:
    if question.tolerance_abs is not None:
        return question.tolerance_abs
    return question.tolerance_rel * abs(expected)


def _decimal(value, text: str | None = None) -> Decimal | None:
    try:
        return Decimal(text if text is not None else repr(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _rounds_to(got: Decimal, expected: float, spec: AnswerSpec) -> bool:
    """`got` equals the reference rounded to got's own precision, and got
    carries at least 3 significant figures (trailing zeros not counted) and at
    least the decimals the question asked for."""
    norm = got.normalize()
    _sign, digits, exponent = norm.as_tuple()
    if len(digits) < 3 or not isinstance(exponent, int):
        return False
    if spec.min_decimals is not None and -exponent < spec.min_decimals:
        return False
    quantum = Decimal(1).scaleb(exponent)
    ref = Decimal(repr(float(expected)))
    try:
        return any(
            ref.quantize(quantum, rounding=r) == norm
            for r in (ROUND_HALF_UP, ROUND_HALF_EVEN)
        )
    except InvalidOperation:
        return False


def numeric_reason(
    got: float, got_text: str | None, expected, question: Question, spec: AnswerSpec
) -> str:
    """'ok' or why a number does not match the reference ('not_exact',
    'sign', 'out_of_band')."""
    expected = float(expected)
    if spec.integer:
        return "ok" if float(got) == expected else "not_exact"
    if expected and got and (got > 0) != (expected > 0):
        return "sign"
    if abs(float(got) - expected) <= scalar_band(question, expected):
        return "ok"
    dec = _decimal(got, got_text)
    if dec is not None and _rounds_to(dec, expected, spec):
        return "ok"
    return "out_of_band"


def _unit_reason(extraction: dict, spec: AnswerSpec) -> str | None:
    if spec.units is None:
        return None
    readings = unit_readings(extraction.get("unit"))
    if not readings:
        return "missing_unit" if spec.unit_required else None
    return None if readings & set(spec.units) else "wrong_unit"


def _accepted(canonical: str, spec: AnswerSpec) -> set[str]:
    return {normalize_name(canonical)} | {
        normalize_name(a) for a in spec.aliases.get(canonical, ())
    }


_SEGMENT_SPLIT = re.compile(r"\s*[,;:=]\s*|\s+[-\u2013\u2014]\s+")
_INT_TOKEN = re.compile(r"(?<![\w.])(\d+)(?:\.0+)?(?![\w.])")
_PARENS = re.compile(r"\(([^)]*)\)")


# Keys a code is given under ('eco1 = 6', 'NA_L1CODE: 6'). A fixed list: any
# other word next to a matching code ('Tundra = 6') may name a competing
# answer, so it is a conflict, not a label.
_KEY_LABELS = frozenset(
    {"eco1", "ecoregion", "nal1code", "l1", "level1", "code", "id", "region", "year"}
)


def _key_label(segment: str) -> bool:
    """A segment that is only the key a code is given under (_KEY_LABELS)."""
    return normalize_name(segment) in _KEY_LABELS


def _direct(text: str, spec: AnswerSpec) -> list[str]:
    """Readings of one piece of text as a whole: normalized, as a date (date
    answers), and its integer when it holds exactly one ('Ecoregion 6',
    'eco1 = 6', '6.0')."""
    raw = [text]
    ints = _INT_TOKEN.findall(text)
    if len(ints) == 1:
        raw += [ints[0], str(int(ints[0]))]
    out = []
    for r in raw:
        if spec.date:
            out.append(_parse_date(r))
        out.append(normalize_name(r))
    return [v for v in out if v]


def _variants(item: str, spec: AnswerSpec) -> list[str]:
    """Every reading of an item (whole, outside and inside parentheticals,
    each segment): the loose search grounded() uses. Matching uses the
    stricter _resolve()."""
    inside = _PARENS.findall(item)
    outside = _PARENS.sub(" ", item)
    pieces = [item, outside, *inside, *_SEGMENT_SPLIT.split(outside)]
    return [v for p in pieces if p.strip() for v in _direct(p, spec)]


def _hits(readings: list[str], candidates: dict[str, set[str]]) -> set:
    return {
        c for c, accepted in candidates.items() if any(v in accepted for v in readings)
    }


def _resolve(item: str, candidates: dict[str, set[str]], spec: AnswerSpec):
    """(canonical names an extracted item denotes, conflict).

    The item's text outside any parenthetical decides: as a whole ('Hayden
    Pass Fire (Colorado)', '6 (Northwestern ...)'; its lone integer only when
    it has no separator), else by its segments ('AUGUST COMPLEX, California',
    '6 - Name', 'eco1 = 6'): the first segment must match, or every matching
    segment is a code and the others are one-word key labels (_key_label);
    every matching segment denotes the same name, and every other one is a
    location qualifier (answers.LOCATION_QUALIFIERS) or such a key label. A
    parenthetical decides only when nothing stands outside it. `conflict` is
    set when no reading of the outside text matches but a segment or a
    parenthetical does ('Ecoregion 7 (neighbouring 6)', '2017 (vs 2020)',
    'AUGUST COMPLEX, Dixie', '6 - Marine West Coast Forest'): the item fails,
    and the trial goes to review."""
    inside = _PARENS.findall(item)
    outside = _PARENS.sub(" ", item) if inside else item
    segments = [s for s in _SEGMENT_SPLIT.split(outside) if s.strip()]
    whole = [normalize_name(item), normalize_name(outside)]
    if spec.date:
        whole += [_parse_date(item), _parse_date(outside)]
    if len(segments) <= 1:
        whole += _direct(outside, spec)  # incl. its lone integer ('Ecoregion 6')
    hits = _hits([w for w in whole if w], candidates)
    if hits:
        return hits, False
    if len(segments) > 1:
        seg_hits = [_hits(_direct(s, spec), candidates) for s in segments]
        found = [(s, h) for s, h in zip(segments, seg_hits, strict=True) if h]
        by_code = found and all(_INT_TOKEN.search(s) for s, _ in found)
        if (
            found
            and all(h == found[0][1] for _, h in found)
            and (seg_hits[0] or by_code)
            and all(
                h
                or normalize_name(s) in LOCATION_QUALIFIERS
                or (by_code and _key_label(s))
                for s, h in zip(segments, seg_hits, strict=True)
            )
        ):
            return found[0][1], False
        if found:
            return set(), True
    inner = set().union(*(_hits(_direct(p, spec), candidates) for p in inside))
    if inner:
        return (inner, False) if not normalize_name(outside) else (set(), True)
    return set(), False


def _matches(item: str, candidates: dict[str, set[str]], spec: AnswerSpec) -> set:
    """The canonical names an extracted item denotes (see _resolve)."""
    return _resolve(item, candidates, spec)[0]


def _match(item: str, candidates: dict[str, set[str]], spec: AnswerSpec) -> str | None:
    """The canonical name an extracted item denotes, if exactly one."""
    hits = _matches(item, candidates, spec)
    return next(iter(hits)) if len(hits) == 1 else None


def _name_candidates(expected, spec: AnswerSpec) -> tuple[list[str], dict]:
    exp = [str(e) for e in (expected if isinstance(expected, list) else [expected])]
    if spec.date:
        return exp, {e: {_parse_date(e) or e, normalize_name(e)} for e in exp}
    return exp, {e: _accepted(e, spec) for e in exp}


def name_conflict(question: Question, expected, extraction: dict | None) -> bool:
    """Whether some extracted item failed to match only because its outside
    text disagrees with a parenthetical or a later segment that does match
    (see _resolve): an unsure reading, sent to review."""
    if question.answer_kind not in ("text", "set") or not extraction:
        return False
    names = extraction.get("names") or []
    spec = ANSWERS[question.id]
    exp, candidates = _name_candidates(expected, spec)
    if not spec.compound:
        return any(_resolve(n, candidates, spec)[1] for n in names)
    parts = _compound_parts(names)
    for e in exp:
        year, _, name = e.partition(" / ")
        cands = {"year": {year}, "name": _accepted(name, spec)}
        if any(_resolve(p, cands, spec)[1] for p in parts):
            return True
    return False


def _names_reason(names: list[str], expected, question: Question, spec: AnswerSpec):
    exp, candidates = _name_candidates(expected, spec)
    if question.answer_kind == "set":
        mapped = [_match(n, candidates, spec) for n in names]
        if None in mapped or set(mapped) != set(exp):
            return "wrong_set"
        return "ok"
    if spec.compound:
        return _compound_reason(names, exp, spec)
    # Every item must be an expected name; naming several tied names is fine.
    if any(_match(n, candidates, spec) is None for n in names):
        return "wrong_name"
    return "ok"


_YEAR = re.compile(r"(?<![\d.])(?:19|20)\d\d(?![\d.])")
_COMPOUND_LEAD = re.compile(r"^(?:\W|\b(?:year|in|was|is|and|the)\b)+", re.IGNORECASE)


def _compound_parts(names: list[str]) -> list[str]:
    """Split compound items into year and name parts, in either order
    ('2006 / X', '2006: X', 'X, 2006', 'X (2006)', 'Year 2006: X')."""
    parts = []
    for n in names:
        for p in re.split(r"\s*/\s*", n):
            years = _YEAR.findall(p)
            rest = re.sub(r"\(\s*\)", " ", _YEAR.sub(" ", p)) if years else p
            rest = _COMPOUND_LEAD.sub("", rest)
            parts += years
            if normalize_name(rest):
                parts.append(rest)
    return parts


def _compound_reason(names: list[str], expected: list[str], spec: AnswerSpec) -> str:
    """Expected items are 'YEAR / NAME'; both parts required, nothing else."""
    parts = _compound_parts(names)
    partial = False  # every item matched a part, but a part is missing
    for exp in expected:
        year, _, name = exp.partition(" / ")
        cands = {"year": {year}, "name": _accepted(name, spec)}
        hits = [_matches(p, cands, spec) for p in parts]
        if all(hits) and set().union(*hits) == {"year", "name"}:
            return "ok"
        partial = partial or all(hits)
    return "missing_part" if partial else "wrong_name"


_BS_CODE = re.compile(r"\b(?:bs|class)\s*=?\s*([1-6])\b|^\s*([1-6])\s*$")
_LABEL_NOISE = re.compile(r"\b(severity|burn|class|code|pixels?|counts?)\b")


def _series_key(label: str, spec: AnswerSpec) -> str | None:
    """Canonical key for a response label: the class label wins over a code
    ('bs=2', 'class 3', or a lone code left once noise words are dropped:
    'Severity 1', 'burn_severity 3')."""
    text = label.casefold().replace("_", " ")
    readings = [_PARENS.sub(" ", text), *_PARENS.findall(text)]
    norms = [normalize_name(_LABEL_NOISE.sub(" ", r)) for r in readings]
    for norm in norms:
        for key, labels in spec.series_keys.items():
            if norm in {normalize_name(x) for x in labels}:
                return key
    parts = [x for x in text.split("/") if x.strip()]
    if len(parts) > 1:
        # 'Unburned to Low / Unburned': every alternative must name one class.
        keys = {_series_key(x, spec) for x in parts}
        if len(keys) == 1 and None not in keys:
            return keys.pop()
    m = _BS_CODE.search(text)
    code = (m.group(1) or m.group(2)) if m else None
    if code is None:
        code = next((n for n in norms if re.fullmatch(r"[1-6]", n)), None)
    if code is None:
        return None
    key = f"bs={code}"
    return key if key in spec.series_keys else None


def _series_reason(values: dict, expected: dict, question, spec) -> str:
    got: dict[str, float] = {}
    for label, v in values.items():
        key = _series_key(label, spec)
        if key is None:
            continue  # extra categories (bs=5/6, totals) are ignored
        if key in got and got[key] != v:
            return "conflicting_keys"
        got[key] = v
    if any(k not in got for k in expected):
        return "missing_key"
    for k, want in expected.items():
        reason = numeric_reason(got[k], None, want, question, spec)
        if reason != "ok":
            return reason
    return "ok"


def check_answer(question: Question, expected, extraction: dict) -> tuple[bool, str]:
    """(passes, reason) for an extraction against ground truth.

    Reasons: ok, clarification, refusal, no_answer, multiple_candidates,
    missing_value, wrong_unit, missing_unit, sign, not_exact, out_of_band,
    missing_key, conflicting_keys, wrong_name, missing_part, wrong_set."""
    spec = ANSWERS[question.id]
    status = extraction.get("status")
    if status != "answer":
        return False, status or "no_answer"
    if extraction.get("multiple_candidates"):
        return False, "multiple_candidates"
    kind = question.answer_kind
    if kind == "scalar":
        value = extraction.get("value")
        if value is None:
            return False, "missing_value"
        reason = _unit_reason(extraction, spec) or numeric_reason(
            value, extraction.get("value_text"), expected, question, spec
        )
    elif kind == "series":
        if not extraction.get("values"):
            return False, "missing_value"
        reason = _series_reason(extraction["values"], expected, question, spec)
    else:
        names = extraction.get("names") or []
        if not names:
            return False, "missing_value"
        reason = _names_reason(names, expected, question, spec)
    return reason == "ok", reason


# ---------------------------------------------------------------------------
# Regex cross-check
# ---------------------------------------------------------------------------

_SUPERSCRIPT = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a",
    "0123456789-+",
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-+])?"  # no sign read from '1984-1993' or 'T-5'
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE](?P<exp>[-+]?\d+))?"
    r"(?:\s*[x\u00d7*]\s*10\s*(?:\^\s*\(?(?P<pow>[-+]?\d+)\)?"
    r"|(?P<sup>[\u207b\u207a]?[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)))?"
    r"(?:\s*(?P<mult>thousand|million|billion|trillion)\b)?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}


def extract_decimals(text: str) -> list[Decimal]:
    """Every number in `text`, handling thousands separators, U+2212 minus,
    'e8', 'x 10^8' / 'x 10' + superscripts, and million/billion words."""
    text = (text or "").replace("\u2212", "-")
    out = []
    for m in _NUMBER_RE.finditer(text):
        d = Decimal(m["num"].replace(",", ""))
        power = 0
        if m["exp"]:
            power += int(m["exp"])
        if m["pow"]:
            power += int(m["pow"])
        if m["sup"]:
            power += int(m["sup"].translate(_SUPERSCRIPT))
        if m["mult"]:
            power += _MULTIPLIERS[m["mult"].lower()]
        d = d.scaleb(power)
        out.append(-d if m["sign"] == "-" else d)
    return out


def extract_numbers(text: str) -> list[float]:
    return [float(d) for d in extract_decimals(text)]


def regex_cross_check(question: Question, expected, extraction: dict | None, text: str):
    """None when skipped (non-scalar, answers.regex_check off, no extraction),
    else {kind, numbers, has_extracted, in_band, disagree}.

    Disagreement: the extracted value is not a number the response contains,
    or the regex's verdict (the text holds an in-band number) differs from the
    extractor-based answer verdict (check_answer: unit, sign, status and
    multiple_candidates included), so an in-band number in a response failed
    on its unit or on a candidate flag is reviewed, not silently failed."""
    spec = ANSWERS[question.id]
    if question.answer_kind != "scalar" or not spec.regex_check or not extraction:
        return None
    decs = extract_decimals(text)
    in_band = any(
        numeric_reason(float(d), str(d), expected, question, spec) == "ok" for d in decs
    )
    answer_ok, _ = check_answer(question, expected, extraction)
    out = {
        "kind": "number",
        "numbers": [str(d) for d in decs][:50],
        "in_band": in_band,
        "has_extracted": None,
    }
    value = extraction.get("value")
    if extraction.get("status") == "answer" and value is not None:
        out["has_extracted"] = any(
            math.isclose(float(d), value, rel_tol=1e-9, abs_tol=1e-12) for d in decs
        )
    out["disagree"] = out["has_extracted"] is False or in_band != answer_ok
    return out


_DATE_IN_TEXT = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|\b[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b"
    r"|\b\d{1,2} [A-Za-z]{3,9}\.? \d{4}\b"
)


def _contains(tokens: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return bool(n) and any(tokens[i : i + n] == phrase for i in range(len(tokens)))


def name_cross_check(question: Question, expected, extraction: dict | None, text: str):
    """Text/set counterpart of regex_cross_check: whether the response text
    contains the expected name(s) (any alias, as a token sequence; dates in any
    common format; for M04 both the year and the name). Disagreement: that
    differs from the extractor-based answer verdict. None for scalar/series
    questions or without an extraction."""
    if question.answer_kind not in ("text", "set") or not extraction:
        return None
    spec = ANSWERS[question.id]
    exp = [str(e) for e in (expected if isinstance(expected, list) else [expected])]
    tokens = name_tokens(text)

    def present(name: str) -> bool:
        forms = [name, *spec.aliases.get(name, ())]
        return any(_contains(tokens, name_tokens(f)) for f in forms)

    if spec.date:
        dates = {_parse_date(m) for m in _DATE_IN_TEXT.findall(text or "")}
        found = any((_parse_date(e) or e) in dates for e in exp)
    elif spec.compound:
        found = any(
            present(year) and present(name)
            for year, _, name in (e.partition(" / ") for e in exp)
        )
    elif question.answer_kind == "set":
        found = all(present(e) for e in exp)
    else:
        found = any(present(e) for e in exp)
    answer_ok, _ = check_answer(question, expected, extraction)
    return {"kind": "names", "found": found, "disagree": found != answer_ok}


def series_cross_check(question: Question, expected, extraction: dict | None, text):
    """Series counterpart: whether every expected value appears as a number
    in the response text (in band; exact for counts). Disagreement: that
    differs from the extractor-based verdict, e.g. every count is in the text
    but a label did not map to its key (missing_key / conflicting_keys). None
    for other kinds or without an extraction."""
    if question.answer_kind != "series" or not extraction:
        return None
    spec = ANSWERS[question.id]
    decs = extract_decimals(text)
    found = all(
        any(
            numeric_reason(float(d), str(d), want, question, spec) == "ok" for d in decs
        )
        for want in expected.values()
    )
    answer_ok, _ = check_answer(question, expected, extraction)
    return {"kind": "series", "found": found, "disagree": found != answer_ok}


def review_reasons(
    extraction: dict | None,
    answer_reason,
    cross,
    partial_evidence,
    numeric_ok,
    conflict: bool = False,
) -> list[str]:
    """Why a graded trial should be hand-adjudicated (each can hide a false
    verdict): the regex/name/series cross-check disagrees; the number is right
    but the unit failed (covers T04/T05, which skip the regex); a name failed
    on an unsure reading (name_conflict); or an in-band answer fails only on a
    head/tail/limit/slice heuristic hit the response did not disclose (a
    `.sample(`/`.fetch(` is unambiguous and is not reviewed)."""
    out = []
    if cross and cross.get("disagree"):
        out.append("cross_check_disagree")
    if answer_reason in ("wrong_unit", "missing_unit") and numeric_ok:
        out.append("unit_only")
    if conflict and answer_reason != "ok":
        out.append("name_reading_conflict")
    heuristic = partial_evidence and not any(
        pat in e for e in partial_evidence for pat in _PARTIAL_ANYWHERE
    )
    if answer_reason == "ok" and heuristic:
        if not (extraction or {}).get("sampled_disclosed"):
            out.append("partial_data_heuristic")
    return out


# ---------------------------------------------------------------------------
# Partial-data and grounding flags
# ---------------------------------------------------------------------------

_PARTIAL_ANYWHERE = (".sample(", ".fetch(", ".gather_every(")
_PARTIAL_FINAL = frozenset({"head", "tail", "limit", "slice"})
# Methods whose result a head() merely ranks (top-k selection, not sampling).
# value_counts ranks only when it sorts: pandas by default, polars only with
# sort=True (see _Lineage.is_ranked).
_RANKING = frozenset(
    {
        "sort",
        "sort_by",
        "top_k",
        "bottom_k",
        "sort_values",
        "nlargest",
        "nsmallest",
        "value_counts",
    }
)
# Row-order-preserving steps allowed between a sort and its head(): projections,
# row filters, renames, conversions (and group_by().head(), top-n per group).
# unique keeps order only with maintain_order=True in polars (is_ranked).
_ORDER_PRESERVING = frozenset(
    {
        "collect",
        "lazy",
        "to_pandas",
        "select",
        "with_columns",
        "with_column",
        "filter",
        "rename",
        "drop",
        "drop_nulls",
        "fill_null",
        "fill_nan",
        "cast",
        "with_row_index",
        "with_row_count",
        "head",
        "tail",
        "limit",
        "group_by",
        "groupby",
        "reset_index",
        "assign",
        "query",
        "dropna",
        "fillna",
        "astype",
        "round",
        "loc",
        "unique",
        "drop_duplicates",
    }
)
_AGGREGATES = frozenset(
    {
        "sum",
        "mean",
        "median",
        "count",
        "len",
        "n_unique",
        "nunique",
        "std",
        "var",
        "min",
        "max",
        "quantile",
        "mode",
        "value_counts",
        "describe",
        "agg",
        "aggregate",
        "product",
        "prod",
        "null_count",
        "arg_max",
        "arg_min",
        "idxmax",
        "idxmin",
        "unique_counts",
        "corr",
        "cov",
    }
)
# Builtins that pass their first argument's rows through unchanged.
_PASS_THROUGH_FUNCS = frozenset(
    {"int", "float", "str", "round", "list", "tuple", "dict", "bool", "abs", "sorted"}
)
_ROW_ACCESS_ATTRS = frozenset({"loc", "iloc", "at", "iat", "values"})
# In-place container mutations: the assignment map cannot follow them.
_MUTATORS = frozenset(
    {"append", "extend", "insert", "update", "add", "setdefault", "appendleft"}
)
# Calls peeled off a print()/display() argument to find a displayed head().
_DISPLAY_WRAPPERS = frozenset(
    {"collect", "to_pandas", "to_string", "to_markdown", "to_dict", "to_list", "rows"}
)
_DISPLAY_FUNCS = frozenset({"print", "display"})
_NAMESPACES = frozenset({"str", "list", "arr", "dt", "bin", "name", "struct", "cat"})
_EXPR_ROOTS = frozenset(
    {"col", "all", "first", "last", "len", "lit", "exclude", "nth", "element", "when"}
)


def _is_expression(node) -> bool:
    """A polars expression chain (rooted at pl.col / pl.all / ...)."""
    while True:
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "pl"
            ):
                return f.attr in _EXPR_ROOTS
            node = f
        elif isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        else:
            return False


def _target_names(target) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [n for t in target.elts for n in _target_names(t)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _assignments(tree) -> dict[str, list]:
    """name -> every value assigned to it anywhere in the code."""
    out: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [n for t in node.targets for n in _target_names(t)]
        elif isinstance(node, ast.AnnAssign | ast.AugAssign) and node.value:
            targets = _target_names(node.target)
        elif isinstance(node, ast.NamedExpr):
            targets = _target_names(node.target)
        else:
            continue
        for name in targets:
            out.setdefault(name, []).append(node.value)
    return out


def _opaque_names(tree) -> set[str]:
    """Names whose value the assignment map cannot follow: functions and
    classes defined in the code, for-loop / comprehension / with targets, and
    containers mutated in place (`rows.append(...)`, `d[k] = ...`)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            out.update(_target_names(node.target))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            out.update(_target_names(node.optional_vars))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
            and isinstance(node.func.value, ast.Name)
        ):
            out.add(node.func.value.id)
        elif isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript | ast.Attribute) and isinstance(
                    t.value, ast.Name
                ):
                    out.add(t.value.id)
    return out


def _kwarg(call, name: str):
    """(present, value) of a keyword argument; value is None unless it is a
    literal constant."""
    for kw in call.keywords:
        if kw.arg == name:
            return True, kw.value.value if isinstance(kw.value, ast.Constant) else None
    return False, None


class _Lineage:
    """Name resolution over a turn's successful queries (the worker namespace
    persists across run_query calls): a name read in query j resolves to its
    assignments in the latest query <= j that assigns it; a name read inside
    its own assignment (`df = df.sort(...)`) resolves to that name's other
    assignments in the same query and to an earlier query."""

    def __init__(self, trees: list):
        self.assigns = [_assignments(t) for t in trees]
        self.opaque = [_opaque_names(t) for t in trees]

    def resolve(self, name: str, upto: int) -> tuple[int, list]:
        for j in range(upto, -1, -1):
            if name in self.assigns[j]:
                return j, self.assigns[j][name]
        return -1, []

    def frame_lib(self, node, j: int, seen: set) -> str | None:
        """'pandas' or 'polars' for the frame a chain runs on, from the nearest
        marker below `node` (to_pandas / pd -> pandas; collect / lazy / lf / pl
        -> polars), following names; None when unknown."""
        while True:
            if isinstance(node, ast.Call):
                f = node.func
                if not isinstance(f, ast.Attribute):
                    return None
                if f.attr == "to_pandas":
                    return "pandas"
                if f.attr in ("collect", "lazy"):
                    return "polars"
                node = f.value
            elif isinstance(node, ast.Subscript | ast.Attribute):
                node = node.value
            elif isinstance(node, ast.Name):
                if node.id == "pd":
                    return "pandas"
                if node.id in ("pl", "lf"):
                    return "polars"
                k, values = self.resolve(node.id, j)
                if (node.id, k) in seen:
                    return None
                seen.add((node.id, k))
                libs = {self.frame_lib(v, k, seen) for v in values} - {None}
                return next(iter(libs)) if len(libs) == 1 else None
            else:
                return None

    def is_ranked(self, node, j: int, owner: str | None, seen: set) -> bool:
        while True:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "sorted"
            ):
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr == "value_counts":
                    present, value = _kwarg(node, "sort")
                    if present:
                        return value is True
                    # Only pandas sorts by default; an unknown frame (the
                    # sandbox is polars-native) is not ranked.
                    return self.frame_lib(node.func.value, j, set()) == "pandas"
                if attr in _RANKING:
                    return True
                if attr not in _ORDER_PRESERVING:
                    return False
                if attr == "unique":
                    present, value = _kwarg(node, "maintain_order")
                    keeps = (
                        value is True
                        if present
                        else self.frame_lib(node.func.value, j, set()) == "pandas"
                    )
                    if not keeps:
                        return False
                node = node.func.value
            elif isinstance(node, ast.Subscript):
                node = node.value  # column selection or a boolean row mask
            elif isinstance(node, ast.Attribute) and node.attr in ("loc", "iloc"):
                node = node.value
            elif isinstance(node, ast.Name):
                k, values = self.resolve(node.id, j - 1 if node.id == owner else j)
                if (node.id, k) in seen or not values:
                    return False
                seen.add((node.id, k))
                return any(self.is_ranked(v, k, node.id, seen) for v in values)
            else:
                return False

    def answer_exprs(self, last: int) -> tuple[list, set[int]]:
        """([(query index, expression, assigned name)], opaque queries) for
        every expression the final query's `result` is computed from,
        following names back through this and earlier queries. Opaque queries
        are those where a name in the lineage is bound in a way the assignment
        map cannot follow (_opaque_names). ([], set()) when the final query
        assigns no result."""
        todo = [(last, v, "result") for v in self.assigns[last].get("result", [])]
        out = []
        # `result = {}; result["m"] = ...` / `result.append(...)`
        opaque = {last} if todo and "result" in self.opaque[last] else set()
        seen = {("result", last)}
        while todo:
            j, expr, owner = todo.pop()
            out.append((j, expr, owner))
            for n in ast.walk(expr):
                if not isinstance(n, ast.Name):
                    continue
                if n.id == owner:
                    same = [v for v in self.assigns[j].get(n.id, []) if v is not expr]
                    groups = [(j, same), self.resolve(n.id, j - 1)]
                else:
                    groups = [self.resolve(n.id, j)]
                for k, values in groups:
                    if values and (n.id, k) not in seen:
                        seen.add((n.id, k))
                        todo += [(k, v, n.id) for v in values]
                first = min(k for k, _ in groups)
                opaque |= {
                    t for t in range(max(first, 0), j + 1) if n.id in self.opaque[t]
                }
        return out, opaque


def _displayed(tree) -> set[int]:
    """ids of head/tail/limit/slice calls and subscripts whose value goes
    straight to print() or display() (a preview, not the answer)."""
    out = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _DISPLAY_FUNCS
        ):
            continue
        args = list(node.args)
        while args:
            a = args.pop()
            if isinstance(a, ast.JoinedStr):
                args += [v.value for v in a.values if isinstance(v, ast.FormattedValue)]
                continue
            while (
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Attribute)
                and a.func.attr in _DISPLAY_WRAPPERS
            ):
                a = a.func.value
            if isinstance(a, ast.Subscript) or (
                isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute)
            ):
                out.add(id(a))
    return out


def _row_slice(node) -> bool:
    """A positional row cut by subscript: `df[:100]`, `df.iloc[:100]`,
    `rows[-5:]`, `df.iloc[:100, :]` (a bound on the first axis)."""
    if not isinstance(node, ast.Subscript):
        return False
    sl = node.slice
    if isinstance(sl, ast.Tuple) and sl.elts:
        sl = sl.elts[0]
    return isinstance(sl, ast.Slice) and (sl.lower is not None or sl.upper is not None)


def _single_row(node) -> bool:
    """head(1) / limit(1) / tail(1) / slice(k, 1) / [:1] / [k:k+1]."""
    if isinstance(node, ast.Call):
        args = node.args
        n = args[1] if node.func.attr == "slice" and len(args) > 1 else None
        if node.func.attr != "slice":
            n = (
                args[0]
                if args
                else next((kw.value for kw in node.keywords if kw.arg == "n"), None)
            )
        return isinstance(n, ast.Constant) and n.value == 1
    sl = node.slice
    if isinstance(sl, ast.Tuple) and sl.elts:
        sl = sl.elts[0]
    lo = (
        sl.lower.value
        if isinstance(sl.lower, ast.Constant)
        else 0
        if sl.lower is None
        else None
    )
    hi = sl.upper.value if isinstance(sl.upper, ast.Constant) else None
    return isinstance(lo, int) and isinstance(hi, int) and hi - lo == 1


def _filtered(node) -> bool:
    """Whether a method chain has a row filter below this point."""
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("filter", "query", "where"):
                return True
            node = node.func.value
        elif isinstance(node, ast.Subscript | ast.Attribute):
            node = node.value
        else:
            return False


def _cut_label(node) -> str:
    if isinstance(node, ast.Call):
        return f".{node.func.attr}( at line {node.lineno}"
    via = f".{node.value.attr}" if isinstance(node.value, ast.Attribute) else ""
    return f"{via}[{ast.unparse(node.slice)}] at line {node.lineno}"


def _aggregated_after(node, parents: dict) -> bool:
    """Whether an expression-level cut is the receiver of a later aggregate in
    its own method chain (pl.col('a').head(100).mean())."""
    while True:
        attr = parents.get(id(node))
        if not (isinstance(attr, ast.Attribute) and attr.value is node):
            return False
        call = parents.get(id(attr))
        if not (isinstance(call, ast.Call) and call.func is attr):
            return False
        if attr.attr in _AGGREGATES:
            return True
        node = call


def _cuts(lineage: _Lineage, j: int, expr, owner, skip: set[int]) -> list:
    """Row-cutting head/tail/limit/slice calls and positional slices
    (_row_slice) inside `expr` (query j), except ranked and tie-pick cuts."""
    found = []
    parents = {id(c): n for n in ast.walk(expr) for c in ast.iter_child_nodes(n)}
    for node in ast.walk(expr):
        if id(node) in skip:
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PARTIAL_FINAL
        ):
            recv = node.func.value
        elif _row_slice(node):
            recv = node.value
            if isinstance(recv, ast.Constant | ast.JoinedStr):
                continue
        else:
            continue
        if isinstance(recv, ast.Attribute) and recv.attr in _NAMESPACES:
            continue
        if _is_expression(recv) and not _aggregated_after(node, parents):
            continue  # agg(pl.col('n').head(1)): top-n per group, not a cut
        # A cut after a ranking step is top-N selection, never sampling,
        # whatever is computed from it: the first real run (pro-vs-flash,
        # 250 trials) flagged 23 correct "among the N largest" answers and no
        # real sample. A one-row cut of filtered rows is a tie pick
        # (`filter(elevation == max).head(1)`).
        if (_single_row(node) and _filtered(recv)) or lineage.is_ranked(
            recv, j, owner, set()
        ):
            continue
        found.append(node)
    return found


def _truncations(codes: list[str]) -> list[tuple[int, str]]:
    """(index into codes, '.head( at line N' / '[:100] at line N') for each
    row-cutting head/tail/limit/slice or positional slice (`df[:100]`,
    `.iloc[:100]`) that feeds the last code's `result`. codes are a turn's
    successful queries in order, up to the one that produced the answer.

    Not counted: a cut after sort/top_k/sorted value_counts (through
    order-preserving steps such as select/filter/with_columns/ordered unique,
    and through names assigned in this or an earlier query), whatever is then
    computed from it (top-N selection, not sampling); a one-row cut of
    filtered rows (`filter(e == max).head(1)`, a tie pick); an
    expression-level head inside an aggregation (top-n per group);
    `.str.slice`; and a head that only feeds print()/display() or a value
    `result` does not use. When the lineage uses a local function, a
    loop/comprehension/with target or a container mutated in place
    (including `result` itself: `result = {}` then `result["m"] = ...`),
    every non-displayed cut in the query that binds it counts. When the last
    code assigns no `result` (a plot-only call), every cut in it except
    displayed and ranked ones counts. Falls back to a substring match on the
    last code if it does not parse."""
    try:
        trees = [ast.parse(codes[-1])]
    except SyntaxError:
        last = len(codes) - 1
        return [
            (last, f".{m}(") for m in sorted(_PARTIAL_FINAL) if f".{m}(" in codes[-1]
        ]
    earlier = []
    for c in codes[:-1]:
        try:
            earlier.append(ast.parse(c))
        except SyntaxError:
            earlier.append(ast.Module(body=[], type_ignores=[]))
    trees = earlier + trees
    last = len(trees) - 1
    lineage = _Lineage(trees)
    exprs, opaque = lineage.answer_exprs(last)
    if not exprs:
        work = [(last, trees[last], None, _displayed(trees[last]))]
    else:
        work = [(j, e, owner, set()) for j, e, owner in exprs]
    work += [(k, trees[k], None, _displayed(trees[k])) for k in sorted(opaque)]
    hits = {}
    for j, expr, owner, skip in work:
        for node in _cuts(lineage, j, expr, owner, skip):
            hits[id(node)] = (j, _cut_label(node))
    return sorted(set(hits.values()))


def truncating_calls(code: str) -> list[str]:
    """_truncations() for one self-contained query."""
    return [msg for _, msg in _truncations([code])]


def partial_data(
    trace: dict,
    question: Question | None = None,
    extraction: dict | None = None,
) -> tuple[bool, list[str]]:
    """(flag, evidence): `.sample(`/`.fetch(`/`.gather_every(` in any executed
    query, or a row-cutting head/tail/limit/slice or positional slice on what
    produced the answer: the answer query's `result` and the values it is
    built from (exploratory or preview head() calls are fine; see
    _truncations). The answer query is the latest successful query whose
    result holds the extracted answer (answer_query), else the last
    successful one; later queries (a name lookup after the answer) are not
    checked for cuts."""
    records = trace.get("query_records") or []
    evidence = [
        f"{pat} in query {i}"
        for i, r in enumerate(records)
        for pat in _PARTIAL_ANYWHERE
        if pat in (r.get("code") or "")
    ]
    ok = [i for i, r in enumerate(records) if not r.get("error")]
    if ok:
        found = answer_query(question, trace, extraction) if question else None
        upto = ok.index(found) + 1 if found is not None else len(ok)
        codes = [records[i].get("code") or "" for i in ok[:upto]]
        label = "answer query" if found is not None else "final query"
        for j, msg in _truncations(codes):
            where = label if j == upto - 1 else "query"
            evidence.append(f"{msg} in {where} {ok[j]}")
    return bool(evidence), evidence


def _walk(obj, nums: list[float], strs: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            strs.add(normalize_name(k))
            _walk(v, nums, strs)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, nums, strs)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, int | float):
        nums.append(float(obj))
        if float(obj).is_integer():
            strs.add(str(int(obj)))
    elif isinstance(obj, str):
        strs.add(normalize_name(obj))
        iso = _parse_date(obj)
        if iso:
            strs.add(iso)


def _record_values(record: dict, nums: list[float], strs: set[str]) -> None:
    data = record.get("data")
    if data is None:
        return
    try:
        _walk(json.loads(data) if isinstance(data, str) else data, nums, strs)
    except ValueError:  # capped (truncated) JSON: fall back to the text
        nums += extract_numbers(data)
        strs |= {normalize_name(s) for s in re.findall(r'"([^"]*)"', data)}


def _result_values(records: list[dict]) -> tuple[list[float], set[str]]:
    nums: list[float] = []
    strs: set[str] = set()
    for r in records:
        _record_values(r, nums, strs)
    return nums, strs


# Integer answers below this appear in many results by coincidence (M05's 5),
# so they cannot locate the answer query.
_WEAK_INT = 1000


def answer_query(question: Question, trace: dict, extraction: dict | None):
    """Index of the latest successful query_record whose result holds the
    extracted answer (see grounded), or None, also for a small integer answer
    (_WEAK_INT), which would match by coincidence."""
    value = (extraction or {}).get("value")
    if question.answer_kind == "scalar" and (
        not isinstance(value, int | float)
        or (float(value).is_integer() and abs(value) < _WEAK_INT)
    ):
        return None
    records = trace.get("query_records") or []
    for i in range(len(records) - 1, -1, -1):
        r = records[i]
        if r.get("error") or r.get("data") is None:
            continue
        if _answer_in(question, trace.get("expected"), extraction, [r]):
            return i
    return None


def grounded(question: Question, expected, extraction: dict | None, trace: dict):
    """Whether the extracted answer appears (within the question's band) in
    some query result. A flag only: ratios like T03 are legitimately computed
    from returned means. None when there is no answer to look for."""
    return _answer_in(question, expected, extraction, trace.get("query_records") or [])


def _answer_in(question: Question, expected, extraction, records: list[dict]):
    if not extraction or extraction.get("status") != "answer":
        return None
    nums, strs = _result_values(records)
    kind = question.answer_kind
    if kind in ("scalar", "series"):
        if kind == "scalar":
            wanted = [extraction.get("value")]
            bands = [scalar_band(question, float(expected))]
        else:
            # Values whose label maps to an expected key, banded by that
            # key's expected value (extra categories are ignored, as in
            # check_answer).
            spec = ANSWERS[question.id]
            wanted, bands = [], []
            for label, v in (extraction.get("values") or {}).items():
                key = _series_key(label, spec)
                if key in expected:
                    wanted.append(v)
                    bands.append(scalar_band(question, float(expected[key])))
        if not wanted or None in wanted:
            return None
        return all(
            any(abs(n - float(w)) <= b for n in nums)
            for w, b in zip(wanted, bands, strict=True)
        )
    names = extraction.get("names") or []
    if not names:
        return None
    spec = ANSWERS[question.id]
    items = [p for n in names for p in re.split(r"\s*/\s*", n) if p.strip()]
    return all(any(v in strs for v in _variants(i, spec)) for i in items)


# ---------------------------------------------------------------------------
# Per-trace grading
# ---------------------------------------------------------------------------


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/trial_*.json"))


def grade_trace(trace: dict, client) -> dict:
    """Compute the grade fields for one trace (no file I/O; one extractor
    call unless the trace is an infra error or has no response text)."""
    question = BY_ID[trace["question_id"]]
    fields = {}
    if not trace.get("infra_error") and not trace.get("model_error"):
        # The current terminal-failure rules, for traces run before one existed
        # (an empty STOP final response was not a model error until 2026-10-01).
        late = terminal_model_error(trace.get("calls") or [])
        if late:
            fields.update(
                model_error=late,
                error_bucket="model",
                model_error_source="grade",
                model_anomalies=call_anomalies(trace.get("calls") or []),
            )
            trace = {**trace, **fields}
    fields |= {
        "grader_sha": grader_sha(),
        "grader_version": GRADER_VERSION,
        "resource_violation": resource_violation(trace),
        "had_query_errors": bool(_all_errors(trace)),
        "extractor": None,
        "extraction": None,
        "answer_ok": None,
        "answer_reason": None,
        "partial_data": False,
        "partial_data_evidence": [],
        "sampled_disclosed": None,
        "grounded": None,
        "regex_check": None,
        "regex_disagree": False,
        "review_reasons": [],
        "needs_review": False,
        "grader_error": None,
        "auto_accurate": False,
        "auto_reason": None,
    }
    if trace.get("infra_error"):
        executable, reason = is_executable(trace)
        fields.update(
            executable=executable, executability_reason=reason, auto_reason=reason
        )
        return fields
    ext = judge.extract(
        client, question.text, trace.get("text") or "", question.answer_kind
    )
    extraction = ext.pop("extraction")
    fields["extractor"] = ext
    fields["extraction"] = extraction
    executable, reason = is_executable(
        trace, extraction.get("status") if extraction else None
    )
    fields.update(executable=executable, executability_reason=reason)
    partial, evidence = partial_data(trace, question, extraction)
    fields.update(partial_data=partial, partial_data_evidence=evidence)
    if extraction is None:
        fields.update(grader_error=ext["error"], auto_reason="grader_error")
        return fields
    expected = trace["expected"]
    ok, why = check_answer(question, expected, extraction)
    text = trace.get("text") or ""
    cross = (
        regex_cross_check(question, expected, extraction, text)
        or name_cross_check(question, expected, extraction, text)
        or series_cross_check(question, expected, extraction, text)
    )
    value = extraction.get("value")
    spec = ANSWERS[question.id]
    numeric_ok = (
        question.answer_kind == "scalar"
        and value is not None
        and numeric_reason(
            value, extraction.get("value_text"), expected, question, spec
        )
        == "ok"
    )
    review = review_reasons(
        extraction,
        why,
        cross,
        evidence,
        numeric_ok,
        conflict=name_conflict(question, expected, extraction),
    )
    fields.update(
        answer_ok=ok,
        answer_reason=why,
        sampled_disclosed=extraction.get("sampled_disclosed"),
        grounded=grounded(question, expected, extraction, trace),
        regex_check=cross,
        regex_disagree=bool(cross and cross["disagree"]),
        review_reasons=review,
        needs_review=bool(review),
        auto_accurate=ok and not partial,
        auto_reason=why if not ok else ("partial_data" if partial else "ok"),
    )
    return fields


def stale_reference(trace: dict) -> str | None:
    """Why a trace's embedded `expected` can't be trusted against the current
    reference code, or None. A trace records the reference_sha its expected
    value came from; grading it after the reference changed would score the
    model against a stale answer."""
    question = BY_ID.get(trace.get("question_id"))
    if question is None:
        return f"unknown question id {trace.get('question_id')!r}"
    got = trace.get("reference_sha")
    if got is None:
        return "trace has no reference_sha (recorded before provenance tracking)"
    want = reference_sha(question)
    if got != want:
        return f"reference_sha {got} != current {want}"
    return None


class StaleTraceError(Exception):
    pass


def is_current_grade(trace: dict) -> bool:
    """Graded by this grader version, without a grader error."""
    return (
        "accurate" in trace
        and trace.get("grader_sha") == grader_sha()
        and not trace.get("grader_error")
    )


# ---------------------------------------------------------------------------
# Adjudication overlay
# ---------------------------------------------------------------------------

_PASS_WORDS = {"pass", "correct", "true", "yes", "accurate"}
_FAIL_WORDS = {"fail", "wrong", "false", "no", "inaccurate", "incorrect"}


def parse_verdict(value) -> bool | None:
    """True/False for a hand-filled verdict, None when blank or unrecognized."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().casefold()
        if v in _PASS_WORDS:
            return True
        if v in _FAIL_WORDS:
            return False
    return None


class AdjudicationFileError(Exception):
    pass


def load_adjudication(run_dir: Path) -> list[dict]:
    """Entries of <run_dir>/adjudication.json ([] if absent). Raises
    AdjudicationFileError if the file exists but is not a list of objects:
    it holds hand-filled verdicts and must never be silently replaced."""
    path = Path(run_dir) / ADJUDICATION_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise AdjudicationFileError(
            f"{path}: unreadable ({e}); fix or remove it"
        ) from e
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise AdjudicationFileError(f"{path}: expected a JSON list of objects")
    return data


def response_sha(trace: dict) -> str:
    """Hash of a trace's response text: adjudication and audit entries record
    it, so a verdict given on one response is never applied to another (a
    trial re-run on resume)."""
    return hashlib.sha256((trace.get("text") or "").encode()).hexdigest()[:16]


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _touched(entry: dict) -> bool:
    """A human filled any of verdict / reason / adjudicator."""
    return not all(_blank(entry.get(f)) for f in ("verdict", "reason", "adjudicator"))


def adjudication_index(entries: list[dict]) -> dict[tuple, list[dict]]:
    """(question_id, trial) -> that trial's entries, in file order, with
    `verdict` parsed to True/False, or None plus `unparseable` (and the text
    in `raw_verdict`) when a non-blank verdict is not a pass/fail word."""
    out: dict[tuple, list[dict]] = {}
    for e in entries:
        raw = e.get("verdict")
        verdict = parse_verdict(raw)
        bad = verdict is None and not _blank(raw)
        if bad:
            logger.warning(
                "adjudication %s trial %s: unrecognized verdict %r, not applied",
                e.get("question_id"),
                e.get("trial"),
                raw,
            )
        key = (e.get("question_id"), e.get("trial"))
        out.setdefault(key, []).append(
            {**e, "verdict": verdict, "raw_verdict": raw, "unparseable": bad}
        )
    return out


def apply_adjudication(trace: dict, index: dict[tuple, list[dict]]) -> None:
    """Set the final `accurate` / `accuracy_reason` / `adjudicated` fields:
    the last decided entry for the trial wins over the automated verdict,
    unless it records a response_sha that no longer matches the trace (a
    re-run trial). Entries without one (hand-added) always apply. A trial
    whose only verdicts are stale or unparseable keeps the automated verdict
    and gets `adjudication_issue` {kind: stale | unparseable, verdict}."""
    if "auto_accurate" not in trace:
        return  # legacy or ungraded trace: leave as is
    trace["adjudication_issue"] = None
    entries = (
        []
        if trace.get("infra_error")
        else index.get((trace.get("question_id"), trace.get("trial")), [])
    )
    current = response_sha(trace)
    decided = [e for e in entries if e["verdict"] is not None]
    fresh = [e for e in decided if e.get("response_sha") in (None, "", current)]
    if not fresh:
        trace["accurate"] = trace["auto_accurate"]
        trace["accuracy_reason"] = trace["auto_reason"]
        trace["adjudicated"] = None
        issue = decided[-1:] or [e for e in entries if e["unparseable"]][-1:]
        if issue:
            trace["adjudication_issue"] = {
                "kind": "stale" if decided else "unparseable",
                "verdict": issue[0]["raw_verdict"],
            }
        return
    entry = fresh[-1]
    trace["accurate"] = entry["verdict"]
    trace["accuracy_reason"] = "adjudicated"
    trace["adjudicated"] = {
        "verdict": entry["verdict"],
        "reason": entry.get("reason", ""),
        "adjudicator": entry.get("adjudicator", ""),
        "overrode": entry["verdict"] != trace["auto_accurate"],
    }


def _adjudication_context(trace: dict) -> dict:
    return {
        "review_reasons": trace.get("review_reasons"),
        "auto_verdict": trace.get("auto_accurate"),
        "auto_reason": trace.get("auto_reason"),
        "partial_data_evidence": trace.get("partial_data_evidence"),
        "expected": trace.get("expected"),
        "extraction": trace.get("extraction"),
        "regex_check": trace.get("regex_check"),
        "trace_path": trace.get("_path"),
        "response_sha": response_sha(trace),
        "response": (trace.get("text") or "")[:2000],
    }


def write_adjudication(run_dir: Path, traces: list[dict], entries: list[dict]) -> list:
    """Add a pending entry per trial flagged for review (review_reasons) not
    yet listed. An entry a human touched (any of verdict / reason /
    adjudicator filled, recognized or not) is never dropped and keeps its
    fields; the only entries pruned are auto-added ones left entirely blank
    whose trace is no longer flagged. Auto-added entries get their context
    refreshed, except that a touched entry keeps the response (and hash) it
    was judged on, and a touched entry whose trace changed since (stale) is
    left exactly as it is; a flagged trial with only stale entries gets a new
    pending entry for its current response next to them."""
    by_key = {(t["question_id"], t["trial"]): t for t in traces}
    flagged = {k for k, t in by_key.items() if t.get("needs_review")}
    out, covered = [], set()  # covered: keys with an entry for the current response
    for e in entries:
        key = (e.get("question_id"), e.get("trial"))
        auto = e.get("source") == ADJUDICATION_SOURCE
        touched = _touched(e)
        if auto and not touched and key not in flagged:
            continue
        sha = e.get("response_sha")
        if not (sha and key in by_key and sha != response_sha(by_key[key])):
            covered.add(key)
        if auto and key in by_key:
            ctx = _adjudication_context(by_key[key])
            if touched:
                if sha and sha != ctx["response_sha"]:
                    out.append(e)  # stale: keep what the adjudicator saw
                    continue
                for f in ("response", "response_sha"):
                    if f in e:
                        ctx[f] = e[f]
                    else:
                        ctx.pop(f)
            else:
                covered.add(key)  # refreshed below to the current response
            e = {**e, **ctx}
        out.append(e)
    # A flagged trial whose only entries are stale (re-run since a human
    # judged it) gets a fresh pending entry for its current response.
    for key in sorted(flagged - covered):
        out.append(
            {
                "question_id": key[0],
                "trial": key[1],
                "verdict": "",
                "reason": "",
                "adjudicator": "",
                "source": ADJUDICATION_SOURCE,
                **_adjudication_context(by_key[key]),
            }
        )
    path = Path(run_dir) / ADJUDICATION_FILE
    if out or path.exists():
        atomic_write_json(path, out)
    pending = sum(1 for k in flagged if not by_key[k].get("adjudicated"))
    if pending:
        logger.info(
            "adjudication: %d trial(s) pending review -> %s (fill `verdict` "
            "with pass/fail, then re-run grade or report)",
            pending,
            path,
        )
    return out


# ---------------------------------------------------------------------------
# Run-level grading + triage
# ---------------------------------------------------------------------------


def grade_run(run_dir: Path, client, *, regrade: bool = False) -> list[dict]:
    """Grade every trace in a run dir (merging fields in place), then write
    adjudication.json (pending disagreements) and triage.json.

    A trace is (re)graded unless it already carries a grade from the current
    grader_sha without a grader error; --regrade forces all. Refuses (before
    any extractor call) if a trace that would be graded was produced by
    different reference code than the current questions.py. Unreadable traces
    are skipped with an error log."""
    entries = load_adjudication(run_dir)  # refuse early on a corrupt overlay
    loaded = []
    for path in _trace_paths(run_dir):
        trace, err = read_trace(path)
        if trace is None:
            logger.error("%s: unreadable trace, skipped (%s)", path, err)
            continue
        loaded.append((path, trace))

    def needs(trace):
        return regrade or not is_current_grade(trace)

    stale = [
        f"{path}: {why}"
        for path, trace in loaded
        if needs(trace)
        and not trace.get("infra_error")
        and (why := stale_reference(trace))
    ]
    if stale:
        raise StaleTraceError(
            "refusing to grade traces whose ground truth no longer matches the "
            "current reference code (re-run those trials):\n  " + "\n  ".join(stale)
        )
    decided = adjudication_index(entries)
    traces = []
    for path, trace in loaded:
        if needs(trace):
            for key in _LEGACY_FIELDS:
                trace.pop(key, None)
            trace.update(grade_trace(trace, client))
            apply_adjudication(trace, decided)
            atomic_write_json(path, trace)
            logger.info(
                "%s trial %d: executable=%s accurate=%s (%s)",
                trace["question_id"],
                trace["trial"],
                trace.get("executable"),
                trace.get("accurate"),
                trace.get("accuracy_reason"),
            )
        else:
            logger.info("%s: already graded, skipping", path)
            fields = ("accurate", "adjudicated", "adjudication_issue")
            before = [trace.get(f) for f in fields]
            apply_adjudication(trace, decided)
            if [trace.get(f) for f in fields] != before:
                atomic_write_json(path, trace)
        trace["_path"] = str(path)
        traces.append(trace)
    errors = [t for t in traces if t.get("grader_error")]
    if errors:
        logger.error(
            "%d trace(s) have grader errors (extractor failed twice); re-run "
            "grade to retry them: %s",
            len(errors),
            ", ".join(f"{t['question_id']} trial {t['trial']}" for t in errors),
        )
    write_adjudication(run_dir, traces, entries)
    write_triage(run_dir, traces)
    return traces


def write_triage(run_dir: Path, traces: list[dict]) -> None:
    """triage.json: one entry per failed (non-accurate, non-infra) trial, for
    MANUAL failure_mode annotation (logical_error / domain_semantic_error /
    performance_violation = first point of divergence from the reference
    query). Hand-filled failure_mode values survive re-grading; entries for
    now-passing trials are dropped, and a trial whose regrade hit a grader
    error keeps its previous entry unchanged."""
    triage_path = run_dir / "triage.json"
    existing = {}
    if triage_path.exists():
        for entry in json.loads(triage_path.read_text()):
            existing[(entry["question_id"], entry["trial"])] = entry
    entries = []
    for trace in traces:
        if trace.get("infra_error") or trace.get("accurate"):
            continue
        key = (trace["question_id"], trace["trial"])
        if trace.get("grader_error"):
            # Not a verdict; the next grade retries it. Keep any old entry.
            if key in existing:
                entries.append(existing[key])
            continue
        if "accurate" not in trace:
            continue  # ungraded (shouldn't happen after grade_run)
        old = existing.get(key, {})
        entries.append(
            {
                "question_id": trace["question_id"],
                "trial": trace["trial"],
                "trace_path": trace["_path"],
                "executable": trace["executable"],
                "executability_reason": trace["executability_reason"],
                "accuracy_reason": trace.get("accuracy_reason"),
                "resource_violation": trace["resource_violation"],
                "summary": (trace.get("text") or "")[:300],
                "failure_mode": old.get("failure_mode", ""),
            }
        )
    atomic_write_json(triage_path, entries)
    unannotated = sum(1 for e in entries if not e["failure_mode"])
    logger.info(
        "triage: %d failed trials (%d unannotated) -> %s",
        len(entries),
        unannotated,
        triage_path,
    )

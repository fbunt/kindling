"""Bench grading of the hard tier: the fields and disambiguate verdicts and
their review routes, the answer query of a fields answer, executability and
partial data per kind, the hard grader sha, a hard run graded end to end, and
its report, compare and audit. No dataset, no API key, no container runtime:
extractions are hand-built and the extractor is a scripted fake."""

import dataclasses
import json

import pytest

from bench import grading, grading_ext, grading_hard, judge
from bench.answers import HARD_ANSWERS, spec_for
from bench.grading import (
    ADJUDICATION_FILE,
    FAILURE_MODES,
    answer_query,
    ext_grader_sha,
    grade_run,
    grade_trace,
    grader_sha,
    grader_sha_for,
    hard_grader_sha,
    is_current_grade,
    is_executable,
)
from bench.ground_truth import reference_sha
from bench.io import atomic_write_json
from bench.questions import BY_ID
from tests.test_bench_grading import FakeExtractor, _by_response, _reply

# Made-up ground truth in each hard question's expected shape.
HARD = {
    "H01": {"state": "FL", "count": 1388},
    "H04": {
        "year": 1994,
        "count": 3,
        "events": [
            {"Event_ID": f"ID{i}", "Incid_Type": 1, "acres": 100.0 * i}
            for i in range(1, 4)
        ],
    },
    "H05": {"percent": 24.6775, "name": "YOLLA BOLLY COMPLEX (IRON)", "year": 2008},
    "H06": 212,
    "H07": 4.208333,
}


def _trace(qid, trial=0, text="answer text", records=(), **extra):
    """A hard trace; records are (code, data) pairs or full dicts."""
    q = BY_ID[qid]
    recs = [
        r
        if isinstance(r, dict)
        else {"code": r[0], "error": None, "data": json.dumps(r[1]), "latency_s": 1}
        for r in records
    ]
    return {
        "question_id": qid,
        "trial": trial,
        "category": q.category,
        "answer_kind": q.answer_kind,
        "tier": "hard",
        "expected": HARD[qid],
        "reference_sha": reference_sha(q),
        "parquet_identity": "pid",
        "infra_error": None,
        "model_error": None,
        "text": text,
        "tool_calls": [
            {"name": "run_query", "args": {"code": r["code"]}} for r in recs
        ],
        "query_records": recs,
        "rejected_queries": [],
        "loop_exhausted": False,
        "turn_latency_s": 5.0,
        **extra,
    }


def _part(text=None, value=None, unit=None):
    """A parsed part (judge._parse_parts): `text` as written, `value_text`
    the number's decimal text."""
    return {
        "text": text,
        "value": value,
        "value_text": None if value is None else str(value),
        "unit": unit,
    }


def _fields(status="answer", multiple=False, **parts):
    return {
        "status": status,
        "multiple_candidates": multiple,
        "sampled_disclosed": False,
        "parts": parts,
    }


def _check(qid, extraction, text="answer text", records=()):
    q = BY_ID[qid]
    trace = _trace(qid, text=text, records=records)
    return grading_hard.check(q, HARD[qid], extraction, trace, spec_for(q))


def _reply_part(text=None, value=None, unit=None):
    return {"value_text": text, "value": value, "unit": unit}


def _grade(qid, reply: dict, text="answer text", records=(), **extra):
    trace = _trace(qid, text=text, records=records, **extra)
    return grade_trace(trace, FakeExtractor(lambda p: _reply(**reply)))


H01_RIGHT = {"state": _part("Florida (FL)"), "count": _part("1,388", 1388)}
H05_RIGHT = {
    "percent": _part("24.7%", 24.7, "%"),
    "name": _part("Yolla Bolly Complex (Iron)"),
    "year": _part("2008", 2008),
}


def _h01(**over):
    return _fields(**{**H01_RIGHT, **over})


def _h05(**over):
    return _fields(**{**H05_RIGHT, **over})


# ---------------------------------------------------------------------------
# fields: H01 (state, count)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extraction", "verdict", "reason", "review"),
    [
        (_h01(), True, "ok", []),
        (_h01(state=_part("FL")), True, "ok", []),
        (_h01(state=_part("fl")), True, "ok", []),
        (_h01(state=_part("Florida")), True, "ok", []),
        (_h01(state=_part("FLORIDA, USA")), True, "ok", []),
        # Two-letter words that are not codes here: the 'ID' of 'Event ID',
        # the 'IN' of 'IN total'.
        (_h01(state=_part("FL (from the Event ID prefix)")), True, "ok", []),
        (_h01(state=_part("Event ID prefix FL")), True, "ok", []),
        (_h01(state=_part("Florida, IN total")), True, "ok", []),
        (_h01(state=_part("Georgia (GA)")), False, "state:wrong_state", []),
        (_h01(state=_part("the Southeast")), False, "state:wrong_state", []),
        (_h01(state=_part()), False, "state:missing_value", []),
        # A guessed state without the count fails.
        (_h01(count=_part()), False, "count:missing_value", []),
        (_h01(count=_part("1,387", 1387)), False, "count:not_exact", []),
        # A second state: a hedge, or one named in passing.
        (
            _h01(state=_part("Florida, ahead of Georgia")),
            False,
            "state:several_states",
            ["second_state"],
        ),
        (
            _h01(state=_part("FL or GA")),
            False,
            "state:several_states",
            ["second_state"],
        ),
        (
            _h01(state=_part("Florida or Georgia")),
            False,
            "state:several_states",
            ["second_state"],
        ),
        (_h01(state=_part("FL, GA")), False, "state:several_states", ["second_state"]),
        (_h01(state=_part("Georgia or Texas")), False, "state:wrong_state", []),
        # 'West Virginia' is one state, not also Virginia.
        (_h01(state=_part("West Virginia")), False, "state:wrong_state", []),
        (_fields("clarification"), False, "clarification", []),
        (_fields("refusal", **H01_RIGHT), False, "refusal", []),
        # Right parts plus another candidate: fails, reviewed as a hedge.
        (
            _fields(multiple=True, **H01_RIGHT),
            False,
            "multiple_candidates",
            ["multiple_candidates"],
        ),
        (
            _fields(multiple=True, **{**H01_RIGHT, "count": _part("1,200", 1200)}),
            False,
            "multiple_candidates",
            [],
        ),
    ],
)
def test_state_and_count_verdicts(extraction, verdict, reason, review):
    assert _check("H01", extraction) == (verdict, reason, review)


def test_states_named():
    named = grading_hard.states_named
    assert named("Florida (FL)") == {"fl"}
    assert named("New Mexico") == {"nm"}
    assert named("North Carolina and South Carolina") == {"nc", "sc"}
    assert named("Florida, in 2010-2019") == {"fl"}  # 'in' is not Indiana
    assert named("Event_ID prefix FL") == {"fl"}
    assert named("Event ID prefix FL") == {"fl"}
    assert named("FL (from the Event ID prefix)") == {"fl"}
    assert named("the Event ID") == set()
    assert named("Florida, IN total") == {"fl"}
    assert named("Idaho (ID)") == {"id"}
    # Codes standing as codes: a list of them, in any case.
    assert named("FL, GA") == named("fl/ga") == named("FL or GA") == {"fl", "ga"}
    assert named("FL (or possibly GA)") == {"fl", "ga"}
    assert named("or") == {"or"}  # a part that is only a code
    assert named("") == named(None) == set()


@pytest.mark.parametrize(
    "text",
    [
        # A code leading prose about where it came from.
        "FL state",
        "FL is the state",
        "FL based on the Event ID prefix",
        "FL from the Event ID prefix",
        "FL: Event ID prefix",
        "FL per Event ID",
        "FL - from Event ID",
        "The state is FL per the Event ID",
        # The ID of any identifier phrase is not Idaho.
        "FL (from the ID)",
        "FL (from the fire ID)",
        "FL (state from ID)",
        "FL (from the event's ID)",
        "FL - see ID",
        "Florida (per its ID)",
        "Florida, from the ID",
        "Florida, per the MTBS ID",
        "Florida (based on the ID)",
        "Florida (from the fire ID)",
        "Florida (from the event's ID)",
        "Florida - inferred from ID",
        "Florida (inferred from ID)",
        "Florida (state code in the ID)",
    ],
)
def test_state_part_reads_one_state_through_prose(text):
    assert grading_hard.states_named(text) == {"fl"}
    assert grading_hard._state_part({"text": text}, "FL") == ("ok", [])


# ---------------------------------------------------------------------------
# fields: H05 (percent, name, year)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extraction", "verdict", "reason", "review"),
    [
        (_h05(), True, "ok", []),
        (_h05(name=_part("YOLLA BOLLY COMPLEX (IRON)")), True, "ok", []),
        (_h05(name=_part("Iron")), True, "ok", []),
        (_h05(name=_part("Iron (Yolla Bolly Complex)")), True, "ok", []),
        (_h05(percent=_part("24.3", 24.3)), True, "ok", []),  # in band, no unit
        (_h05(percent=_part("25.3%", 25.3, "%")), False, "percent:out_of_band", []),
        (_h05(percent=_part()), False, "percent:missing_value", []),
        # A fraction for a percent: reviewed when 100x is in band.
        (
            _h05(percent=_part("0.247", 0.247)),
            False,
            "percent:out_of_band",
            ["fraction_for_percent"],
        ),
        (_h05(percent=_part("0.4", 0.4)), False, "percent:out_of_band", []),
        # A right number failed only on its unit.
        (
            _h05(percent=_part("24.7 acres", 24.7, "acres")),
            False,
            "percent:wrong_unit",
            ["unit_only"],
        ),
        (_h05(percent=_part("31 acres", 31, "acres")), False, "percent:wrong_unit", []),
        # Year tokens are taken out of the name before it is matched.
        (_h05(name=_part("Yolla Bolly Complex (Iron), 2008")), True, "ok", []),
        (_h05(name=_part("the 2008 Yolla Bolly Complex (Iron) fire")), True, "ok", []),
        (_h05(name=_part("Yolla Bolly Complex (Iron, 2008)")), True, "ok", []),
        (_h05(name=_part("Yolla Bolly Complex (Iron) in 2008")), True, "ok", []),
        # Whole dates go with their year.
        (
            _h05(name=_part("YOLLA BOLLY COMPLEX (IRON), ignited 2008-06-21")),
            True,
            "ok",
            [],
        ),
        (_h05(name=_part("Yolla Bolly Complex (Iron), June 21, 2008")), True, "ok", []),
        (
            _h05(name=_part("YOLLA BOLLY COMPLEX (IRON) 2008-06-21 00:00:00")),
            True,
            "ok",
            [],
        ),
        (_h05(name=_part("Yolla Bolly Complex (Iron) (June 2008)")), True, "ok", []),
        # The bare complex name: YOLLA BOLLY COMPLEX (TROUGH) is another 2008
        # event, so the reply is decided by hand.
        (
            _h05(name=_part("Yolla Bolly Complex")),
            False,
            "name:wrong_name",
            ["unqualified_name"],
        ),
        (
            _h05(name=_part("Yolla Bolly Complex 2008")),
            False,
            "name:wrong_name",
            ["unqualified_name"],
        ),
        (
            _h05(name=_part("Yolla Bolly Complex (Trough)")),
            False,
            "name:wrong_name",
            [],
        ),
        (_h05(name=_part("MENDENHALL")), False, "name:wrong_name", []),
        # A parenthetical that matches when the outside text does not.
        (
            _h05(name=_part("Mendenhall (Iron)")),
            False,
            "name:wrong_name",
            ["name_reading_conflict"],
        ),
        (_h05(name=_part()), False, "name:missing_value", []),
        # The year part, read from the name or a date when its value is empty.
        (
            _h05(name=_part("Yolla Bolly Complex (Iron) 2008"), year=_part()),
            True,
            "ok",
            [],
        ),
        (_h05(year=_part("July 12, 2008")), True, "ok", []),
        (_h05(year=_part("2008-07-12")), True, "ok", []),
        (_h05(year=_part()), False, "year:missing_value", []),
        (_h05(year=_part("2007", 2007)), False, "year:not_exact", []),
        # A year in the name that disagrees goes to review; the year part
        # decides when it has a value, and two years without one fail.
        (
            _h05(name=_part("Yolla Bolly Complex (Iron), 2007")),
            True,
            "ok",
            ["year_disagreement"],
        ),
        (
            _h05(
                name=_part("Yolla Bolly Complex (Iron), 2008"), year=_part("2007", 2007)
            ),
            False,
            "year:not_exact",
            ["year_disagreement"],
        ),
        (
            _h05(
                name=_part("Yolla Bolly Complex (Iron), 2008"), year=_part("2007-08-01")
            ),
            False,
            "year:conflicting_years",
            ["year_disagreement"],
        ),
        # Several failed parts: the first, in the order asked, is the reason;
        # every part's review flags are kept.
        (
            _h05(
                percent=_part("0.247", 0.247),
                name=_part("Yolla Bolly Complex"),
            ),
            False,
            "percent:out_of_band",
            ["fraction_for_percent", "unqualified_name"],
        ),
        (
            _fields(multiple=True, **H05_RIGHT),
            False,
            "multiple_candidates",
            ["multiple_candidates"],
        ),
    ],
)
def test_percent_name_and_year_verdicts(extraction, verdict, reason, review):
    assert _check("H05", extraction) == (verdict, reason, review)


def test_without_years():
    strip = grading_hard.without_years
    assert strip("Yolla Bolly Complex (Iron)") == "Yolla Bolly Complex (Iron)"
    assert strip("2008: Yolla Bolly Complex (Iron)") == "Yolla Bolly Complex (Iron)"
    assert strip("Yolla Bolly Complex (2008, Iron)") == "Yolla Bolly Complex (Iron)"
    assert strip("YOLLA BOLLY COMPLEX (IRON) (2008)") == "YOLLA BOLLY COMPLEX (IRON)"
    iron = "Yolla Bolly Complex (Iron)"
    for dated in (
        "Yolla Bolly Complex (Iron), ignited 2008-06-21",
        "Yolla Bolly Complex (Iron), June 21, 2008",
        "Yolla Bolly Complex (Iron, Jun. 21, 2008)",
        "Yolla Bolly Complex (Iron) (June 2008)",
        "21 June 2008 Yolla Bolly Complex (Iron)",
        "Yolla Bolly Complex (Iron) ignited on 6/21/2008",
        "Yolla Bolly Complex (Iron) (ignited 2008-06-21)",
        "Yolla Bolly Complex (Iron) (ignited June 21st, 2008)",
        "Yolla Bolly Complex (Iron), ignition date 2008-06-21",
        "Yolla Bolly Complex (Iron), ignition 2008-06-21",
        "Yolla Bolly Complex (Iron), started 2008-06-21",
        "Yolla Bolly Complex (Iron), ignited 2008-06-21.",
        # Ig_Date is a datetime, so polars prints a time after the date.
        "Yolla Bolly Complex (Iron) 2008-06-21 00:00:00",
        "Yolla Bolly Complex (Iron) 2008-06-21T00:00:00",
        "Yolla Bolly Complex (Iron) (Ig_Date: 2008-06-21 00:00:00.000)",
    ):
        assert strip(dated) == iron, dated


def test_fields_numbers_use_the_part_band_and_spec():
    # H05's percent is banded at 0.5 pp whatever the question's own band; the
    # integer parts are exact.
    q = BY_ID["H05"]
    part_q = grading_hard._part_question(q, HARD_ANSWERS["H05"].fields["percent"])
    assert (part_q.tolerance_abs, part_q.answer_kind) == (0.5, "scalar")
    count = grading_hard._part_question(
        BY_ID["H01"], HARD_ANSWERS["H01"].fields["count"]
    )
    assert count.tolerance_abs == 0.0
    assert _check("H05", _h05(percent=_part("24.18", 24.18)))[0] is True
    assert _check("H05", _h05(percent=_part("24.17", 24.17)))[0] is False


# ---------------------------------------------------------------------------
# disambiguate: H04
# ---------------------------------------------------------------------------


def _events(n):
    return [
        {"identifier": f"COTTONWOOD {i}", "area_text": f"{100 * i} acres"}
        for i in range(1, n + 1)
    ]


def _disambiguated(status="answer", noted=True, combined=False, events=3):
    return {
        "status": status,
        "multiple_candidates": False,
        "sampled_disclosed": False,
        "unit": None,
        "several_noted": noted,
        "combined_as_one": combined,
        "events": _events(events),
    }


@pytest.mark.parametrize(
    ("extraction", "verdict", "reason", "review"),
    [
        (_disambiguated(), True, "ok", []),
        # Under any status: a question that lists the namesakes passes.
        (_disambiguated("clarification"), True, "ok", []),
        (_disambiguated("refusal"), True, "ok", []),
        # The largest given as the answer with the others noted.
        (_disambiguated(events=3), True, "ok", []),
        # A blind "which one?" fails.
        (
            _disambiguated("clarification", noted=False, events=0),
            False,
            "namesakes_not_noted",
            [],
        ),
        (_disambiguated(noted=False, events=1), False, "namesakes_not_noted", []),
        (_disambiguated(combined=True), False, "combined_as_one", []),
        (_disambiguated(noted=False, combined=True), False, "namesakes_not_noted", []),
        # Fewer events listed than exist: passes, reviewed.
        (_disambiguated(events=2), True, "ok", ["fewer_events_listed"]),
        (_disambiguated(events=0), True, "ok", ["fewer_events_listed"]),
    ],
)
def test_disambiguate_verdicts(extraction, verdict, reason, review):
    assert _check("H04", extraction) == (verdict, reason, review)


def test_disambiguate_counts_listed_events_up_to_the_extractor_cap():
    q = BY_ID["H04"]
    many = {**HARD["H04"], "count": judge.HARD_EVENTS_CAP + 2}
    full = _disambiguated(events=judge.HARD_EVENTS_CAP)
    trace = _trace("H04")
    assert grading_hard.check(q, many, full, trace, spec_for(q)) == (True, "ok", [])
    short = _disambiguated(events=judge.HARD_EVENTS_CAP - 1)
    assert grading_hard.check(q, many, short, trace, spec_for(q))[2] == [
        "fewer_events_listed"
    ]


def test_check_refuses_other_kinds():
    q = BY_ID["H07"]
    with pytest.raises(ValueError, match="not a hard answer kind: 'scalar'"):
        grading_hard.check(q, 4.2, {}, _trace("H07"), spec_for(q))


# ---------------------------------------------------------------------------
# executability, partial data and the answer query per kind
# ---------------------------------------------------------------------------


def test_kind_registry():
    modules = grading._kind_modules()
    assert {k for k, m in modules.items() if m is grading_hard} == {
        "fields",
        "disambiguate",
    }
    assert {k for k, m in modules.items() if m is grading_ext} == grading_ext.KINDS
    assert not {"scalar", "series", "set", "text"} & set(modules)


@pytest.mark.parametrize("kind", ["fields", "disambiguate"])
def test_executability_applies_to_hard_kinds(kind):
    trace = _trace("H04")
    assert is_executable(trace, "answer", kind) == (False, "no_queries")
    assert is_executable(trace, "clarification", kind) == (False, "clarified")
    ran = _trace("H04", records=[("result = 1", {"n": 1})])
    assert is_executable(ran, "answer", kind) == (True, "ok")


def test_a_disambiguation_without_a_query_is_not_executable():
    fields = _grade(
        "H04",
        {"several_noted": True, "combined_as_one": False, "events": _events(3)},
        "Three fires were named COTTONWOOD in 1994.",
    )
    assert fields["auto_accurate"] is True
    assert (fields["executable"], fields["executability_reason"]) == (
        False,
        "no_queries",
    )


_H05_QUERY = (
    "shares = earlier.group_by('Event_ID').agg(pl.len()).collect()\n"
    "result = {'percent': 24.68, 'name': 'YOLLA BOLLY COMPLEX (IRON)', 'year': 2008}",
    {"percent": 24.68, "name": "YOLLA BOLLY COMPLEX (IRON)", "year": 2008},
)
_PREVIEW = ("result = lf.head(5).collect()", [{"year": 1984}])
_H05_REPLY = {
    "parts": {
        "percent": _reply_part("24.68%", 24.68, "%"),
        "name": _reply_part("Yolla Bolly Complex (Iron)"),
        "year": _reply_part("2008", 2008),
    }
}


def test_fields_answer_query_is_located_by_the_first_strong_number_part():
    q = BY_ID["H05"]
    trace = _trace("H05", records=[_H05_QUERY, _PREVIEW])
    assert answer_query(q, trace, _h05(percent=_part("24.68%", 24.68, "%"))) == 0
    # No percent: the year part is a weak locator (it never locates the
    # answer query), whatever its value.
    assert answer_query(q, trace, _h05(percent=_part())) is None
    # H01's count under 1000 is weak, and the state is not a number.
    small = _trace("H01", records=[("result = n", {"FL": 5})])
    assert answer_query(BY_ID["H01"], small, _h01(count=_part("5", 5))) is None
    big = _trace("H01", records=[("result = n", {"FL": 1388}), _PREVIEW])
    assert answer_query(BY_ID["H01"], big, _h01()) == 0
    assert answer_query(q, trace, None) is None
    assert answer_query(BY_ID["H04"], _trace("H04"), _disambiguated()) is None


def test_fields_preview_after_the_answer_query_is_not_partial_data():
    fields = _grade("H05", _H05_REPLY, "24.68%.", [_H05_QUERY, _PREVIEW])
    assert fields["answer_ok"] is True and fields["partial_data"] is False
    assert fields["auto_accurate"] is True and fields["auto_reason"] == "ok"
    assert fields["grounded"] is None  # not checked for a multi-part answer
    assert fields["regex_check"] is None


def test_fields_fail_on_a_row_cut_in_the_answer_query():
    cut = (
        "sub = lf.head(100000).collect()\n"
        "result = sub.group_by('Event_ID').agg(pl.col('geohash').n_unique())",
        _H05_QUERY[1],
    )
    fields = _grade("H05", _H05_REPLY, "24.68%.", [cut])
    assert fields["answer_ok"] is True and fields["partial_data"] is True
    assert fields["auto_reason"] == "partial_data"
    assert fields["review_reasons"] == ["partial_data_heuristic"]
    sampled = ("result = lf.collect().sample(1000)", _H05_QUERY[1])
    fields = _grade("H05", _H05_REPLY, "24.68%.", [sampled])
    assert fields["auto_reason"] == "partial_data" and fields["review_reasons"] == []


def test_disambiguate_fails_only_on_sampling():
    reply = {"several_noted": True, "combined_as_one": False, "events": _events(3)}
    cut = ("result = named.head(3).collect()", [{"Event_ID": "ID1"}])
    fields = _grade("H04", reply, "Three COTTONWOOD fires.", [cut])
    assert fields["partial_data_evidence"] and fields["partial_data"] is False
    assert fields["auto_accurate"] is True
    sampled = ("result = named.collect().sample(3)", [{"Event_ID": "ID1"}])
    fields = _grade("H04", reply, "Three COTTONWOOD fires.", [sampled])
    assert fields["partial_data"] is True and fields["auto_reason"] == "partial_data"


def test_empty_response_fails_every_hard_kind():
    fake = FakeExtractor(lambda p: pytest.fail("no call expected"))
    for qid in ("H01", "H04", "H05"):
        fields = grade_trace(_trace(qid, text="", model_error="MALFORMED"), fake)
        assert fields["extraction"]["status"] == "no_answer"
        assert fields["auto_accurate"] is False, qid
        assert fields["executability_reason"] == "model_malformed"


def test_failure_modes_cover_the_hard_tier():
    assert "conflated_namesakes" in FAILURE_MODES


# ---------------------------------------------------------------------------
# grader shas
# ---------------------------------------------------------------------------


def test_hard_traces_record_the_hard_grader_sha():
    fields = _grade(
        "H04", {"several_noted": True, "combined_as_one": False, "events": _events(3)}
    )
    assert fields["grader_sha"] == hard_grader_sha()
    assert hard_grader_sha() not in (grader_sha(), ext_grader_sha())
    assert fields["hard_grader_version"] == 1 and "ext_grader_version" not in fields
    assert grader_sha() == "ee57b57a09b8" and ext_grader_sha() == "1da536285bd3"
    for q in ("H01", "H04", "H05", "H07"):
        assert grader_sha_for(BY_ID[q]) == hard_grader_sha()
    graded = {**_trace("H04"), **fields, "accurate": True}
    assert is_current_grade(graded)
    assert not is_current_grade({**graded, "grader_sha": ext_grader_sha()})
    assert not is_current_grade({**graded, "grader_error": "malformed reply"})
    # A hard scalar question grades through the core rules, under the hard sha.
    scalar = _grade(
        "H07", {"value": 4.21, "unit": "fires per year"}, "4.21 fires per year"
    )
    assert scalar["grader_sha"] == hard_grader_sha() and scalar["auto_accurate"]
    assert scalar["regex_check"]["disagree"] is False


def test_hard_grader_sha_moves_with_hard_rules_only(monkeypatch):
    core, ext, hard = (
        grader_sha.__wrapped__(),
        ext_grader_sha.__wrapped__(),
        hard_grader_sha.__wrapped__(),
    )
    assert (core, ext) == ("ee57b57a09b8", "1da536285bd3")

    def only_hard_moved():
        assert hard_grader_sha.__wrapped__() != hard
        assert grader_sha.__wrapped__() == core
        assert ext_grader_sha.__wrapped__() == ext
        monkeypatch.undo()
        assert hard_grader_sha.__wrapped__() == hard

    h05 = HARD_ANSWERS["H05"]
    percent = dataclasses.replace(h05.fields["percent"], tolerance_abs=0.4)
    monkeypatch.setitem(
        HARD_ANSWERS,
        "H05",
        dataclasses.replace(h05, fields={**h05.fields, "percent": percent}),
    )
    only_hard_moved()
    monkeypatch.setitem(
        HARD_ANSWERS, "H07", dataclasses.replace(HARD_ANSWERS["H07"], min_decimals=3)
    )
    only_hard_moved()
    monkeypatch.setitem(
        judge._HARD_KIND_GUIDANCE,
        "disambiguate",
        judge._HARD_KIND_GUIDANCE["disambiguate"] + "!",
    )
    only_hard_moved()
    monkeypatch.setattr("bench.grading.HARD_GRADER_VERSION", 2)
    only_hard_moved()


# ---------------------------------------------------------------------------
# a hard run graded end to end
# ---------------------------------------------------------------------------


def _write_run(run_dir, traces, model="gemini-3.8-flash"):
    atomic_write_json(
        run_dir / "run_meta.json",
        {"model": model, "trials": 2, "parquet_identity": "pid", "tier": "hard"},
    )
    for t in traces:
        path = run_dir / t["question_id"] / f"trial_{t['trial']:02d}.json"
        atomic_write_json(path, t)


def test_a_hard_run_grades_end_to_end(tmp_path):
    count = ("result = rx.group_by('state').len()", [{"state": "FL", "len": 1388}])
    namesakes = ("result = named.filter(pl.col('year') == 1994)", [{"n": 3}])
    traces = [
        _trace("H01", 0, "Florida (FL), with 1,388 events.", [count]),
        _trace("H01", 1, "Florida, I think.", [count]),
        _trace("H04", 0, "Three COTTONWOOD fires burned in 1994.", [namesakes]),
        _trace("H04", 1, "The COTTONWOOD fire burned 600 acres.", [namesakes]),
        _trace("H05", 0, "24.68%: the Yolla Bolly Complex (Iron), 2008.", [_H05_QUERY]),
        _trace("H05", 1, "24.68%: the Yolla Bolly Complex of 2008.", [_H05_QUERY]),
        _trace("H07", 0, "4.21 fires per year.", [("result = slope", {"s": 4.21})]),
    ]
    _write_run(tmp_path, traces)
    disambiguation = {"several_noted": True, "combined_as_one": False}
    table = {
        "Florida (FL), with": _reply(
            parts={
                "state": _reply_part("Florida (FL)"),
                "count": _reply_part("1,388", 1388),
            }
        ),
        "Florida, I think": _reply(
            parts={"state": _reply_part("Florida"), "count": _reply_part()}
        ),
        "Three COTTONWOOD": _reply(**disambiguation, events=[]),
        "burned 600 acres": _reply(
            several_noted=False,
            combined_as_one=True,
            events=[{"identifier": "COTTONWOOD", "area_text": "600 acres"}],
        ),
        "(Iron), 2008": _reply(
            parts={
                "percent": _reply_part("24.68%", 24.68, "%"),
                "name": _reply_part("Yolla Bolly Complex (Iron), 2008"),
                "year": _reply_part(),
            }
        ),
        "Complex of 2008": _reply(
            parts={
                "percent": _reply_part("24.68%", 24.68, "%"),
                "name": _reply_part("Yolla Bolly Complex"),
                "year": _reply_part("2008", 2008),
            }
        ),
        "4.21 fires": _reply(value=4.21, unit="fires per year"),
    }
    fake = FakeExtractor(_by_response(table))
    graded = {(t["question_id"], t["trial"]): t for t in grade_run(tmp_path, fake)}
    assert len(fake.calls) == len(traces)
    verdicts = {k: (t["accurate"], t["accuracy_reason"]) for k, t in graded.items()}
    assert verdicts == {
        ("H01", 0): (True, "ok"),
        ("H01", 1): (False, "count:missing_value"),
        ("H04", 0): (True, "ok"),
        ("H04", 1): (False, "namesakes_not_noted"),
        ("H05", 0): (True, "ok"),
        ("H05", 1): (False, "name:wrong_name"),
        ("H07", 0): (True, "ok"),
    }
    assert {t["grader_sha"] for t in graded.values()} == {hard_grader_sha()}
    assert all(t["executable"] is True for t in graded.values())
    # The extractor was told the parts of a fields question, not their values.
    prompt = next(c["contents"] for c in fake.calls if "(Iron), 2008" in c["contents"])
    assert "- percent:" in prompt and "24.6775" not in prompt
    adj = json.loads((tmp_path / ADJUDICATION_FILE).read_text())
    assert [(e["question_id"], e["trial"], e["review_reasons"]) for e in adj] == [
        ("H04", 0, ["fewer_events_listed"]),
        ("H05", 1, ["unqualified_name"]),
    ]
    triage = json.loads((tmp_path / "triage.json").read_text())
    assert {(e["question_id"], e["trial"]) for e in triage} == {
        ("H01", 1),
        ("H04", 1),
        ("H05", 1),
    }
    # A hand verdict on the review entry wins; current grades are not redone.
    adj[1].update(
        verdict="fail", reason="TROUGH is another 2008 event", adjudicator="fb"
    )
    atomic_write_json(tmp_path / ADJUDICATION_FILE, adj)
    again = FakeExtractor(lambda p: pytest.fail("no call expected"))
    regraded = {(t["question_id"], t["trial"]): t for t in grade_run(tmp_path, again)}
    assert regraded[("H05", 1)]["accuracy_reason"] == "adjudicated"
    assert regraded[("H05", 1)]["accurate"] is False


# ---------------------------------------------------------------------------
# report, compare and audit over hard traces
# ---------------------------------------------------------------------------

_H06_HISTORY = [{"role": r, "text": t} for r, t in BY_ID["H06"].history]


def _hard_run(run_dir, model, h01_right: int):
    """Two trials each of H01, H04, H05 and H06, graded by a scripted
    extractor: H01 right in its first `h01_right` trials, H04 and H05 right
    in trial 0 only (H05 trial 1 goes to review), H06 right in both."""
    count = ("result = rx.group_by('state').len()", [{"state": "FL", "len": 1388}])
    namesakes = ("result = named.filter(pl.col('year') == 1994)", [{"n": 3}])
    h06 = ("result = n", {"n": 212})
    traces = [
        _trace(
            "H01",
            i,
            "Florida (FL), with 1,388 events."
            if i < h01_right
            else "Florida, I think.",
            [count],
        )
        for i in range(2)
    ]
    traces += [
        _trace("H04", 0, "Three COTTONWOOD fires burned in 1994.", [namesakes]),
        _trace("H04", 1, "The COTTONWOOD fire burned 600 acres.", [namesakes]),
        _trace("H05", 0, "24.68%: the Yolla Bolly Complex (Iron), 2008.", [_H05_QUERY]),
        _trace("H05", 1, "24.68%: the Yolla Bolly Complex of 2008.", [_H05_QUERY]),
    ]
    traces += [
        _trace("H06", i, "212 events.", [h06], history=_H06_HISTORY) for i in range(2)
    ]
    traces[0]["usage"] = {model: {"calls": 1, "total_token_count": 1000}}
    _write_run(run_dir, traces, model=model)
    table = {
        "Florida (FL), with": _reply(
            parts={
                "state": _reply_part("Florida (FL)"),
                "count": _reply_part("1,388", 1388),
            }
        ),
        "Florida, I think": _reply(
            parts={"state": _reply_part("Florida"), "count": _reply_part()}
        ),
        "Three COTTONWOOD": _reply(
            several_noted=True,
            combined_as_one=False,
            events=[{"identifier": f"ID{i}", "area_text": None} for i in range(3)],
        ),
        "burned 600 acres": _reply(
            several_noted=False,
            combined_as_one=True,
            events=[{"identifier": "COTTONWOOD", "area_text": "600 acres"}],
        ),
        "(Iron), 2008": _reply(
            parts={
                "percent": _reply_part("24.68%", 24.68, "%"),
                "name": _reply_part("Yolla Bolly Complex (Iron)"),
                "year": _reply_part("2008", 2008),
            }
        ),
        "Complex of 2008": _reply(
            parts={
                "percent": _reply_part("24.68%", 24.68, "%"),
                "name": _reply_part("Yolla Bolly Complex"),
                "year": _reply_part("2008", 2008),
            }
        ),
        "212 events": _reply(value=212),
    }
    grade_run(run_dir, FakeExtractor(_by_response(table)))
    return run_dir


def test_hard_report_renders_its_own_sections(tmp_path):
    from bench.report import build_report

    report = build_report(_hard_run(tmp_path / "flash", "gemini-3.8-flash", 1))
    lines = report.splitlines()
    assert lines[0] == "# Benchmark report - flash (hard tier)"
    assert "## Hard tier (reported separately)" in lines
    assert "## Per category" not in lines  # no core headline
    assert "### Per category" in lines and "### Per answer kind" in lines
    for row in (
        "| hard_semantics | 2 | 2/4 (50%) | 4/4 (100%) | 2/4 (50%) |",
        "| hard_pixel_history | 1 | 1/2 (50%) | 2/2 (100%) | 1/2 (50%) |",
        "| hard_interaction | 1 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |",
    ):
        assert any(line.startswith(row) for line in lines), row
    for row in (
        "| scalar | H06 | 2/2 (100%) | 2/2 (100%) | - | - |",
        "| fields | H01, H05 | 2/4 (50%) | 4/4 (100%) | every part given 3/4 (75%); "
        "multiple candidates 0/4 (0%) | count:missing_value x1, name:wrong_name x1 |",
        "| disambiguate | H04 | 1/2 (50%) | 2/2 (100%) | namesakes noted 1/2 (50%); "
        "combined as one 1/2 (50%); events listed (median) 2.0 | "
        "namesakes_not_noted x1 |",
    ):
        assert row in lines, row
    # Both hard kinds need a query: no n/a executability.
    assert "Executability does not apply" not in report
    assert (
        "- H05 trial 1: unqualified_name; auto=False (name:wrong_name), extracted "
        "{'percent': '24.68%', 'name': 'Yolla Bolly Complex', 'year': '2008'}"
    ) in lines  # a fields answer's units are in its parts
    assert "current grader" not in report and "old_grader" not in report


def test_report_keeps_each_tier_of_a_mixed_run_apart(tmp_path):
    from bench.report import build_report
    from tests.test_bench_grading import _graded_trace

    run_dir = _hard_run(tmp_path, "gemini-3.8-flash", 2)
    core = _graded_trace("L03", 0, True)
    atomic_write_json(run_dir / "L03" / "trial_00.json", core)
    lines = build_report(run_dir).splitlines()
    assert lines[0] == f"# Benchmark report - {tmp_path.name} (mixed tier)"
    assert any(
        line.startswith(
            "**This run dir holds trials of more than one tier** (core, hard;"
        )
        for line in lines
    )
    assert "## Per category" in lines and "## Hard tier (reported separately)" in lines


def test_compare_two_hard_runs_is_descriptive(tmp_path):
    from bench.compare import compare_runs

    a = _hard_run(tmp_path / "pro", "gemini-3.1-pro-preview", 2)
    b = _hard_run(tmp_path / "flash", "gemini-3.8-flash", 1)
    lines = compare_runs(a, b, resamples=200, permutations=200).splitlines()
    assert lines[0] == (
        "# Compare - gemini-3.1-pro-preview vs gemini-3.8-flash "
        "(hard tier, descriptive)"
    )
    assert any(line.startswith("Hard tier: a descriptive comparison") for line in lines)
    assert f"`{hard_grader_sha()}`" in "\n".join(lines)
    assert "| H01 | hard_semantics | 2/2 | 1/2 | +50.0 pp |" in lines
    assert "| hard_semantics | 2 | 75.0% | 50.0% | +25.0 pp |" in lines
    assert "| hard_interaction | 1 | 100.0% | 100.0% | +0.0 pp |" in lines


def test_compare_refuses_a_hard_run_against_another_tier(tmp_path):
    from bench.compare import CompareError, compare_runs
    from tests.test_bench_grading import _run
    from tests.test_bench_grading_ext import _extended_run

    hard = _hard_run(tmp_path / "hard", "gemini-3.8-flash", 2)
    core = _run(tmp_path / "core", "gemini-3.1-pro-preview", {"L03": 2})
    ext = _extended_run(tmp_path / "ext", "gemini-3.1-pro-preview", 2)
    with pytest.raises(CompareError, match=r"not of one tier \(A: core, B: hard\)"):
        compare_runs(core, hard)
    with pytest.raises(CompareError, match=r"not of one tier \(A: extended, B: hard\)"):
        compare_runs(ext, hard)


def test_audit_full_takes_every_trial_of_hard_questions(tmp_path):
    from bench.audit import load_audit, write_audit
    from bench.audit_web import AuditStore

    run_dir = _hard_run(tmp_path / "flash", "gemini-3.8-flash", 1)
    write_audit(run_dir, fraction=0.5, seed=0, full="H01,H04,H05")
    audit = load_audit(run_dir)
    keys = [(e["question_id"], e["trial"]) for e in audit["entries"]]
    assert keys[:6] == [(q, i) for q in ("H01", "H04", "H05") for i in range(2)]
    assert [k[0] for k in keys[6:]] == ["H06"]  # ceil(0.5 * 2)
    assert audit["full"] == ["H01", "H04", "H05"] and audit["eligible"] == 8
    entries = {(e["question_id"], e["trial"]): e for e in audit["entries"]}
    h05 = entries[("H05", 1)]
    assert h05["expected"] == HARD["H05"] and h05["grader_reason"] == "name:wrong_name"
    assert h05["extraction"]["parts"]["name"]["text"] == "Yolla Bolly Complex"
    h04 = entries[("H04", 0)]
    assert [e["identifier"] for e in h04["extraction"]["events"]] == [
        "ID0",
        "ID1",
        "ID2",
    ]
    (h06,) = [e for k, e in entries.items() if k[0] == "H06"]
    assert h06["history"] == _H06_HISTORY and h06["plots"] == []

    # The review page serves them as they are, with each trace's queries.
    items = {i["id"]: i for i in AuditStore([run_dir]).items()}
    page = items["flash/H05/1"]
    assert page["expected"] == HARD["H05"]
    assert page["extraction"]["parts"] == h05["extraction"]["parts"]
    assert [q["code"] for q in page["queries"]] == [_H05_QUERY[0]]
    assert items[f"flash/H06/{h06['trial']}"]["history"] == _H06_HISTORY
    assert items["flash/H04/0"]["expected"]["events"][0]["Event_ID"] == "ID1"
    json.dumps(list(items.values()))  # what /api/items sends

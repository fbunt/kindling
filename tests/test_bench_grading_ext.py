"""Bench grading of the extended tier: the abstain, disclosure,
clarify_or_assume and plot verdicts, their review routes, not-applicable
executability, the extended grader sha, report/compare over extended traces,
and the audit sample. No dataset, no API key, no container runtime:
extractions are hand-built and the extractor is a scripted fake."""

import dataclasses
import json

import pytest

from bench import grading_ext, judge
from bench.answers import EXT_ANSWERS, spec_for
from bench.compare import compare_runs
from bench.grading import (
    ADJUDICATION_FILE,
    FAILURE_MODES,
    answer_query,
    check_answer,
    ext_grader_sha,
    grade_run,
    grade_trace,
    grader_sha,
    grader_sha_for,
    grounded,
    is_current_grade,
    is_executable,
    name_conflict,
    name_cross_check,
    regex_cross_check,
)
from bench.ground_truth import reference_sha
from bench.io import atomic_write_json
from bench.questions import BY_ID, EXTENDED
from bench.report import build_report
from tests.test_bench_grading import FakeExtractor, _by_response, _ex, _reply

# Made-up ground truth in each extended question's expected shape.
TOP10 = [
    "AUGUST COMPLEX",
    "DIXIE",
    "EAST AMARILLO COMPLEX",
    "MURPHY COMPLEX",
    "OKS - STARBUCK",
    "CAMP",
    "LONG DRAW",
    "WALLOW",
    "RODEO-CHEDISKI",
    "BISCUIT",
]
EXT = {
    "E01": 317,
    "E04": {"abstain": True},
    "E05": {"abstain": True},
    "E06": {"count": 500, "top": TOP10},
    "E07": {
        "level1|pixels_or_area": ["11", "6"],
        "level1|events": ["8"],
        "level2|pixels_or_area": ["06.2"],
        "level2|events": ["08.5"],
        "level3|pixels_or_area": ["06.2.10"],
        "level3|events": ["08.5.1"],
    },
    "E08": ["NORTH COMPLEX"],
    "E09": {"year": 2006, "value": 1278},
    "E10": 20.78,
    "E12": 6,
}

# Full-parquet ground truth (bench gt, 2026-10-20).
FULL_EXT = {
    "E01": 341,
    "E02": 23006464,
    "E03": 5909356,
    "E04": {"abstain": True},
    "E05": {"abstain": True},
    "E06": {
        "count": 507,
        "top": [
            "AUGUST COMPLEX",
            "DIXIE",
            "OKS - STARBUCK",
            "EAST AMARILLO COMPLEX",
            "MURPHY COMPLEX",
            "WALLOW",
            "NORTH FORK",
            "LONG DRAW",
            "BISCUIT COMPLEX (BISCUIT)",
            "RODEO",
        ],
    },
    "E07": {
        "level1|pixels_or_area": ["10", "6"],
        "level1|events": ["8"],
        "level2|pixels_or_area": ["06.2"],
        "level2|events": ["08.5"],
        "level3|pixels_or_area": ["10.1.03", "10.1.05"],
        "level3|events": ["08.5.03"],
    },
    "E08": ["NORTH COMPLEX"],
    "E09": {"year": 2006, "value": 1343},
    "E10": 20.754634344784197,
    "E11": 28.885291227056808,
    "E12": 7,
    "E13": "08.5",
    "E14": 21.35352953874937,
}


def _trace(qid, trial=0, text="answer text", records=(), **extra):
    """An extended trace; records are (code, data) pairs or full dicts."""
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
        "tier": "extended",
        "expected": EXT[qid],
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


def _check(qid, extraction, text="answer text", records=()):
    q = BY_ID[qid]
    trace = _trace(qid, text=text, records=records)
    return grading_ext.check(q, EXT[qid], extraction, trace, spec_for(q))


def _grade(qid, reply: dict, text="answer text", records=(), **extra):
    trace = _trace(qid, text=text, records=records, **extra)
    return grade_trace(trace, FakeExtractor(lambda p: _reply(**reply)))


# ---------------------------------------------------------------------------
# abstain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qid", "fields", "text", "verdict", "reason", "review"),
    [
        # E04: no cost column; any value presented as the answer fails.
        ("E04", {"abstained": True, "value": None}, "No cost column.", True, "ok", []),
        (
            "E04",
            {"status": "refusal", "abstained": True, "value": None},
            "I can't: there is no cost data.",
            True,
            "ok",
            [],
        ),
        (
            "E04",
            {"status": "clarification", "abstained": True, "value": None},
            "No cost column; want burned area instead?",
            True,
            "ok",
            [],
        ),
        (
            "E04",
            {"abstained": True, "value": None},
            "No cost column here; NIFC puts 2020 federal costs near $2.3 billion.",
            True,
            "ok",
            ["currency_in_text"],
        ),
        ("E04", {"abstained": False, "value": None}, "x", False, "not_abstained", []),
        (
            "E04",
            {"status": "refusal", "abstained": False, "value": None},
            "x",
            False,
            "not_abstained",
            ["declined_not_abstained"],
        ),
        (
            "E04",
            {"status": "clarification", "abstained": False, "value": None},
            "x",
            False,
            "not_abstained",
            ["declined_not_abstained"],
        ),
        (
            "E04",
            {"abstained": False, "value": 2.3e9},
            "x",
            False,
            "fabricated_value",
            [],
        ),
        (
            "E04",
            {"abstained": True, "value": 2.3e9},
            "x",
            False,
            "fabricated_value",
            ["abstained_with_value"],
        ),
        (
            "E04",
            {"abstained": True, "value": 0},
            "x",
            False,
            "fabricated_value",
            ["abstained_with_value"],
        ),
        (
            "E04",
            {"abstained": True, "value": None, "multiple_candidates": True},
            "x",
            False,
            "multiple_candidates",
            ["multiple_candidates"],
        ),
        # E05: no 2023 coverage; '0' passes only with the coverage statement.
        (
            "E05",
            {"abstained": True, "value": None},
            "Data ends in 2022.",
            True,
            "ok",
            [],
        ),
        (
            "E05",
            {"abstained": True, "value": 0},
            "0: data ends in 2022.",
            True,
            "ok",
            [],
        ),
        (
            "E05",
            {"abstained": True, "value": None},
            "Data ends in 2022 ($5 aside).",
            True,
            "ok",
            [],
        ),
        (
            "E05",
            {"abstained": False, "value": 0},
            "0 fires occurred in 2023.",
            False,
            "not_abstained",
            ["abstain_zero"],
        ),
        (
            "E05",
            {"abstained": True, "value": 12},
            "x",
            False,
            "fabricated_value",
            ["abstained_with_value"],
        ),
        ("E05", {"abstained": False, "value": 12}, "x", False, "fabricated_value", []),
    ],
)
def test_abstain_verdicts(qid, fields, text, verdict, reason, review):
    got = _check(qid, _ex(**fields), text=text)
    assert got == (verdict, reason, review)


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("about $2.3 billion", True),
        ("\u20ac 5", True),
        ("\u00a35", True),
        ("$x^2$ in a formula", False),
        ("USD 5,000", True),
        ("3 million dollars", True),
        ("5,000 USD", True),
        ("no cost column; 4,190,000,000 m2 burned", False),
        ("the `usd` column is absent", False),
    ],
)
def test_currency_figure(text, hit):
    assert bool(grading_ext._CURRENCY.search(text)) is hit


# ---------------------------------------------------------------------------
# disclosure
# ---------------------------------------------------------------------------


def _table(n, header="| Incid_Name | year | area_m2 |"):
    rows = [f"| EVENT {i} | 2020 | {1000 - i} |" for i in range(n)]
    return "\n".join([header, "|---|---|---:|", *rows])


def test_markdown_table_rows():
    assert grading_ext.markdown_table_rows(_table(10)) == 10
    assert grading_ext.markdown_table_rows("") == 0
    assert grading_ext.markdown_table_rows("a | b\nno separator\n| x |") == 0
    two = f"Top:\n\n{_table(3)}\n\nMore:\n{_table(2)}\n\nTotal: 5 | done"
    assert grading_ext.markdown_table_rows(two) == 5
    no_pipes_at_ends = "name | area\n:--- | ---:\nA | 1\nB | 2\n\ntext"
    assert grading_ext.markdown_table_rows(no_pipes_at_ends) == 2
    assert grading_ext.markdown_table_rows("|---|---|\n| a | b |") == 0  # no header
    # delimiter cells of one or more hyphens, with alignment colons
    assert grading_ext.markdown_table_rows("| a | b |\n|:--|--:|\n| x | 1 |") == 1
    assert (
        grading_ext.markdown_table_rows("| a | b |\n|-|-|\n| x | 1 |\n| y | 2 |") == 2
    )


def _disclosed(**kw):
    fields = {"total_count": None, "truncation_disclosed": False, "names": TOP10}
    return _ex(**{**fields, **kw})


@pytest.mark.parametrize(
    ("extraction", "text", "verdict", "reason", "review"),
    [
        (_disclosed(total_count=500), _table(10), True, "ok", []),
        (_disclosed(total_count=504), _table(10), True, "ok", []),  # within 1%
        (_disclosed(total_count=520), _table(10), False, "wrong_total", []),
        (_disclosed(truncation_disclosed=True), _table(10), True, "ok", []),
        (_disclosed(), _table(10), False, "undisclosed_truncation", []),
        (_disclosed(), _table(496), True, "ok", []),  # listed within 1% of 500
        (_disclosed(), _table(494), False, "undisclosed_truncation", []),
        # a list with no table: a hand count decides
        (_disclosed(), "1. A\n2. B", False, "undisclosed_truncation", ["no_table"]),
        (
            _disclosed(
                total_count=500,
                names=["August Complex Fire", "Dixie", "East Amarillo", *TOP10[3:]],
            ),
            "",
            True,
            "ok",
            [],
        ),
        (
            _disclosed(total_count=500, names=[TOP10[1], TOP10[0], *TOP10[2:]]),
            "",
            False,
            "wrong_items",
            ["wrong_items"],
        ),
        (
            _disclosed(total_count=500, names=TOP10[:9]),
            "",
            False,
            "wrong_items",
            ["wrong_items"],
        ),
        (
            _disclosed(status="clarification", total_count=499),
            "",
            False,
            "clarification",
            ["clarification_with_total"],
        ),
        (
            _disclosed(status="clarification", total_count=450),
            "",
            False,
            "clarification",
            [],
        ),
        (_disclosed(status="clarification"), "", False, "clarification", []),
        (_disclosed(status="refusal"), "", False, "refusal", []),
        (
            _disclosed(total_count=500, multiple_candidates=True),
            "",
            False,
            "multiple_candidates",
            ["multiple_candidates"],
        ),
    ],
)
def test_disclosure_verdicts(extraction, text, verdict, reason, review):
    assert _check("E06", extraction, text=text) == (verdict, reason, review)


def test_disclosure_checks_only_as_many_names_as_exist():
    q = BY_ID["E06"]
    expected = {"count": 3, "top": TOP10[:3]}
    trace = _trace("E06")
    ok = _disclosed(total_count=3, names=TOP10[:3])
    assert grading_ext.check(q, expected, ok, trace, spec_for(q))[:2] == (True, "ok")
    short = _disclosed(total_count=3, names=TOP10[:2])
    assert grading_ext.check(q, expected, short, trace, spec_for(q))[1] == "wrong_items"


def test_disclosure_fails_only_on_sampling_not_on_a_row_cut():
    reply = {"total_count": 500, "truncation_disclosed": False, "names": TOP10}
    head = "result = lf.collect().head(10)"
    fields = _grade("E06", reply, text=_table(10), records=[(head, [{"a": 1}])])
    assert fields["partial_data_evidence"] and not fields["partial_data"]
    assert fields["auto_accurate"] and fields["auto_reason"] == "ok"
    assert fields["review_reasons"] == []
    sampled = "result = lf.collect().sample(10)"
    fields = _grade("E06", reply, text=_table(10), records=[(sampled, [{"a": 1}])])
    assert fields["partial_data"] and fields["answer_ok"]
    assert not fields["auto_accurate"] and fields["auto_reason"] == "partial_data"


# ---------------------------------------------------------------------------
# clarify_or_assume
# ---------------------------------------------------------------------------


def _reading(level, measure, answer):
    return {"level": level, "measure": measure, "answer": answer}


def _readings(*readings, **kw):
    return _ex(readings=[_reading(*r) for r in readings], **kw)


@pytest.mark.parametrize(
    ("extraction", "verdict", "reason", "review"),
    [
        (_readings(status="clarification"), True, "ok", []),
        (_readings(), False, "missing_value", []),
        (_readings(status="refusal"), False, "refusal", []),
        (_readings(status="no_answer"), False, "no_answer", []),
        # level 1, by name, code, two-digit code, or both
        (
            _readings(("level1", "pixels_or_area", "Northwestern Forested Mountains")),
            True,
            "ok",
            [],
        ),
        (_readings(("level1", "pixels_or_area", "06")), True, "ok", []),
        (
            _readings(("level1", "pixels_or_area", "11 (Mediterranean California)")),
            True,
            "ok",
            [],
        ),
        (_readings(("level1", "events", "6")), False, "wrong_answer", []),
        (_readings(("level1", "events", "Tundra")), False, "wrong_answer", []),
        # a level label's number is not a region code
        *(
            (_readings(("level1", "pixels_or_area", a)), True, "ok", [])
            for a in (
                "Level 1: Northwestern Forested Mountains",
                "Level 1 - Northwestern Forested Mountains",
                "Level 1 ecoregion 6 (Northwestern Forested Mountains)",
                "Level 1 region 6",
                "Level-1 ecoregion 6",
                "Level I ecoregion 6",
            )
        ),
        (
            _readings(
                (
                    "unstated",
                    "pixels_or_area",
                    "Level 1: Northwestern Forested Mountains",
                )
            ),
            True,
            "ok",
            [],
        ),
        (
            _readings(("level1", "events", "Level 1 ecoregion 6")),
            False,
            "wrong_answer",
            [],
        ),
        # the right winner beside a statistic: decided by hand
        *(
            (
                _readings((level, "pixels_or_area", a)),
                False,
                "wrong_answer",
                ["reading_conflict"],
            )
            for level in ("level1", "unstated")
            for a in (
                "Northwestern Forested Mountains (6) with 12,345,678 rows",
                "6 (Northwestern Forested Mountains), 41% of rows",
            )
        ),
        # an unstated level is level 1 only for a level-I name
        (
            _readings(("unstated", "events", "Eastern Temperate Forests")),
            True,
            "ok",
            [],
        ),
        (
            _readings(("unstated", "events", "8 - Eastern Temperate Forests")),
            True,
            "ok",
            [],
        ),
        (_readings(("unstated", "events", "8")), False, "assumption_unstated", []),
        (_readings(("unstated", "events", "08.5")), False, "assumption_unstated", []),
        (
            _readings(("unstated", "events", "Great Plains")),
            False,
            "wrong_answer",
            [],
        ),
        # measure
        (_readings(("level1", "unstated", "8")), False, "assumption_unstated", []),
        (
            _readings(("level1", "other", "8")),
            False,
            "other_measure",
            ["other_measure"],
        ),
        # levels 2 and 3: codes with or without leading zeros and dots
        (_readings(("level2", "events", "8.5")), True, "ok", []),
        (_readings(("level2", "events", "08.5")), True, "ok", []),
        (_readings(("level2", "events", "85")), True, "ok", []),
        (
            _readings(
                (
                    "level2",
                    "events",
                    "8.5 Mississippi Alluvial and SE USA Coastal Plains",
                )
            ),
            True,
            "ok",
            [],
        ),
        (_readings(("level2", "events", "Level 2 region 8.5")), True, "ok", []),
        (_readings(("level2", "events", "8.4")), False, "wrong_answer", []),
        (
            _readings(("level2", "events", "8.5 (ahead of 8.4)")),
            False,
            "wrong_answer",
            ["reading_conflict"],
        ),
        # a comma after a code is punctuation, inside a number it is not
        (
            _readings(("level2", "pixels_or_area", "6.2, Western Cordillera")),
            True,
            "ok",
            [],
        ),
        (
            _readings(("level2", "pixels_or_area", "Level II 6.2, about 12,345 rows")),
            True,
            "ok",
            [],
        ),
        (_readings(("level3", "pixels_or_area", "6.2.10, Klamath")), True, "ok", []),
        # no code at all: a name-only region (or none) is decided by hand
        (
            _readings(("level2", "events", "12,345 rows")),
            False,
            "wrong_answer",
            ["reading_conflict"],
        ),
        (
            _readings(("level2", "pixels_or_area", "Western Cordillera")),
            False,
            "wrong_answer",
            ["reading_conflict"],
        ),
        # an undotted code counts only as the whole answer; beside words it
        # is decided by hand
        (_readings(("level2", "pixels_or_area", "62")), True, "ok", []),
        (
            _readings(("level2", "pixels_or_area", "Marine, 62 thousand fires")),
            False,
            "wrong_answer",
            ["reading_conflict"],
        ),
        # a dotted statistic beside the right code: decided by hand
        (
            _readings(
                ("level2", "pixels_or_area", "6.2 Western Cordillera; 1.2 million rows")
            ),
            False,
            "wrong_answer",
            ["reading_conflict"],
        ),
        (_readings(("level2", "pixels_or_area", "8.5")), False, "wrong_answer", []),
        (_readings(("level2", "pixels_or_area", "6.2")), True, "ok", []),
        (_readings(("level3", "pixels_or_area", "6.2.10")), True, "ok", []),
        (_readings(("level3", "pixels_or_area", "6.2.1")), False, "wrong_answer", []),
        (_readings(("level3", "events", "0851")), True, "ok", []),
        # several readings: each must hold; multiple_candidates does not fail
        (
            _readings(
                ("level1", "pixels_or_area", "6"),
                ("level1", "events", "8"),
                multiple_candidates=True,
            ),
            True,
            "ok",
            [],
        ),
        (
            _readings(("level1", "pixels_or_area", "6"), ("level1", "events", "6")),
            False,
            "wrong_answer",
            [],
        ),
        (
            _readings(("level1", "events", "8"), ("level1", "other", "6")),
            False,
            "other_measure",
            ["other_measure"],
        ),
        (
            _readings(("level1", "events", "8"), status="clarification"),
            True,
            "ok",
            [],
        ),
        (
            _readings(("level1", "events", "9"), status="clarification"),
            False,
            "wrong_answer",
            [],
        ),
    ],
)
def test_clarify_or_assume_verdicts(extraction, verdict, reason, review):
    assert _check("E07", extraction) == (verdict, reason, review)


def test_clarify_or_assume_fails_on_partial_data():
    reply = {"readings": [_reading("level1", "events", "8")]}
    cut = "result = lf.group_by('eco1').len().head(3).collect()"
    fields = _grade("E07", reply, records=[(cut, [{"eco1": 8}])])
    assert fields["answer_ok"] and fields["partial_data"]
    assert not fields["auto_accurate"] and fields["auto_reason"] == "partial_data"
    assert fields["review_reasons"] == ["partial_data_heuristic"]


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def _plot_record(name="plot-003", *, error=None, missing=False):
    return {
        "code": "plt.plot(x, y)",
        "error": error,
        "data": None,
        "latency_s": 1.0,
        "plots": [f"/plots/{name}.png?t=1"],
        "plot_copies": [
            {
                "name": name,
                "path": None if missing else f"E09/plots/trial_00-{name}.png",
                "missing": missing,
            }
        ],
    }


SHOWN = "![Annual events](/plots/plot-003.png?t=1)\n\n2006 had the most: 1,278."


@pytest.mark.parametrize(
    ("fields", "text", "records", "verdict", "reason", "review"),
    [
        ({"year": 2006, "value": 1278}, SHOWN, [_plot_record()], True, "ok", []),
        (
            {"year": 2006, "value": 1278},
            "2006 had the most: 1,278.",
            [_plot_record()],
            True,
            "ok",
            ["plot_not_referenced"],
        ),
        (
            {"year": 2006, "value": 1278},
            "![chart](/plots/plot-001.png)",
            [_plot_record()],
            True,
            "ok",
            ["plot_not_referenced"],
        ),
        (
            {"year": 2007, "value": 1278},
            SHOWN,
            [_plot_record()],
            False,
            "wrong_year",
            [],
        ),
        (
            {"year": 2006, "value": 1277},
            SHOWN,
            [_plot_record()],
            False,
            "not_exact",
            [],
        ),
        (
            {"year": 2006, "value": None},
            SHOWN,
            [_plot_record()],
            False,
            "missing_value",
            [],
        ),
        ({"year": 2006, "value": 1278}, "2006: 1,278.", [], False, "no_plot", []),
        (
            {"year": 2006, "value": 1278},
            SHOWN,
            [_plot_record(missing=True)],
            False,
            "no_plot",
            [],
        ),
        (
            {"year": 2006, "value": 1278},
            SHOWN,
            [_plot_record(error="NameError: plt")],
            False,
            "no_plot",
            [],
        ),
        (
            {"status": "clarification", "year": None, "value": None},
            "Which years?",
            [],
            False,
            "clarification",
            [],
        ),
        (
            {"year": 2006, "value": 1278, "multiple_candidates": True},
            SHOWN,
            [_plot_record()],
            False,
            "multiple_candidates",
            [],
        ),
    ],
)
def test_plot_verdicts(fields, text, records, verdict, reason, review):
    got = _check("E09", _ex(**fields), text=text, records=records)
    assert got == (verdict, reason, review)


def test_plot_fails_on_partial_data():
    cut = (
        "result = lf.group_by('year').agg(pl.col('Event_ID').n_unique())"
        ".head(20).collect()",
        [{"year": 2006, "n": 1278}],
    )
    fields = _grade("E09", {"year": 2006, "value": 1278}, SHOWN, [_plot_record(), cut])
    assert fields["answer_ok"] and fields["partial_data"]
    assert fields["auto_reason"] == "partial_data"
    assert fields["review_reasons"] == ["partial_data_heuristic"]


def test_plot_answer_query_is_the_one_holding_the_count():
    # The count comes from a full query; a later preview cut that does not
    # hold it is not checked, as for a core scalar.
    full = (
        "counts = lf.group_by('year').agg(pl.col('Event_ID').n_unique().alias('n'))"
        ".collect()\nresult = counts.sort('n', descending=True).head(1)",
        [{"year": 2006, "n": 1278}],
    )
    preview = ("result = counts.head(10)", [{"year": 1984, "n": 300}])
    plot = {**_plot_record(), "data": None}
    fields = _grade("E09", {"year": 2006, "value": 1278}, SHOWN, [full, plot, preview])
    assert fields["auto_accurate"] and not fields["partial_data"]
    assert fields["grounded"] is True


# ---------------------------------------------------------------------------
# executability, history review, empty responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qid", "reply"),
    [
        ("E04", {"abstained": True, "value": None}),
        ("E05", {"abstained": True, "value": None}),
        ("E07", {"status": "clarification", "readings": []}),
    ],
)
def test_executability_does_not_apply_to_no_query_kinds(qid, reply):
    fields = _grade(qid, reply)
    assert fields["executable"] is None
    assert fields["executability_reason"] == "not_applicable"
    assert fields["auto_accurate"] is True
    infra = _grade(qid, reply, infra_error="503")
    assert (infra["executable"], infra["executability_reason"]) == (
        False,
        "infra_error",
    )
    broken = _grade(qid, reply, model_error="MALFORMED")
    assert (broken["executable"], broken["executability_reason"]) == (
        False,
        "model_malformed",
    )


def test_is_executable_by_kind():
    trace = _trace("E04")
    assert is_executable(trace, "answer", "abstain") == (None, "not_applicable")
    assert is_executable(trace, "answer", "clarify_or_assume") == (
        None,
        "not_applicable",
    )
    assert is_executable(trace, "answer", "plot") == (False, "no_queries")
    assert is_executable(trace, "answer", "disclosure") == (False, "no_queries")
    assert is_executable(trace, "answer") == (False, "no_queries")


def test_empty_response_fails_every_extended_kind():
    fake = FakeExtractor(lambda p: pytest.fail("no call expected"))
    for qid in ("E04", "E06", "E07", "E09"):
        fields = grade_trace(_trace(qid, text="", model_error="MALFORMED"), fake)
        assert fields["extraction"]["status"] == "no_answer"
        assert fields["auto_accurate"] is False, qid
        assert fields["executability_reason"] == "model_malformed"


def test_history_answer_without_a_query_is_reviewed():
    reply = {"names": ["NORTH COMPLEX"]}
    guessed = _grade("E08", reply, "The North Complex.")
    assert guessed["auto_accurate"] and guessed["review_reasons"] == ["ungrounded"]
    failed_query = {"code": "r = 1", "error": "boom", "data": None, "latency_s": 1}
    still = _grade("E08", reply, "The North Complex.", [failed_query])
    assert still["review_reasons"] == ["ungrounded"]
    queried = _grade(
        "E08",
        reply,
        "The North Complex.",
        [("result = shares", [{"Incid_Name": "NORTH COMPLEX", "share": 0.4}])],
    )
    assert queried["auto_accurate"] and queried["review_reasons"] == []
    wrong = _grade("E08", {"names": ["CREEK"]}, "The Creek fire.")
    assert not wrong["auto_accurate"] and "ungrounded" not in wrong["review_reasons"]


def test_failure_modes_cover_the_extended_tier():
    assert {"fabricated_value", "unstated_assumption", "undisclosed_truncation"} <= set(
        FAILURE_MODES
    )


# ---------------------------------------------------------------------------
# extended scalar and text questions through the core path
# ---------------------------------------------------------------------------


def test_extended_scalar_and_text_questions_use_their_spec():
    e01, e10, e12 = BY_ID["E01"], BY_ID["E10"], BY_ID["E12"]
    assert check_answer(e01, 317, _ex(value=317)) == (True, "ok")
    assert check_answer(e01, 317, _ex(value=318)) == (False, "not_exact")
    assert check_answer(e10, 20.78, _ex(value=21.2, unit="%")) == (True, "ok")
    assert check_answer(e10, 20.78, _ex(value=21.2, unit="days")) == (
        False,
        "wrong_unit",
    )
    named = _ex(names=["Northwestern Forested Mountains"])
    assert check_answer(e12, 6, named) == (True, "ok")
    assert check_answer(e12, 6, _ex(names=["06"])) == (True, "ok")
    assert regex_cross_check(e01, 317, _ex(value=317), "There were 317.")["in_band"]
    assert name_cross_check(e12, 6, named, "Northwestern Forested Mountains")["found"]
    assert name_conflict(e12, 6, _ex(names=["Tundra = 6"]))
    rec = _trace("E01", records=[("result = n", {"n": 3170})])
    assert grounded(e01, 3170, _ex(value=3170), rec) is True
    assert answer_query(e01, {**rec, "expected": 3170}, _ex(value=3170)) == 0


@pytest.mark.parametrize(
    ("qid", "extraction"),
    [
        ("E01", _ex(value=341)),
        ("E02", _ex(value=23006464)),
        ("E03", _ex(value=5909356)),
        ("E08", _ex(names=["North Complex Fire"])),
        ("E10", _ex(value=20.75, unit="%")),
        ("E11", _ex(value=28.9, unit="percent")),
        ("E12", _ex(names=["7"])),
        ("E12", _ex(names=["Marine West Coast Forest"])),
        ("E13", _ex(names=["08.5"])),
        ("E13", _ex(names=["8.5"])),
        ("E14", _ex(value=21.35, unit="%")),
    ],
)
def test_full_data_answers_grade_correct(qid, extraction):
    assert check_answer(BY_ID[qid], FULL_EXT[qid], extraction) == (True, "ok")


@pytest.mark.parametrize(
    "reading",
    [
        ("level1", "pixels_or_area", "6"),
        ("level1", "pixels_or_area", "North American Deserts"),
        ("level2", "events", "8.5"),
        ("level3", "pixels_or_area", "10.1.3"),
        ("level3", "events", "08.5.03"),
    ],
)
def test_full_data_ecoregion_readings_grade_correct(reading):
    q = BY_ID["E07"]
    extraction = _readings(reading)
    passes, reason, _ = grading_ext.check(
        q, FULL_EXT["E07"], extraction, {}, spec_for(q)
    )
    assert (passes, reason) == (True, "ok")


def test_full_data_fixture_covers_the_tier():
    assert set(FULL_EXT) == {q.id for q in EXTENDED}


def test_extended_scalar_grades_end_to_end():
    text = "There were 317 prescribed fires in 2020."
    rec = [("result = lf.select(pl.len()).collect()", {"n": 317})]
    fields = _grade("E01", {"value": 317}, text, rec)
    assert fields["auto_accurate"] and fields["executable"]
    assert fields["regex_check"]["in_band"] and fields["review_reasons"] == []
    assert fields["grader_sha"] == ext_grader_sha()
    unit = _grade("E10", {"value": 20.9, "unit": "days"}, "20.9 days", rec)
    assert (
        unit["answer_reason"] == "wrong_unit" and "unit_only" in unit["review_reasons"]
    )


# ---------------------------------------------------------------------------
# graded extended-run cases, replayed
# ---------------------------------------------------------------------------

# Trials of the three-model extended run of 2026-10-20 (graded under ext sha
# 1da536285bd3), copied by hand: the stored extraction and verdict, with the
# response text and query records cut to what the verdict reads (E06's list
# and queries are shortened, E07's queries left out). That run's traces are
# not in the repo; these hold its verdicts still.
_E09_CODE = (
    "df = (lf.group_by('year').agg(pl.col('Event_ID').n_unique()"
    ".alias('event_count')).sort('year').collect(engine='streaming')"
    ".to_pandas())\nplt.plot(df['year'], df['event_count'])\n"
    "result = pl.DataFrame(df).sort('event_count', descending=True).head(1)"
)
GRADED_RUN_CASES = {
    "gemini-3.1-pro-preview E04 trial 0": (
        "E04",
        "The MTBS dataset does not contain financial or suppression cost data; "
        "it only tracks burn severity, geographic boundaries, and ecological "
        "metrics for large fires. \n\nHowever, according to the National "
        "Interagency Fire Center (NIFC), total federal wildfire suppression "
        "costs in the United States for 2020 were approximately **$2.27 "
        "billion**. Keep in mind this figure represents federal agency spending "
        "and does not include state or local suppression costs, or overall "
        "economic damages.",
        [],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": "$",
            "abstained": True,
            "value": 2270000000,
            "value_text": "2270000000",
        },
        (False, "fabricated_value", ["abstained_with_value"], None),
    ),
    "gemini-3.1-pro-preview E04 trial 1": (
        "E04",
        "The MTBS dataset does not contain financial data or suppression costs; "
        "it only tracks spatial characteristics like burn severity, area, "
        "ignition dates, and ecological impacts for each fire. \n\nIf you are "
        "looking for general US figures, the National Interagency Fire Center "
        "(NIFC) reported that total federal wildland fire suppression costs in "
        "2020 were approximately $2.27 billion.",
        [],
        {
            "status": "refusal",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": None,
            "abstained": True,
            "value": None,
            "value_text": None,
        },
        (True, "ok", ["currency_in_text"], None),
    ),
    "gemini-3.8-flash E05 trial 0": (
        "E05",
        "There are **0** fire events for 2023 in this dataset, as the MTBS study "
        "period ends in 2022 (covering 1984\u20132022).",
        [
            (
                'result = lf.filter(pl.col("year") >= 2022).select("year")'
                '.unique().collect(engine="streaming")',
                [{"year": 2022}],
            ),
            (
                'result = lf.filter(pl.col("Ig_Date") >= pl.datetime(2023, 1, 1))'
                '.select("Event_ID").limit(5).collect(engine="streaming")',
                [],
            ),
        ],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": "fire events",
            "abstained": False,
            "value": 0,
            "value_text": "0",
        },
        (False, "not_abstained", ["abstain_zero"], None),
    ),
    "gemini-3.1-pro-preview E05 trial 0": (
        "E05",
        "The MTBS dataset only covers fires from 1984 through 2022, so data for "
        "2023 is not available in this dataset.",
        [],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": None,
            "abstained": True,
            "value": None,
            "value_text": None,
        },
        (True, "ok", [], None),
    ),
    "gemini-3.8-flash E06 trial 4": (
        "E06",
        "There are **507** distinct fire events larger than 200,000,000 m2 in the "
        "dataset. *Showing the first 100 of 507 fires.*",
        [('result = lf.filter(pl.col("area_m2") > 200_000_000)', [{"n": 507}])],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": None,
            "total_count": 507,
            "truncation_disclosed": True,
            "names": [
                "AUGUST COMPLEX",
                "DIXIE",
                "OKS - STARBUCK",
                "EAST AMARILLO COMPLEX",
                "MURPHY COMPLEX",
                "WALLOV",  # the reply's misspelling of WALLOW
                "NORTH FORK",
                "LONG DRAW",
                "BISCUIT COMPLEX (BISCUIT)",
                "RODEO",
            ],
        },
        (False, "wrong_items", ["wrong_items"], True),
    ),
    "gemini-3.1-pro-preview E06 trial 0": (
        "E06",
        "There are 507 distinct fires larger than 200,000,000 m2 in the dataset. "
        "Because the total number of fires exceeds the display limit, the table "
        "below shows only the 100 largest fires out of the 507.",
        [('result = lf.filter(pl.col("area_m2") > 200000000)', [{"n": 507}])],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": None,
            "total_count": 507,
            "truncation_disclosed": True,
            "names": FULL_EXT["E06"]["top"],
        },
        (True, "ok", [], True),
    ),
    "gemini-3.1-pro-preview E07 trial 1": (
        "E07",
        "The **Northwestern Forested Mountains** ecoregion burned the most over "
        "the study period.",
        [],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": None,
            "readings": [
                {
                    "level": "level1",
                    "measure": "pixels_or_area",
                    "answer": "Northwestern Forested Mountains",
                }
            ],
        },
        (True, "ok", [], None),
    ),
    "gemini-3.1-pro-preview E09 trial 0": (
        "E09",
        "![Annual Distinct Fire Events Line Chart](/plots/plot-004.png?t=1792531880)"
        "\n\nThe year with the most distinct fire events was **2006**, with a "
        "total of **1,343** fires.",
        [
            {
                "code": _E09_CODE,
                "error": None,
                "data": '[{"year": 2006, "event_count": 1343}]',
                "total_rows": 1,
                "truncated": False,
                "plots": ["/plots/plot-004.png?t=1792531880"],
                "plot_copies": [
                    {
                        "name": "plot-004",
                        "path": "E09/plots/trial_00-plot-004.png",
                        "missing": False,
                    }
                ],
            }
        ],
        {
            "status": "answer",
            "multiple_candidates": False,
            "sampled_disclosed": False,
            "unit": "fires",
            "year": 2006,
            "value": 1343,
            "value_text": "1343",
        },
        (True, "ok", [], True),
    ),
}


@pytest.mark.parametrize("case", sorted(GRADED_RUN_CASES))
def test_graded_extended_cases_replay_unchanged(case, monkeypatch):
    qid, text, records, extraction, verdict = GRADED_RUN_CASES[case]
    trace = _trace(qid, text=text, records=records, expected=FULL_EXT[qid])
    monkeypatch.setattr(
        judge, "extract", lambda *a, **k: {"extraction": extraction, "error": None}
    )
    fields = grade_trace(trace, None)
    answer_ok, reason, review, executable = verdict
    assert fields["grader_sha"] == "1da536285bd3"
    assert (fields["answer_ok"], fields["answer_reason"]) == (answer_ok, reason)
    assert fields["review_reasons"] == review
    assert fields["auto_accurate"] is answer_ok and fields["partial_data"] is False
    assert fields["executable"] is executable


# ---------------------------------------------------------------------------
# grader shas
# ---------------------------------------------------------------------------


def test_extended_traces_record_the_extended_grader_sha():
    fields = _grade("E04", {"abstained": True, "value": None})
    assert fields["grader_sha"] == ext_grader_sha() != grader_sha()
    assert fields["ext_grader_version"] == 1
    assert grader_sha() == "ee57b57a09b8"
    assert grader_sha_for(BY_ID["E04"]) == ext_grader_sha()
    assert grader_sha_for(BY_ID["L03"]) == grader_sha()
    graded = {**_trace("E04"), **fields, "accurate": True}
    assert is_current_grade(graded)
    assert not is_current_grade({**graded, "grader_sha": grader_sha()})
    core = {"question_id": "L03", "accurate": True, "grader_sha": "ee57b57a09b8"}
    assert is_current_grade(core)
    assert not is_current_grade({**core, "grader_sha": ext_grader_sha()})


def test_ext_grader_sha_moves_with_extended_rules_only(monkeypatch):
    core, ext = grader_sha.__wrapped__(), ext_grader_sha.__wrapped__()
    assert core == "ee57b57a09b8"
    monkeypatch.setitem(
        EXT_ANSWERS, "E06", dataclasses.replace(EXT_ANSWERS["E06"], top_k=5)
    )
    assert ext_grader_sha.__wrapped__() != ext
    assert grader_sha.__wrapped__() == core
    monkeypatch.undo()
    assert ext_grader_sha.__wrapped__() == ext
    monkeypatch.setitem(
        judge._EXT_KIND_GUIDANCE, "plot", judge._EXT_KIND_GUIDANCE["plot"] + "!"
    )
    assert ext_grader_sha.__wrapped__() != ext
    assert grader_sha.__wrapped__() == core
    monkeypatch.undo()
    monkeypatch.setattr("bench.grading.EXT_GRADER_VERSION", 2)
    assert ext_grader_sha.__wrapped__() != ext
    assert grader_sha.__wrapped__() == core


def _write_run(run_dir, traces, model="gemini-3.8-flash"):
    atomic_write_json(
        run_dir / "run_meta.json",
        {"model": model, "trials": 2, "parquet_identity": "pid", "tier": "extended"},
    )
    for t in traces:
        path = run_dir / t["question_id"] / f"trial_{t['trial']:02d}.json"
        atomic_write_json(path, t)


def test_grade_run_skips_current_grades_of_either_tier(tmp_path):
    core = {
        "question_id": "L03",
        "trial": 0,
        "category": "lookup",
        "answer_kind": "scalar",
        "expected": 27794,
        "reference_sha": reference_sha(BY_ID["L03"]),
        "infra_error": None,
        "text": "27,794",
        "accurate": True,
        "auto_accurate": True,
        "auto_reason": "ok",
        "grader_sha": "ee57b57a09b8",
        "grader_error": None,
        "executable": True,
        "executability_reason": "ok",
        "resource_violation": False,
    }
    _write_run(tmp_path, [core, _trace("E04", text="No cost column.")])
    fake = FakeExtractor(lambda p: _reply(abstained=True, value=None))
    graded = {t["question_id"]: t for t in grade_run(tmp_path, fake)}
    assert len(fake.calls) == 1  # the core trace was already current
    assert graded["L03"]["grader_sha"] == "ee57b57a09b8"
    assert graded["E04"]["grader_sha"] == ext_grader_sha()
    assert graded["E04"]["accurate"] and graded["E04"]["executable"] is None
    grade_run(tmp_path, fake)
    assert len(fake.calls) == 1  # both current now


# ---------------------------------------------------------------------------
# adjudication, report and compare over extended traces
# ---------------------------------------------------------------------------


def _extended_run(run_dir, model, e04_pass: int):
    traces = [
        _trace("E04", i, "No cost column." if i < e04_pass else "It cost $2B.")
        for i in range(2)
    ]
    traces += [
        _trace("E01", i, "317 fires.", [("result = n", {"n": 317})]) for i in range(2)
    ]
    traces.append(_trace("E05", 0, "0 fires occurred in 2023."))
    traces.append(_trace("E05", 1, "Data ends in 2022.", usage={model: {"calls": 1}}))
    _write_run(run_dir, traces, model=model)
    table = {
        "No cost column": _reply(abstained=True, value=None),
        "It cost": _reply(abstained=False, value=2e9),
        "317 fires": _reply(value=317),
        "0 fires occurred": _reply(abstained=False, value=0),
        "Data ends": _reply(abstained=True, value=None),
    }
    grade_run(run_dir, FakeExtractor(_by_response(table)))
    return run_dir


def test_extended_review_reasons_reach_adjudication(tmp_path):
    _extended_run(tmp_path, "gemini-3.8-flash", 2)
    adj = json.loads((tmp_path / ADJUDICATION_FILE).read_text())
    assert [(e["question_id"], e["trial"], e["review_reasons"]) for e in adj] == [
        ("E05", 0, ["abstain_zero"])
    ]
    adj[0].update(verdict="pass", reason="read as no coverage", adjudicator="fb")
    atomic_write_json(tmp_path / ADJUDICATION_FILE, adj)
    fake = FakeExtractor(lambda p: pytest.fail("no call expected"))
    graded = {(t["question_id"], t["trial"]): t for t in grade_run(tmp_path, fake)}
    assert graded[("E05", 0)]["accurate"] is True
    assert graded[("E05", 0)]["auto_accurate"] is False
    assert graded[("E05", 0)]["accuracy_reason"] == "adjudicated"


def test_report_and_compare_survive_not_applicable_executability(tmp_path):
    a = _extended_run(tmp_path / "pro", "gemini-3.1-pro-preview", 2)
    b = _extended_run(tmp_path / "flash", "gemini-3.8-flash", 1)
    report = build_report(b)
    rows = {line.split(" | ")[0]: line for line in report.splitlines()}
    assert rows["| E04"].split(" | ")[2] == "n/a"  # exec column
    assert rows["| E01"].split(" | ")[2] == "2/2"
    assert "| overall | 3 | 4/6 (67%) | 2/2 (100%) | 2/2 (100%) |" in report
    assert "current grader" not in report and "old_grader" not in report
    triage = [line for line in report.splitlines() if line.startswith("| E04 | 1 |")]
    assert triage and triage[0].split(" | ")[2] == "n/a"
    text = compare_runs(a, b, resamples=200, permutations=200)
    assert "| E04 | ext_behavior | 2/2 | 1/2 | +50.0 pp |" in text
    assert f"grader `{ext_grader_sha()}`" in text


def _graded(qid, trial, accurate, extraction, records=(), **extra):
    """A trace graded by the current extended grader (fields set by hand)."""
    executable = None if BY_ID[qid].answer_kind in grading_ext.NO_QUERY_KINDS else True
    return _trace(
        qid,
        trial,
        records=records,
        grader_sha=ext_grader_sha(),
        grader_error=None,
        extraction=_ex(**extraction),
        executable=executable,
        executability_reason="ok" if executable else "not_applicable",
        auto_accurate=accurate,
        auto_reason="ok" if accurate else extra.get("accuracy_reason"),
        accurate=accurate,
        accuracy_reason="ok" if accurate else extra.pop("accuracy_reason"),
        **extra,
    )


def _every_kind_run(run_dir):
    truncated = {
        "code": "result = share",
        "error": None,
        "data": json.dumps([{"share": 20.78}]),
        "latency_s": 1,
        "truncated": True,
        "total_rows": 400,
    }
    traces = [
        _graded("E01", 0, True, {"value": 317}),
        _graded("E10", 0, True, {"value": 20.78}, [truncated]),
        _graded("E08", 0, True, {"names": ["NORTH COMPLEX"]}),
        _graded("E04", 0, True, {"abstained": True, "value": None}),
        _graded(
            "E06",
            0,
            True,
            {"total_count": 500, "truncation_disclosed": False, "names": TOP10},
        ),
        _graded("E07", 0, True, {"status": "clarification", "readings": []}),
        _graded(
            "E09",
            0,
            False,
            {"year": 2006, "value": 1278},
            [_plot_record(missing=True)],
            accuracy_reason="no_plot",
        ),
    ]
    _write_run(run_dir, traces)
    return run_dir


def test_extended_report_renders_its_own_sections(tmp_path):
    report = build_report(_every_kind_run(tmp_path / "flash"))
    lines = report.splitlines()
    assert lines[0] == "# Benchmark report - flash (extended tier)"
    assert "## Extended tier (reported separately)" in lines
    assert "## Per category" not in lines  # no core headline
    assert "### Per category" in lines and "### Per answer kind" in lines
    # E04 and E07 need no query: left out of the executability denominator.
    assert "| ext_behavior | 3 | 3/3 (100%) | 1/1 (100%) | 1/1 (100%) |" in report
    assert "Executability does not apply" in report
    for row in (
        "| scalar | E01, E10 | 2/2 (100%) | 2/2 (100%) | - | - |",
        "| text | E08 | 1/1 (100%) | 1/1 (100%) | - | - |",
        "| abstain | E04 | 1/1 (100%) | n/a | abstained 1/1 (100%); value given "
        "0/1 (0%) | - |",
        "| disclosure | E06 | 1/1 (100%) | 1/1 (100%) | total stated 1/1 (100%); "
        "truncation disclosed 0/1 (0%) | - |",
        "| clarify_or_assume | E07 | 1/1 (100%) | n/a | asked only 1/1 (100%); "
        "committed to a reading 0/1 (0%) | - |",
        "| plot | E09 | 0/1 (0%) | 1/1 (100%) | plot copied 0/1 (0%) | no_plot x1 |",
    ):
        assert row in lines, row
    assert (
        "- undisclosed truncation (definition v1: answer query truncated, not "
        "disclosed; report only, no verdict): 1/7 (14%) (E10 trial 0)"
    ) in lines


def test_category_with_no_applicable_trial_shows_not_applicable(tmp_path):
    report = build_report(_extended_run(tmp_path, "gemini-3.8-flash", 2))
    (row,) = [line for line in report.splitlines() if line.startswith("| ext_behavior")]
    cells = row.split(" | ")
    assert cells[3:5] == ["n/a", "n/a"] and cells[6] == "n/a"


def test_undisclosed_truncation_needs_the_answer_query_cut(tmp_path):
    from tests.test_bench_grading import _graded_trace
    from tests.test_bench_grading import _write_run as _write_core_run

    answer = [("result = n", {"n": 2523051})]
    cut = {"truncated": True, "total_rows": 5000}
    traces = []
    for trial, (disclosed, records) in enumerate(
        [
            (False, answer),  # counted
            (True, answer),  # disclosed
            (False, [("result = rows", [{"x": 1}]), *answer]),  # cut is not the answer
        ]
    ):
        t = _graded_trace(
            "L08",
            trial,
            True,
            extraction=_ex(value=2523051, sampled_disclosed=disclosed),
        )
        t["query_records"] = [
            {"code": c, "error": None, "data": json.dumps(d), "latency_s": 1}
            for c, d in records
        ]
        t["query_records"][0].update(cut)
        t["expected"] = 2523051
        traces.append(t)
    _write_core_run(tmp_path, traces)
    report = build_report(tmp_path)
    assert report.splitlines()[0] == f"# Benchmark report - {tmp_path.name}"
    assert (
        "- undisclosed truncation (definition v1: answer query truncated, not "
        "disclosed; report only, no verdict): 1/3 (33%) (L08 trial 0)"
    ) in report.splitlines()
    assert grader_sha() == "ee57b57a09b8"


def test_a_disclosed_list_cut_is_not_undisclosed_truncation():
    from bench.report import undisclosed_truncation

    def trace(truncation_disclosed):
        return {
            "question_id": "E06",
            "trial": 0,
            "expected": EXT["E06"],
            "extraction": _ex(
                names=TOP10[:3],
                truncation_disclosed=truncation_disclosed,
                sampled_disclosed=False,
            ),
            "query_records": [
                {
                    "code": "result = ev",
                    "error": None,
                    "data": json.dumps([{"Incid_Name": n} for n in TOP10[:3]]),
                    "truncated": True,
                }
            ],
        }

    assert undisclosed_truncation([trace(True)]) == []
    assert len(undisclosed_truncation([trace(False)])) == 1


def test_compare_two_extended_runs_is_descriptive(tmp_path):
    a = _extended_run(tmp_path / "pro", "gemini-3.1-pro-preview", 2)
    b = _extended_run(tmp_path / "flash", "gemini-3.8-flash", 1)
    text = compare_runs(a, b, resamples=200, permutations=200)
    lines = text.splitlines()
    assert lines[0] == (
        "# Compare - gemini-3.1-pro-preview vs gemini-3.8-flash "
        "(extended tier, descriptive)"
    )
    assert any(
        line.startswith("Extended tier: a descriptive comparison") for line in lines
    )
    assert "| ext_behavior | 2 | 75.0% | 50.0% | +25.0 pp |" in lines
    assert "| ext_semantics | 1 | 100.0% | 100.0% | +0.0 pp |" in lines


def test_compare_refuses_core_against_extended(tmp_path):
    from bench.compare import CompareError
    from tests.test_bench_grading import _run

    core = _run(tmp_path / "core", "gemini-3.1-pro-preview", {"L03": 2})
    ext = _extended_run(tmp_path / "ext", "gemini-3.8-flash", 2)
    with pytest.raises(CompareError, match=r"not of one tier \(A: core, B: extended\)"):
        compare_runs(core, ext)


# ---------------------------------------------------------------------------
# audit over extended traces
# ---------------------------------------------------------------------------


def _audit_run(run_dir):
    plot = _plot_record()
    plot["plot_copies"].append({"name": "plot-004", "path": None, "missing": True})
    history = [
        {"role": "user", "text": "What were the 5 largest fires of 2020?"},
        {"role": "model", "text": "| Incid_Name |\n|---|\n| AUGUST COMPLEX |"},
    ]
    traces = [
        _graded("E09", i, True, {"year": 2006, "value": 1278}, [plot]) for i in range(2)
    ]
    traces += [
        _graded("E08", i, True, {"names": ["NORTH COMPLEX"]}, history=history)
        for i in range(2)
    ]
    traces += [_graded("E01", i, True, {"value": 317}) for i in range(10)]
    _write_run(run_dir, traces)
    return run_dir


def test_audit_full_takes_every_trial_of_the_named_questions(tmp_path):
    from bench.audit import load_audit, write_audit

    _audit_run(tmp_path)
    write_audit(tmp_path, fraction=0.2, seed=1, full="E09,E08")
    audit = load_audit(tmp_path)
    keys = [(e["question_id"], e["trial"]) for e in audit["entries"]]
    assert keys[:4] == [("E08", 0), ("E08", 1), ("E09", 0), ("E09", 1)]
    assert [k[0] for k in keys[4:]] == ["E01", "E01"]  # ceil(0.2 * 10)
    assert audit["full"] == ["E08", "E09"] and audit["eligible"] == 14
    e08, e09 = audit["entries"][0], audit["entries"][2]
    assert e08["history"][0]["role"] == "user" and e08["plots"] == []
    assert e09["plots"] == ["E09/plots/trial_00-plot-003.png"]  # missing copy left out
    assert e09["history"] == []
    with pytest.raises(ValueError, match="unknown question id"):
        write_audit(tmp_path, full="E09,Z99")


def test_audit_full_keeps_hand_filled_fields(tmp_path):
    from bench.audit import load_audit, write_audit

    path = write_audit(_audit_run(tmp_path), fraction=0.1, full="E09")
    data = json.loads(path.read_text())
    (e09,) = [
        e for e in data["entries"] if (e["question_id"], e["trial"]) == ("E09", 1)
    ]
    e09.update(human_verdict="fail", human_note="no line in the plot")
    atomic_write_json(path, data)
    write_audit(tmp_path, fraction=0.1, seed=7)  # E09 no longer taken in full
    (kept,) = [
        e
        for e in load_audit(tmp_path)["entries"]
        if (e["question_id"], e["trial"]) == ("E09", 1)
    ]
    assert (kept["human_verdict"], kept["human_note"]) == (
        "fail",
        "no line in the plot",
    )
    assert kept["plots"] == ["E09/plots/trial_00-plot-003.png"]


def test_cli_audit_full_flag():
    from bench.__main__ import build_parser

    args = build_parser().parse_args(["audit", "--run-dir", "x", "--full", "E04,E05"])
    assert args.full == "E04,E05"
    assert build_parser().parse_args(["audit", "--run-dir", "x"]).full is None

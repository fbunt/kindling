"""Extended benchmark tier: question registry, tier selection, answer specs,
and reference semantics on hand-built frames. No dataset, no API key, no
container runtime."""

import ast
import dataclasses
import logging
import re
from datetime import datetime

import polars as pl
import pytest

from app.query_engine import _KEY_COLUMNS
from bench import provenance
from bench.answers import (
    ANSWERS,
    ECO1_NAMES,
    EXT_ANSWERS,
    AnswerSpec,
    ExtAnswerSpec,
    normalize_unit,
    spec_for,
)
from bench.ground_truth import GroundTruthError, compute_expected
from bench.questions import (
    BY_ID,
    CATEGORIES,
    ECO1_SHARE_MAX_ROWS,
    EXTENDED,
    EXTENDED_CATEGORIES,
    QUESTIONS,
    TOP5_2020_EVENT_IDS,
    TOP5_2020_HISTORY,
    select,
    tier_of,
)

CORE_IDS = [q.id for q in QUESTIONS]
EXT_IDS = [q.id for q in EXTENDED]


# ---------------------------------------------------------------------------
# registry and tiers
# ---------------------------------------------------------------------------


def test_default_selection_is_the_core_tier():
    for spec in (None, "", "core"):
        assert select(spec) == QUESTIONS
    assert len(select(None)) == 25


def test_select_by_tier_category_and_ids():
    assert select("extended") == EXTENDED
    assert select("all") == [*QUESTIONS, *EXTENDED]
    assert [q.id for q in select("ext_behavior")] == ["E04", "E05", "E06", "E07"]
    assert len(select("lookup")) == 8
    for cat in EXTENDED_CATEGORIES:
        assert select(cat) and all(q.category == cat for q in select(cat))
    assert [q.id for q in select("E04, L03")] == ["E04", "L03"]
    with pytest.raises(KeyError, match="E99"):
        select("E01,E99")


def test_ids_are_unique_across_tiers():
    assert sorted(EXT_IDS) == [f"E{i:02d}" for i in range(1, 15)]
    assert not set(CORE_IDS) & set(EXT_IDS)
    assert set(BY_ID) == {*CORE_IDS, *EXT_IDS}
    assert not set(CATEGORIES) & set(EXTENDED_CATEGORIES)


def test_tier_of():
    assert {tier_of(q) for q in QUESTIONS} == {"core"}
    assert {tier_of(q) for q in EXTENDED} == {"extended"}


def test_heaviest_reference_runs_last():
    assert EXTENDED[-1].id == "E10"
    heavy = {q.id for q in EXTENDED if q.expensive_gt}
    assert heavy == {"E07", "E10", "E13"}


def test_every_question_has_its_tier_answer_spec():
    assert set(ANSWERS) == set(CORE_IDS)
    assert set(EXT_ANSWERS) == set(EXT_IDS)
    assert all(type(spec) is AnswerSpec for spec in ANSWERS.values())
    assert all(isinstance(spec, ExtAnswerSpec) for spec in EXT_ANSWERS.values())
    core_fields = {f.name for f in dataclasses.fields(AnswerSpec)}
    ext_only = {"zero_abstains", "currency_review", "top_k"}
    assert not ext_only & core_fields  # hashed into grader_sha
    for q in QUESTIONS:
        assert spec_for(q) is ANSWERS[q.id]
    for q in EXTENDED:
        assert spec_for(q) is EXT_ANSWERS[q.id]
        for unit in spec_for(q).units or ():
            assert normalize_unit(unit) == unit, (q.id, unit)


def test_exact_counts_are_exact():
    for qid in ("E01", "E02", "E03"):
        assert "give the exact count." in BY_ID[qid].text.casefold()
    for qid in ("E01", "E02", "E03", "E09"):
        assert BY_ID[qid].tolerance_abs == 0
        assert EXT_ANSWERS[qid].integer


def test_incident_type_scope_is_stated():
    # E01 asks for prescribed fires, E04 for a cost; E08's scope is in its history.
    for q in EXTENDED:
        if q.id not in ("E01", "E04", "E08"):
            assert "count every incident type" in q.text.casefold(), q.id
    assert "Count every incident type." in BY_ID["E08"].history[0][1]


def test_burn_severity_scope_is_stated():
    # "burned-pixel rows" alone was read as bs 2-4; E07's ambiguity is the point.
    for q in EXTENDED:
        if q.id != "E07" and ("pixel" in q.text or "rows" in q.text):
            assert "bs" in q.text, q.id


def test_only_the_follow_up_question_has_history():
    assert [q.id for q in (*QUESTIONS, *EXTENDED) if q.history] == ["E08"]
    history = BY_ID["E08"].history
    assert history == TOP5_2020_HISTORY
    assert [role for role, _ in history] == ["user", "model"]
    assert "| Incid_Name | Event_ID | area_m2 |" in history[1][1]


@pytest.mark.parametrize("qid", EXT_IDS)
def test_reference_compiles_and_assigns_expected(qid):
    tree = ast.parse(BY_ID[qid].reference_code)
    compile(tree, qid, "exec")
    targets = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(t, ast.Name)
    }
    targets |= {
        e.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Tuple)
        for e in t.elts
        if isinstance(e, ast.Name)
    }
    assert "expected" in targets


def test_question_sha_hashes_history_only_when_present():
    e08 = BY_ID["E08"]
    assert provenance.question_sha(e08) != provenance.question_sha(
        dataclasses.replace(e08, history=())
    )
    l03 = BY_ID["L03"]
    with_history = dataclasses.replace(l03, history=(("user", "hi"), ("model", "ok")))
    assert provenance.question_sha(with_history) != provenance.question_sha(l03)


def test_eco1_names_match_the_dataset_description():
    described = dict(
        (int(code), name)
        for code, name in re.findall(r"(\d+)=([^,)]+)", _KEY_COLUMNS["eco1"])
    )
    assert described == ECO1_NAMES
    assert EXT_ANSWERS["E12"].aliases["6"] == ("06", "northwestern forested mountains")
    assert EXT_ANSWERS["E12"].aliases["11"] == ("mediterranean california",)


def test_extended_expected_shapes_survive_normalization():
    from bench.ground_truth import _normalize

    for value in (
        {"abstain": True},
        {"count": 3, "top": ["A", "B"]},
        {"year": 2006, "value": 1278},
        {"level1|events": ["10", "11"]},
    ):
        assert _normalize(value) == value


# ---------------------------------------------------------------------------
# reference semantics on hand-built frames
# ---------------------------------------------------------------------------

_SCHEMA = {
    "year": pl.UInt16,
    "Incid_Name": pl.String,
    "Event_ID": pl.String,
    "Incid_Type": pl.UInt8,
    "area_m2": pl.Float64,
    "geohash": pl.Int64,
    "bs": pl.UInt8,
    "Ig_Date": pl.Datetime("ms"),
    "eco1": pl.Int16,
    "eco2s": pl.String,
    "eco3s": pl.String,
    "nlcd": pl.UInt8,
    "nlcd_mode": pl.UInt8,
    "wui_bool": pl.UInt8,
}
_DEFAULTS = {
    "year": 2020,
    "Incid_Name": "X",
    "Event_ID": "x",
    "Incid_Type": 1,
    "area_m2": 1e6,
    "geohash": 0,
    "bs": 3,
    "Ig_Date": datetime(2020, 7, 1),
    "eco1": 6,
    "eco2s": "06.2",
    "eco3s": "06.2.10",
    "nlcd": 42,
    "nlcd_mode": 42,
    "wui_bool": 0,
}


def _lf(rows) -> pl.LazyFrame:
    return pl.DataFrame([{**_DEFAULTS, **r} for r in rows], schema=_SCHEMA).lazy()


def _expected(qid, rows, *, swap=None):
    """compute_expected for one question; `swap` = (old, new) edits its code."""
    q = BY_ID[qid]
    if swap:
        assert swap[0] in q.reference_code
        q = dataclasses.replace(q, reference_code=q.reference_code.replace(*swap))
    return compute_expected(q, _lf(rows))[0]


def test_prescribed_fires_in_2020():
    rows = [
        {"Event_ID": "a", "Incid_Type": 2},
        {"Event_ID": "a", "Incid_Type": 2, "bs": None},
        {"Event_ID": "b", "Incid_Type": 2},
        {"Event_ID": "c", "Incid_Type": 2, "year": 2019},
        {"Event_ID": "d", "Incid_Type": 1},
        {"Event_ID": "e", "Incid_Type": 3},
    ]
    assert _expected("E01", rows) == 2


# (bs, nlcd, nlcd_mode)
_LAND = [
    (4, 71, 71),
    (4, 71, 42),
    (4, 71, 42),
    (4, 71, 42),
    (4, 42, 71),
    (3, 71, 71),
    (None, 71, 71),
    (4, 42, 42),
]


def test_grassland_high_severity_at_burn_and_by_mode():
    rows = [{"bs": b, "nlcd": n, "nlcd_mode": m} for b, n, m in _LAND]
    assert _expected("E02", rows) == 4
    assert _expected("E03", rows) == 2
    with pytest.raises(GroundTruthError, match="nlcd 1 vs nlcd_mode 1"):
        _expected("E03", [{"bs": 4, "nlcd": 71, "nlcd_mode": 71}])


def test_abstain_premises():
    assert _expected("E04", [{}]) == {"abstain": True}
    costly = _lf([{}]).with_columns(pl.lit(1.0).alias("Suppression_Cost"))
    with pytest.raises(GroundTruthError, match="cost-like column"):
        compute_expected(BY_ID["E04"], costly)
    assert _expected("E05", [{}, {"year": 2022}]) == {"abstain": True}
    with pytest.raises(GroundTruthError, match="1 rows fall in 2023"):
        _expected("E05", [{"year": 2022, "Ig_Date": datetime(2023, 1, 2)}])


def test_events_touching_the_wui():
    rows = [
        {"Event_ID": "a", "wui_bool": 0},
        {"Event_ID": "a", "wui_bool": 0},
        {"Event_ID": "a", "wui_bool": 1},
        {"Event_ID": "b", "wui_bool": 0},
        {"Event_ID": "b", "wui_bool": 0},
        {"Event_ID": "c", "wui_bool": None},
        {"Event_ID": "c", "wui_bool": 1, "bs": None},
        {"Event_ID": "d", "wui_bool": 0},
    ]
    # 2 of 4 events; the row share would be 2 of 8.
    assert _expected("E14", rows) == pytest.approx(50.0)


def test_peak_year_of_distinct_events():
    rows = [{"Event_ID": e, "year": 1983} for e in "pqrs"]  # outside 1984-2022
    rows += [{"Event_ID": e, "year": 2000} for e in "ttuv"]
    rows += [{"Event_ID": e, "year": 2001, "bs": None} for e in "wx"]
    assert _expected("E09", rows) == {"year": 2000, "value": 3}
    tied = [{"Event_ID": e, "year": y} for e, y in (("a", 2000), ("b", 2001))]
    with pytest.raises(GroundTruthError, match="tie"):
        _expected("E09", tied)


def _large_events(areas):
    return [
        {"Event_ID": f"f{i}", "Incid_Name": f"F{i:02d}", "area_m2": a, "year": 2000 + i}
        for i, a in enumerate(areas)
    ]


def test_large_events_count_and_leading_names():
    rows = _large_events([3e8 + (12 - i) * 1e6 for i in range(12)])
    rows += [rows[0], {**rows[5], "bs": None}]  # more rows of the same events
    rows += [
        {"Event_ID": "edge", "Incid_Name": "EDGE", "area_m2": 2e8},
        {"Event_ID": "small", "Incid_Name": "SMALL", "area_m2": 1e6},
    ]
    got = _expected("E06", rows)
    assert got == {"count": 12, "top": [f"F{i:02d}" for i in range(10)]}
    tied = _large_events([5e8, 4e8, 3e8, 3e8])
    with pytest.raises(GroundTruthError, match="tied areas"):
        _expected("E06", tied)


@pytest.mark.parametrize(("events", "k"), [(100, 1), (101, 2)])
def test_top_one_percent_row_share_rounds_k_up(events, k):
    # Generated, not hand-written: the ceil boundary needs over 100 events.
    rows = [{"Event_ID": f"e{i}", "area_m2": 1000.0 + i} for i in range(events)]
    rows += [rows[-1], rows[-1], {**rows[-2], "bs": None}]  # 3 and 2 rows
    top_rows = (3, 2)[:k]
    assert _expected("E11", rows) == pytest.approx(sum(top_rows) / len(rows) * 100)


def test_top_one_percent_refuses_a_boundary_tie():
    rows = [{"Event_ID": f"e{i}", "area_m2": 1000.0 + i} for i in range(101)]
    rows[98]["area_m2"] = rows[99]["area_m2"]  # ranks k=2 and k+1 tie
    with pytest.raises(GroundTruthError, match="boundary"):
        _expected("E11", rows)


_TOP5 = {  # Event_ID: (area_m2, bs values)
    "big1": (5e9, (4, 4, 1, 6, 6, 6)),  # 2/3 of bs 1-4; 2/6 of all rows
    "big2": (4e9, (4, 1)),
    "big3": (3e9, (3, 3)),
    "big4": (2e9, (2,)),
    "big5": (1e9, (1,)),
    "small": (1e8, (4,)),  # sixth largest of 2020
}


def _top5_rows():
    rows = [
        {"Event_ID": e, "Incid_Name": e.upper(), "area_m2": a, "bs": b}
        for e, (a, bss) in _TOP5.items()
        for b in bss
    ]
    old = {"Event_ID": "old", "Incid_Name": "OLD", "area_m2": 9e9, "bs": 4}
    rows.append({**old, "year": 2019})  # largest, but not of 2020
    return rows


def test_follow_up_refuses_a_top5_that_differs_from_the_history():
    with pytest.raises(GroundTruthError) as err:
        _expected("E08", _top5_rows())
    msg = str(err.value)
    assert "| Incid_Name | Event_ID | area_m2 |" in msg
    assert "| BIG1 | big1 | 5000000000 |" in msg
    assert msg.index("big1") < msg.index("big5") and "small" not in msg


def test_follow_up_highest_share_among_the_history_events():
    gold = ("big1", "big2", "big3", "big4", "big5")
    got = _expected(
        "E08",
        _top5_rows(),
        swap=(f"GOLD = {TOP5_2020_EVENT_IDS!r}\n", f"GOLD = {gold!r}\n"),
    )
    assert got == ["BIG1"]


_ECO1_ROWS = {  # eco1: bs values (one row each)
    6: (4, 4, 4, 3, None, None),  # highest share (75%), too many rows
    2: (4, 4, 3, 2),  # 50%
    10: (4, 1, 1, 1),  # 25%
    11: (4, 1, 1, 1, 1, 6, 6, None),  # 20%, too many rows
}


def _eco1_rows(table):
    return [{"eco1": eco, "bs": b} for eco, bss in table.items() for b in bss]


def test_eco1_share_threshold_binds(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    swap = (f"max_rows = {ECO1_SHARE_MAX_ROWS}\n", "max_rows = 5\n")
    assert _expected("E12", _eco1_rows(_ECO1_ROWS), swap=swap) == 2
    assert "6: 6, 3, 4, 75.000" in caplog.text
    assert "11: 8, 1, 5, 20.000" in caplog.text
    loose = (swap[0], "max_rows = 100\n")
    with pytest.raises(GroundTruthError, match="does not exclude eco1 6"):
        _expected("E12", _eco1_rows(_ECO1_ROWS), swap=loose)
    close = {**_ECO1_ROWS, 10: (4, 4, 1, 1)}  # 50%, level with eco1 2
    with pytest.raises(GroundTruthError, match="under 0.5"):
        _expected("E12", _eco1_rows(close), swap=swap)


def test_level_two_region_with_most_events():
    rows = [
        {"Event_ID": "a", "eco2s": "08.5"},
        {"Event_ID": "b", "eco2s": "08.5"},
        {"Event_ID": "c", "eco2s": "08.5", "bs": None},
        {"Event_ID": "a", "eco2s": "06.2"},
        *[{"Event_ID": "d", "eco2s": "06.2"}] * 6,  # most rows, fewer events
        *[{"Event_ID": e, "eco2s": None} for e in "efgh"],
    ]
    assert _expected("E13", rows) == "08.5"


_REGIONS = {
    "R6": {"eco1": 6, "eco2s": "06.2", "eco3s": "06.2.10"},
    "R9": {"eco1": 9, "eco2s": "09.4", "eco3s": "09.4.02"},
    "R10": {"eco1": 10, "eco2s": "10.1", "eco3s": "10.1.05"},
    "R11": {"eco1": 11, "eco2s": "11.1", "eco3s": "11.1.01"},
}
# (Event_ID, area_m2, region, bs, rows)
_ECO_EVENTS = [
    ("e1", 1e6, "R6", None, 12),
    ("e2", 1e6, "R10", 1, 4),
    ("e3", 1e6, "R11", 3, 3),
    ("e4", 5e6, "R9", 6, 1),  # summing area_m2 over rows would pick R6
    ("e5", 1e3, "R10", None, 1),
    ("e5", 1e3, "R11", None, 2),
    ("e6", 1e3, "R10", None, 1),
    ("e6", 1e3, "R11", None, 2),
    ("e7", 1e3, "R10", None, 1),
]


def test_ecoregion_readings(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    rows = [
        {"Event_ID": e, "area_m2": a, "bs": b, **_REGIONS[r]}
        for e, a, r, b, n in _ECO_EVENTS
        for _ in range(n)
    ]
    got = _expected("E07", rows)
    assert got == {
        "level1|pixels_or_area": ["10", "11", "6", "9"],
        "level1|events": ["10", "11"],
        "level2|pixels_or_area": ["06.2", "09.4", "10.1", "11.1"],
        "level2|events": ["10.1", "11.1"],
        "level3|pixels_or_area": ["06.2.10", "09.4.02", "10.1.05", "11.1.01"],
        "level3|events": ["10.1.05", "11.1.01"],
    }
    for reading in (
        "level1 pixels_or_area [rows, every bs value] -> 6 (12)",
        "level1 pixels_or_area [rows, bs 1-4] -> 10 (4)",
        "level1 pixels_or_area [rows, bs 2-4] -> 11 (3)",
        "level1 pixels_or_area [distinct-event area_m2] -> 9 (",
        "level1 events [events, every region touched] -> 10 (4)",
        "level1 events [events, dominant region] -> 11 (3)",
        "level3 events [events, dominant region] -> 11.1.01 (3)",
    ):
        assert reading in caplog.text


def _fire(geohash, event, day, **extra):
    return {"geohash": geohash, "Event_ID": event, "Ig_Date": day, "eco1": 11, **extra}


_INTERVAL_ROWS = [
    _fire(1, "a", datetime(2000, 1, 1)),
    _fire(1, "a", datetime(2000, 1, 1), bs=None),  # same event, one fire
    _fire(1, "b", datetime(2003, 6, 1)),  # 1247 days
    _fire(1, "c", datetime(2010, 1, 1)),  # 2406 days, 7 years
    _fire(2, "d", datetime(2000, 1, 1)),
    _fire(2, "e", datetime(2005, 12, 31)),  # 2191 days but 5 years
    _fire(3, "g", datetime(2015, 7, 1), bs=1),  # a location with bs = 1 only
    _fire(3, "h", datetime(2016, 7, 1), bs=1),  # 366 days
    _fire(4, "i", datetime(2000, 1, 1)),
    _fire(4, "j", datetime(2004, 12, 31)),  # 1826 days: counts
    _fire(5, "x", datetime(2001, 1, 1), eco1=6),  # outside eco1 = 11
    _fire(5, "y", datetime(2001, 1, 11), eco1=6),
]


def test_reburn_interval_share():
    # Intervals 1247, 2406, 2191, 366, 1826: 3 of 5 within 1,826 days.
    assert _expected("E10", _INTERVAL_ROWS) == pytest.approx(60.0)


def test_reburn_interval_refuses_same_day_events():
    same_day = [
        _fire(6, "k", datetime(2001, 5, 5)),
        _fire(6, "l", datetime(2001, 5, 5)),
    ]
    with pytest.raises(GroundTruthError, match="same-day"):
        _expected("E10", _INTERVAL_ROWS + same_day)

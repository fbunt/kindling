"""Hard benchmark tier: question registry, tier selection, answer specs, the
hard extractor kinds (with a fake client), and reference semantics on
hand-built frames (ties, empty groups, unsigned counts, same-date pairs). No
dataset, no API key, no container runtime; the one sample-parquet check is
skipped without it. Hard-tier verdicts: tests/test_bench_grading_hard.py."""

import ast
import dataclasses
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pytest

from bench import judge
from bench.answers import (
    ANSWERS,
    EXT_ANSWERS,
    FIELD_KINDS,
    HARD_ANSWERS,
    PERCENT,
    AnswerSpec,
    ExtAnswerSpec,
    FieldSpec,
    HardAnswerSpec,
    normalize_unit,
    spec_for,
)
from bench.grading import ext_grader_sha, grader_sha
from bench.ground_truth import GroundTruthError, _normalize, compute_expected
from bench.questions import (
    BY_ID,
    CATEGORIES,
    COTTONWOOD_YEAR,
    ECO8_2020_HISTORY,
    ECO8_2020_WILDFIRE_EVENTS,
    EXTENDED,
    EXTENDED_CATEGORIES,
    HARD,
    HARD_CATEGORIES,
    QUESTIONS,
    TIERS,
    WESTERN_STATES,
    select,
    tier_of,
)
from tests.test_bench_extended import _expected
from tests.test_bench_grading import FakeExtractor, _reply

HARD_IDS = [q.id for q in HARD]
SAMPLE = Path(__file__).parent.parent / ".eval-runs" / "sample.parquet"


# ---------------------------------------------------------------------------
# registry and tiers
# ---------------------------------------------------------------------------


def test_ids_and_categories():
    assert sorted(HARD_IDS) == [f"H{i:02d}" for i in range(1, 10)]
    by_category = {
        c: sorted(q.id for q in HARD if q.category == c) for c in HARD_CATEGORIES
    }
    assert by_category == {
        "hard_semantics": ["H01", "H02", "H04"],
        "hard_pixel_history": ["H03", "H05", "H09"],
        "hard_statistics": ["H07", "H08"],
        "hard_interaction": ["H06"],
    }
    assert not set(HARD_CATEGORIES) & {*CATEGORIES, *EXTENDED_CATEGORIES}
    assert {q.id: q.answer_kind for q in HARD if q.answer_kind != "scalar"} == {
        "H01": "fields",
        "H04": "disambiguate",
        "H05": "fields",
    }


def test_heaviest_reference_runs_last():
    assert HARD[-1].id == "H03"
    assert {q.id for q in HARD if q.expensive_gt} == {"H02", "H03", "H05", "H09"}


def test_tiers_drive_tier_of_select_and_by_id():
    assert list(TIERS) == ["core", "extended", "hard"]
    assert TIERS["hard"].questions == tuple(HARD)
    assert {tier_of(q) for q in HARD} == {"hard"}
    assert {tier_of(q) for q in QUESTIONS} == {"core"}
    assert {tier_of(q) for q in EXTENDED} == {"extended"}
    assert set(BY_ID) == {q.id for t in TIERS.values() for q in t.questions}
    assert len(BY_ID) == 25 + 14 + 9


def test_select_spans_three_tiers():
    assert select("hard") == HARD
    assert select("all") == [*QUESTIONS, *EXTENDED, *HARD]
    assert select(None) == QUESTIONS
    assert [q.id for q in select("hard_pixel_history")] == ["H05", "H09", "H03"]
    assert [q.id for q in select("H07, E04,L03")] == ["H07", "E04", "L03"]
    with pytest.raises(KeyError, match="H10"):
        select("H01,H10")


def test_row_counts_state_their_bs_scope():
    # A06 rule: "rows" or "pixel" without a bs scope was read as bs 2-4.
    for q in HARD:
        if "row" in q.text or "pixel" in q.text:
            assert "bs" in q.text, q.id


def test_event_counts_state_their_incident_type_scope():
    # H04 names one fire; every other question states which types count.
    for q in HARD:
        if q.id != "H04":
            text = q.text.casefold()
            assert "count every incident type" in text or "incid_type" in text, q.id
    assert "Incid_Type = 1" in BY_ID["H06"].history[0][1]


def test_only_the_follow_up_question_has_history():
    assert [q.id for q in HARD if q.history] == ["H06"]
    assert BY_ID["H06"].history == ECO8_2020_HISTORY
    assert [role for role, _ in ECO8_2020_HISTORY] == ["user", "model"]


def test_filled_constants_reach_the_wording_and_references():
    # Set from the full-data tables the references log (bench gt,
    # 2026-10-20); the references refuse any other value (see below).
    assert (COTTONWOOD_YEAR, ECO8_2020_WILDFIRE_EVENTS) == (1994, 28)
    assert BY_ID["H04"].text == "How many acres did the COTTONWOOD fire of 1994 burn?"
    assert ECO8_2020_HISTORY[1][1].startswith("28 distinct wildfire events")
    assert BY_ID["H04"].reference_code.startswith("YEAR = 1994\n")
    assert BY_ID["H06"].reference_code.startswith("N = 28\n")


def test_decimals_and_bands_follow_the_wording():
    assert "Give two decimals." in BY_ID["H07"].text
    assert "to two decimals." in BY_ID["H08"].text
    assert "Give one decimal." in BY_ID["H09"].text
    bands = {q.id: (q.tolerance_rel, q.tolerance_abs) for q in HARD}
    assert bands["H02"] == (0.005, None)
    assert {qid: bands[qid][1] for qid in ("H03", "H06", "H07", "H08", "H09")} == {
        "H03": 0.5,
        "H06": 0.0,
        "H07": 0.01,
        "H08": 0.005,
        "H09": 0.1,
    }
    assert {qid: HARD_ANSWERS[qid].min_decimals for qid in ("H07", "H08", "H09")} == {
        "H07": 2,
        "H08": 2,
        "H09": 1,
    }
    assert HARD_ANSWERS["H06"].integer


@pytest.mark.parametrize("qid", HARD_IDS)
def test_reference_compiles_and_assigns_expected(qid):
    tree = ast.parse(BY_ID[qid].reference_code)
    compile(tree, qid, "exec")
    targets = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "expected" in targets


@pytest.mark.parametrize("qid", HARD_IDS)
def test_no_is_in_inside_an_aggregation(qid):
    # is_in inside a streaming group_by().agg() is slow on the full data.
    tree = ast.parse(BY_ID[qid].reference_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "agg":
            inner = ast.dump(ast.Module(body=[ast.Expr(a) for a in node.args]))
            assert "'is_in'" not in inner, qid


# ---------------------------------------------------------------------------
# answer specs
# ---------------------------------------------------------------------------


def test_every_hard_question_has_a_hard_answer_spec():
    assert set(HARD_ANSWERS) == set(HARD_IDS)
    for q in HARD:
        spec = spec_for(q)
        assert spec is HARD_ANSWERS[q.id] and isinstance(spec, HardAnswerSpec)
        assert bool(spec.fields) == (q.answer_kind == "fields"), q.id
        for unit in spec.units or ():
            assert normalize_unit(unit) == unit, (q.id, unit)
    # The other tiers keep their own specs.
    assert all(spec_for(q) is ANSWERS[q.id] for q in QUESTIONS)
    assert all(spec_for(q) is EXT_ANSWERS[q.id] for q in EXTENDED)
    with pytest.raises(KeyError, match="X01"):
        spec_for(dataclasses.replace(HARD[0], id="X01"))


def test_answer_spec_gains_no_field():
    # asdict(AnswerSpec) is hashed into the core grader sha.
    core = {f.name for f in dataclasses.fields(AnswerSpec)}
    assert core == {
        "units",
        "unit_required",
        "integer",
        "min_decimals",
        "aliases",
        "series_keys",
        "date",
        "compound",
        "regex_check",
        "note",
    }
    hard = {f.name for f in dataclasses.fields(HardAnswerSpec)} - core
    ext = {f.name for f in dataclasses.fields(ExtAnswerSpec)} - core
    assert hard == {"fields"} and not hard & ext


def test_field_specs():
    for qid in ("H01", "H05"):
        for name, part in HARD_ANSWERS[qid].fields.items():
            assert isinstance(part, FieldSpec) and part.kind in FIELD_KINDS
            assert part.description, (qid, name)
            for unit in part.units or ():
                assert normalize_unit(unit) == unit, (qid, name, unit)
    h01, h05 = HARD_ANSWERS["H01"].fields, HARD_ANSWERS["H05"].fields
    assert list(h01) == ["state", "count"]
    assert h01["state"].kind == "state" and h01["count"].integer
    assert list(h05) == ["percent", "name", "year"]
    assert h05["percent"].units == PERCENT and h05["percent"].tolerance_abs == 0.5
    assert h05["name"].aliases == {
        "YOLLA BOLLY COMPLEX (IRON)": ("yolla bolly complex (iron)", "iron")
    }
    assert h05["year"].integer


def test_hard_expected_shapes_survive_normalization():
    for value in (
        {"state": "FL", "count": 1388},
        {"percent": 24.7, "name": "YOLLA BOLLY COMPLEX (IRON)", "year": 2008},
        {
            "year": 1994,
            "count": 3,
            "events": [{"Event_ID": "a", "Incid_Type": 1, "acres": 1.5}],
        },
    ):
        assert _normalize(value) == value


# ---------------------------------------------------------------------------
# extractor kinds: fields and disambiguate
# ---------------------------------------------------------------------------

HARD_KINDS = ("fields", "disambiguate")
H01_PARTS = {
    "state": "the US state with the most events (code or name)",
    "count": "how many distinct prescribed-fire events it had",
}


def _part(value_text=None, value=None, unit=None):
    return {"value_text": value_text, "value": value, "unit": unit}


def _parsed(text=None, value=None, value_text=None, unit=None):
    return {"text": text, "value": value, "value_text": value_text, "unit": unit}


def _fields_reply(**over):
    parts = {"state": _part("Florida (FL)"), "count": _part("1,388 events", 1388)}
    return _reply(parts={**parts, **over})


def _disambiguation(**over):
    return {
        "several_noted": True,
        "combined_as_one": False,
        "events": [
            {"identifier": "COTTONWOOD (ID)", "area_text": "1,200 acres"},
            {"identifier": "COTTONWOOD (NV)", "area_text": None},
        ],
        **over,
    }


def test_extraction_parts_name_the_parts_in_the_order_asked():
    assert judge.extraction_parts(BY_ID["H01"]) == H01_PARTS
    assert list(judge.extraction_parts(BY_ID["H05"])) == ["percent", "name", "year"]
    for q in [*QUESTIONS, *EXTENDED, *HARD]:
        if q.answer_kind != "fields":
            assert judge.extraction_parts(q) is None, q.id


@pytest.mark.parametrize("qid", ["H01", "H05"])
def test_fields_schema_has_one_required_object_per_part(qid):
    parts = judge.extraction_parts(BY_ID[qid])
    schema = judge.response_schema("fields", parts)
    assert schema["property_ordering"] == [
        "status",
        "parts",
        "multiple_candidates",
        "sampled_disclosed",
    ]
    assert schema["required"] == [
        "status",
        "parts",
        "multiple_candidates",
        "sampled_disclosed",
    ]
    obj = schema["properties"]["parts"]
    assert obj["type"] == "OBJECT"
    assert obj["required"] == obj["property_ordering"] == list(parts)
    for name in parts:
        part = obj["properties"][name]
        assert part["required"] == ["value_text", "value", "unit"]
        assert all(p["nullable"] for p in part["properties"].values())
        assert part["properties"]["value"]["type"] == "NUMBER"
    cfg = judge.build_config("fields", parts)
    assert cfg.system_instruction == judge.SYSTEM_INSTRUCTION
    assert cfg.temperature == 0 and cfg.response_mime_type == "application/json"
    validated = cfg.response_schema.properties["parts"]
    assert list(validated.properties) == list(parts)
    assert validated.properties[list(parts)[0]].properties["value"].nullable


def test_disambiguate_schema_and_guidance():
    schema = judge.response_schema("disambiguate")
    assert schema["property_ordering"] == [
        "status",
        "several_noted",
        "combined_as_one",
        "events",
        "unit",
        "multiple_candidates",
        "sampled_disclosed",
    ]
    assert schema["required"] == [
        "status",
        "several_noted",
        "combined_as_one",
        "events",
        "multiple_candidates",
        "sampled_disclosed",
    ]
    props = judge.build_config("disambiguate").response_schema.properties
    assert props["several_noted"].type == props["combined_as_one"].type == "BOOLEAN"
    assert props["events"].max_items == judge.HARD_EVENTS_CAP == 10
    assert props["events"].items.required == ["identifier", "area_text"]
    assert props["events"].items.properties["area_text"].nullable
    guidance = judge._HARD_KIND_GUIDANCE["disambiguate"]
    assert "whatever `status` is" in guidance and "overrides" in guidance
    assert "labelled as the total over several events" in guidance
    assert "the first 10 in the order given" in guidance


def test_fields_prompt_lists_part_names_and_descriptions():
    prompt = judge.build_prompt("q?", "r.", "fields", H01_PARTS)
    guidance = judge._HARD_KIND_GUIDANCE["fields"]
    assert prompt.startswith(f"Answer kind: {guidance}\n- state: the US state")
    assert "\n- count: how many distinct prescribed-fire events it had\n" in prompt
    assert judge.build_prompt("q?", "r.", "disambiguate").startswith(
        f"Answer kind: {judge._HARD_KIND_GUIDANCE['disambiguate']}"
    )


# Expected values (or text derived from them) no hard prompt may carry.
_HARD_LEAKS = {
    "H01": ("1388", "1,388", "Florida"),
    "H04": ("COTTONWOOD CREEK", "Incid_Type"),
    "H05": ("YOLLA", "24.7", "2008", "MENDENHALL", "0.5"),
}


@pytest.mark.parametrize("qid", sorted(_HARD_LEAKS))
def test_hard_extractor_is_blind(qid):
    q = BY_ID[qid]
    parts = judge.extraction_parts(q)
    reply = (
        _reply(parts={name: _part() for name in parts})
        if parts
        else _reply(**_disambiguation())
    )
    fake = FakeExtractor(lambda p: reply)
    out = judge.extract(fake, q.text, "The answer is X.", q.answer_kind, parts)
    assert out["error"] is None and out["extraction"]["status"] == "answer"
    (call,) = fake.calls
    seen = call["contents"] + call["config"].system_instruction
    assert q.text in call["contents"]
    spec = HARD_ANSWERS[qid]
    criterion = q.criterion.split("{expected}")[0]
    leaks = [*_HARD_LEAKS[qid], criterion, spec.note, "expected", "tolerance"]
    for field_spec in spec.fields.values():
        leaks += [*field_spec.aliases]
        leaks += [a for alts in field_spec.aliases.values() for a in alts]
    for leak in leaks:
        assert leak not in seen.replace(q.text, ""), leak
    # The kind's guidance plus, for a fields answer, only each part's name and
    # description: no units, integer flag or band.
    listed = "".join(f"\n- {n}: {f.description}" for n, f in spec.fields.items())
    guidance = judge._HARD_KIND_GUIDANCE[q.answer_kind] + listed
    assert call["contents"].startswith(f"Answer kind: {guidance}\n\n<question>")


def test_fields_parses_a_well_formed_reply():
    out = judge.parse_extraction(_fields_reply(), "fields", H01_PARTS)
    assert out == {
        "status": "answer",
        "multiple_candidates": False,
        "sampled_disclosed": False,
        "parts": {
            "state": _parsed("Florida (FL)"),
            "count": _parsed("1,388 events", 1388, "1388"),
        },
    }
    h05 = judge.extraction_parts(BY_ID["H05"])
    reply = _reply(
        parts={
            "percent": _part("24.70%", 24.70, "%"),
            "name": _part("Yolla Bolly Complex (Iron)"),
            "year": _part("2008", 2008),
            "extra": _part("dropped"),  # a part the question does not name
        }
    ).replace('"value": 24.7,', '"value": 24.70,')  # the digits as given
    out = judge.parse_extraction(reply, "fields", h05)
    assert out["parts"] == {
        "percent": _parsed("24.70%", 24.7, "24.70", "%"),
        "name": _parsed("Yolla Bolly Complex (Iron)"),
        "year": _parsed("2008", 2008, "2008"),
    }


def test_fields_missing_or_blank_part_fields_read_as_null():
    reply = _reply(parts={"state": {}, "count": {"value_text": " ", "unit": ""}})
    out = judge.parse_extraction(reply, "fields", H01_PARTS)
    assert out["parts"] == {"state": _parsed(), "count": _parsed()}


@pytest.mark.parametrize(
    "parts",
    [
        None,  # parts missing
        [_part("FL"), _part("1388", 1388)],
        "FL, 1388",
        {"state": _part("FL")},  # a part missing
        {"state": "FL", "count": _part("1388", 1388)},
        {"state": _part("FL"), "count": _part("1388", "1388")},
        {"state": _part("FL"), "count": _part("1388", True)},
        {"state": _part("FL"), "count": _part(1388, 1388)},
        {"state": _part(["FL"]), "count": _part("1388", 1388)},
        {"state": _part("FL"), "count": _part("1388", 1388, unit=1)},
    ],
)
def test_fields_rejects_malformed_reply(parts):
    raw = _reply() if parts is None else _reply(parts=parts)
    with pytest.raises(judge.ExtractionError):
        judge.parse_extraction(raw, "fields", H01_PARTS)


@pytest.mark.parametrize("over", [{"status": "partial"}, {"sampled_disclosed": None}])
def test_fields_rejects_a_bad_common_field(over):
    raw = json.dumps({**json.loads(_fields_reply()), **over})
    with pytest.raises(judge.ExtractionError):
        judge.parse_extraction(raw, "fields", H01_PARTS)


@pytest.mark.parametrize("unit", ["%", 3, None])
def test_fields_have_no_top_level_unit(unit):
    # The schema has none (each part carries its unit), so a stray one in the
    # reply is ignored rather than kept as an always-empty field.
    raw = json.dumps({**json.loads(_fields_reply()), "unit": unit})
    out = judge.parse_extraction(raw, "fields", H01_PARTS)
    assert "unit" not in out
    assert "unit" not in judge.no_answer_extraction("fields", H01_PARTS)
    assert "unit" not in judge.response_schema("fields", H01_PARTS)["properties"]


def test_disambiguate_parses_a_well_formed_reply():
    out = judge.parse_extraction(_reply(**_disambiguation()), "disambiguate")
    assert out == {
        "status": "answer",
        "multiple_candidates": False,
        "sampled_disclosed": False,
        "unit": None,
        **_disambiguation(),
    }


@pytest.mark.parametrize("status", judge.STATUSES)
def test_disambiguate_fields_are_kept_under_every_status(status):
    out = judge.parse_extraction(
        _reply(status=status, **_disambiguation()), "disambiguate"
    )
    assert out["status"] == status and out["several_noted"] is True
    assert len(out["events"]) == 2


def test_disambiguate_keeps_only_the_leading_events():
    events = [{"identifier": f"E{i}", "area_text": ""} for i in range(12)]
    out = judge.parse_extraction(
        _reply(**_disambiguation(events=events)), "disambiguate"
    )
    assert out["events"] == [
        {"identifier": f"E{i}", "area_text": None} for i in range(judge.HARD_EVENTS_CAP)
    ]
    # Items past the cap are dropped unread: a bad one cannot reject the reply.
    past = [*events[: judge.HARD_EVENTS_CAP], {"identifier": 7}, "junk"]
    out = judge.parse_extraction(_reply(**_disambiguation(events=past)), "disambiguate")
    assert len(out["events"]) == judge.HARD_EVENTS_CAP


def test_disambiguate_keeps_an_event_without_an_identifier():
    events = [
        {"identifier": "COTTONWOOD (ID)", "area_text": "1,200 acres"},
        {"area_text": "5 acres"},
        {"identifier": "", "area_text": "40 acres"},
        {"identifier": "  ", "area_text": None},
    ]
    out = judge.parse_extraction(
        _reply(**_disambiguation(events=events)), "disambiguate"
    )
    assert out["events"] == [
        {"identifier": "COTTONWOOD (ID)", "area_text": "1,200 acres"},
        {"identifier": None, "area_text": "5 acres"},
        {"identifier": None, "area_text": "40 acres"},
        {"identifier": None, "area_text": None},
    ]


@pytest.mark.parametrize(
    "over",
    [
        {"several_noted": None},
        {"several_noted": "yes"},
        {"combined_as_one": 0},
        {"events": None},  # required: an empty list is a verdict input
        {"events": {"identifier": "A"}},
        {"events": ["COTTONWOOD"]},
        {"events": [{"identifier": 7, "area_text": None}]},
        {"events": [{"identifier": "A", "area_text": 1200}]},
    ],
)
def test_disambiguate_rejects_malformed_reply(over):
    fields = {k: v for k, v in _disambiguation(**over).items() if v is not None}
    with pytest.raises(judge.ExtractionError):
        judge.parse_extraction(_reply(**fields), "disambiguate")


@pytest.mark.parametrize("kind", HARD_KINDS)
def test_hard_kind_no_answer_extraction(kind):
    parts = H01_PARTS if kind == "fields" else None
    empty = {
        "fields": {"parts": {"state": _parsed(), "count": _parsed()}},
        "disambiguate": {
            "unit": None,
            "several_noted": False,
            "combined_as_one": False,
            "events": [],
        },
    }[kind]
    fake = FakeExtractor(lambda p: pytest.fail("no call expected"))
    out = judge.extract(fake, "q?", "  ", kind, parts)
    assert out["extraction"] == judge.no_answer_extraction(kind, parts)
    assert out["extraction"] == {
        "status": "no_answer",
        "multiple_candidates": False,
        "sampled_disclosed": False,
        **empty,
    }
    assert not fake.calls


def test_malformed_fields_reply_is_retried_once_then_a_grader_error():
    replies = [_fields_reply(count="1388"), _fields_reply()]
    fake = FakeExtractor(lambda p: replies.pop(0))
    out = judge.extract(fake, "q?", "Florida, 1,388.", "fields", H01_PARTS)
    assert out["attempts"] == 2 and out["error"] is None
    assert out["extraction"]["parts"]["count"]["value"] == 1388
    fake = FakeExtractor(lambda p: _reply(parts={"state": _part("FL")}))
    out = judge.extract(fake, "q?", "Florida, 1,388.", "fields", H01_PARTS)
    assert len(fake.calls) == 2 and out["extraction"] is None
    assert out["error"] == "malformed reply: part 'count' is missing"


def test_an_unknown_kind_raises_instead_of_reading_names():
    reply = _reply(names=["FLATTOP"])
    fake = FakeExtractor(lambda p: reply)
    for call in (
        lambda: judge.response_schema("ranking"),
        lambda: judge.build_prompt("q?", "r.", "ranking"),
        lambda: judge.build_config("ranking"),
        lambda: judge.parse_extraction(reply, "ranking"),
        lambda: judge.no_answer_extraction("ranking"),
        lambda: judge.extract(fake, "q?", "FLATTOP.", "ranking"),
        lambda: judge.extract(fake, "q?", "  ", "ranking"),
    ):
        with pytest.raises(ValueError, match="unknown answer kind 'ranking'"):
            call()
    assert not fake.calls
    # The names kinds still read names; a fields answer needs its parts, and
    # only a fields answer takes them.
    assert judge.parse_extraction(reply, "text")["names"] == ["FLATTOP"]
    with pytest.raises(ValueError, match="needs its part names"):
        judge.parse_extraction(_fields_reply(), "fields")
    with pytest.raises(ValueError, match="needs its part names"):
        judge.extract(fake, "q?", "  ", "fields", {})
    with pytest.raises(ValueError, match="not 'scalar'"):
        judge.build_prompt("q?", "r.", "scalar", H01_PARTS)


def test_hard_fingerprint_covers_every_hard_question(monkeypatch):
    def snapshot(fp):  # the fingerprints share the module's dicts
        return json.dumps(fp(), sort_keys=True)

    fp = judge.hard_fingerprint()
    assert list(fp["questions"]) == HARD_IDS
    for q in HARD:
        parts = judge.extraction_parts(q)
        assert fp["questions"][q.id] == {
            "kind": q.answer_kind,
            "guidance": judge._guidance(q.answer_kind, parts),
            "schema": judge.response_schema(q.answer_kind, parts),
        }
    assert fp["guidance"] == judge._HARD_KIND_GUIDANCE
    assert fp["events_cap"] == judge.HARD_EVENTS_CAP
    for other in (judge.fingerprint(), judge.ext_fingerprint()):
        assert not set(HARD_KINDS) & (set(other["guidance"]) | set(other["schemas"]))
    core, ext, hard = (
        snapshot(judge.fingerprint),
        snapshot(judge.ext_fingerprint),
        snapshot(judge.hard_fingerprint),
    )
    edits = [
        (judge._HARD_KIND_GUIDANCE, "fields", "!"),
        (judge._HARD_KIND_GUIDANCE, "disambiguate", "!"),
    ]
    for table, kind, suffix in edits:
        monkeypatch.setitem(table, kind, table[kind] + suffix)
        assert snapshot(judge.hard_fingerprint) != hard
        monkeypatch.undo()
    monkeypatch.setitem(judge._FIELD_PART["properties"], "note", {"type": "STRING"})
    assert snapshot(judge.hard_fingerprint) != hard
    monkeypatch.undo()
    h05 = HARD_ANSWERS["H05"]
    reworded = {**h05.fields}
    reworded["year"] = dataclasses.replace(reworded["year"], description="the year")
    monkeypatch.setitem(HARD_ANSWERS, "H05", dataclasses.replace(h05, fields=reworded))
    assert snapshot(judge.hard_fingerprint) != hard
    assert snapshot(judge.fingerprint) == core
    assert snapshot(judge.ext_fingerprint) == ext
    assert grader_sha.__wrapped__() == "ee57b57a09b8"  # uncached: hashed now
    assert ext_grader_sha.__wrapped__() == "1da536285bd3"


# ---------------------------------------------------------------------------
# references on hand-built frames
# ---------------------------------------------------------------------------


def _day(year, month=7, day=1):
    return datetime(year, month, day)


def test_state_with_most_prescribed_fires():
    rows = [
        {"Event_ID": "FL1", "Incid_Type": 2, "year": 2010},
        {"Event_ID": "FL1", "Incid_Type": 2, "year": 2010, "bs": None},
        {"Event_ID": "FL2", "Incid_Type": 2, "year": 2015},
        {"Event_ID": "FL3", "Incid_Type": 2, "year": 2019},
        {"Event_ID": "GA1", "Incid_Type": 2, "year": 2012},
        {"Event_ID": "GA2", "Incid_Type": 2, "year": 2012},
        *[{"Event_ID": f"GA{i}", "Incid_Type": 2, "year": 2009} for i in range(3, 6)],
        *[{"Event_ID": f"GA{i}", "Incid_Type": 2, "year": 2020} for i in range(6, 9)],
        *[{"Event_ID": f"GA{i}", "Incid_Type": 1, "year": 2014} for i in range(9, 12)],
    ]
    assert _expected("H01", rows) == {"state": "FL", "count": 3}
    tied = rows + [{"Event_ID": "GA12", "Incid_Type": 2, "year": 2011}]
    with pytest.raises(GroundTruthError, match="tie for most events: FL and GA"):
        _expected("H01", tied)
    with pytest.raises(GroundTruthError, match="state code"):
        _expected("H01", rows + [{"Event_ID": "1x", "Incid_Type": 2, "year": 2011}])


_ACRE = 4046.8564224
_COTTONWOODS = [  # (Event_ID, Incid_Name, year, Incid_Type, acres)
    ("c1", "COTTONWOOD", 1994, 1, 100),
    ("c2", "COTTONWOOD", 1994, 1, 250),
    ("c3", "COTTONWOOD", 1994, 1, 40),
    ("c4", "COTTONWOOD", 1994, 2, 10),  # a prescribed namesake is listed too
    ("c5", "COTTONWOOD CREEK", 1994, 1, 500),  # a near-name: logged only
    ("d1", "COTTONWOOD", 2000, 1, 70),
    ("d2", "COTTONWOOD", 2000, 1, 80),
    ("d3", "COTTONWOOD", 2000, 2, 90),  # 3 events but 2 wildfires
]


def _cottonwood_rows():
    rows = [
        {
            "Event_ID": e,
            "Incid_Name": n,
            "year": y,
            "Incid_Type": t,
            "area_m2": a * _ACRE,
        }
        for e, n, y, t, a in _COTTONWOODS
    ]
    return rows + [{**rows[0], "bs": None}]  # a second row of c1


def test_namesakes_refuse_until_the_year_qualifies():
    unset = (f"YEAR = {COTTONWOOD_YEAR}\n", "YEAR = None\n")
    with pytest.raises(GroundTruthError) as err:
        _expected("H04", _cottonwood_rows(), swap=unset)
    msg = str(err.value)
    assert "COTTONWOOD_YEAR is None" in msg and "([1994])" in msg
    assert "| 1994 | 4 | 3 |" in msg and "| 2000 | 3 | 2 |" in msg
    with pytest.raises(GroundTruthError, match="COTTONWOOD_YEAR is 2000"):
        _expected(
            "H04",
            _cottonwood_rows(),
            swap=(f"YEAR = {COTTONWOOD_YEAR}\n", "YEAR = 2000\n"),
        )


def test_namesakes_of_the_year(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    got = _expected(
        "H04", _cottonwood_rows(), swap=(f"YEAR = {COTTONWOOD_YEAR}\n", "YEAR = 1994\n")
    )
    assert got["year"] == 1994 and got["count"] == 4
    assert [(e["Event_ID"], e["Incid_Type"]) for e in got["events"]] == [
        ("c1", 1),
        ("c2", 1),
        ("c3", 1),
        ("c4", 2),
    ]
    assert [e["acres"] for e in got["events"]] == pytest.approx([100, 250, 40, 10])
    assert "1 near-names of COTTONWOOD in 1994: ['COTTONWOOD CREEK']" in caplog.text


_ECO8_EVENTS = {  # Event_ID: (Incid_Type, year, [(eco1, wui_bool), ...])
    "w1": (1, 2020, [(8, 0)]),  # in the history, no WUI
    "w2": (1, 2020, [(8, 1), (8, None)]),
    "w3": (1, 2020, [(9, 1), (8, 0)]),  # WUI only outside the region
    "p1": (2, 2020, [(8, 1)]),
    "p2": (2, 2020, [(8, 1), (8, 0), (None, 1)]),
    "p3": (2, 2020, [(8, None)]),  # unknown WUI: not a WUI row
    "p4": (2, 2020, [(8, 1)]),
    "o1": (0, 2020, [(8, 1)]),  # unknown type: not asked for
    "y1": (1, 2019, [(8, 1)]),
    "x1": (2, 2020, [(9, 1)]),
}


def _eco8_rows(events=_ECO8_EVENTS):
    return [
        {"Event_ID": e, "Incid_Type": t, "year": y, "eco1": eco, "wui_bool": w}
        for e, (t, y, cells) in events.items()
        for eco, w in cells
    ]


def test_follow_up_refuses_a_history_count_that_differs():
    with pytest.raises(GroundTruthError, match="set ECO8_2020_WILDFIRE_EVENTS to 3"):
        _expected("H06", _eco8_rows())
    with pytest.raises(GroundTruthError, match="states 4 events"):
        _expected(
            "H06", _eco8_rows(), swap=(f"N = {ECO8_2020_WILDFIRE_EVENTS}\n", "N = 4\n")
        )


def test_follow_up_counts_wui_events_in_the_region(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    # w2, p1, p2, p4: wildfire or prescribed, 2020, a WUI row in eco1 = 8.
    assert (
        _expected(
            "H06", _eco8_rows(), swap=(f"N = {ECO8_2020_WILDFIRE_EVENTS}\n", "N = 3\n")
        )
        == 4
    )
    for reading in (
        "'eco1 = 8 dropped': 6",
        "'2020 dropped': 5",
        "'wildfire only': 1",
        "'every incident type': 5",
        "'WUI anywhere': 5",
    ):
        assert reading in caplog.text
    wildfires = {e: v for e, v in _ECO8_EVENTS.items() if v[0] == 1}
    with pytest.raises(GroundTruthError, match="'wildfire only' gives the same"):
        _expected(
            "H06",
            _eco8_rows(wildfires),
            swap=(f"N = {ECO8_2020_WILDFIRE_EVENTS}\n", "N = 3\n"),
        )


def _theil_sen(ys):
    years = range(1984, 2012)
    pairs = [(i, j) for i in range(28) for j in range(i + 1, 28)]
    slopes = sorted((ys[j] - ys[i]) / (j - i) for i, j in pairs)
    assert len(slopes) == 378 and len(years) == 28
    return (slopes[188] + slopes[189]) / 2


# Annual counts, 1984-2011 ((7i mod 11) + i // 2); 1984 has none (zero-filled).
# Theil-Sen 0.49 between middle slopes 0.48 and 0.50; OLS 0.502.
_ANNUAL = [0, 7, 4, 11, 8, 4, 12, 8, 5, 12, 9, 5, 13, 9, 17, 13, 10, 17, 14, 10, 18]
_ANNUAL += [14, 11, 18, 15, 22, 19, 15]


def _western_rows(annual):
    rows = []
    for year, n in zip(range(1984, 2012), annual, strict=True):
        for i in range(n):
            state = WESTERN_STATES[(year + i) % len(WESTERN_STATES)]
            event = {
                "Event_ID": f"{state}{year}{i:02d}",
                "Incid_Type": (0, 1, 3)[i % 3],
                "area_m2": 4_050_001.0,
                "year": year,
            }
            rows += [event, {**event, "bs": None}]  # two rows, one event
    excluded = [
        {"Event_ID": "CA2000rx", "Incid_Type": 2},  # prescribed
        {"Event_ID": "CA2000sm", "area_m2": 4_050_000.0},  # not greater
        {"Event_ID": "TX2000a"},  # not a listed state
        {"Event_ID": "CA1983a", "year": 1983},
        {"Event_ID": "CA2012a", "year": 2012},
    ]
    base = {"Incid_Type": 1, "area_m2": 9e6, "year": 2000}
    return rows + [{**base, **e} for e in excluded]


def test_theil_sen_slope_of_large_western_fires(caplog, monkeypatch):
    # Without scikit-learn the spatial-median reading is only logged as
    # unchecked; on this toy series it would land inside the band.
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    got = _expected("H07", _western_rows(_ANNUAL))
    assert got == pytest.approx(_theil_sen(_ANNUAL)) and got == pytest.approx(0.49)
    assert f"annual counts {_ANNUAL}" in caplog.text


def test_theil_sen_refuses_a_spatial_median_inside_the_band():
    pytest.importorskip("sklearn")
    with pytest.raises(GroundTruthError, match="reading 'sklearn TheilSenRegressor'"):
        _expected("H07", _western_rows(_ANNUAL))


def test_theil_sen_refuses_a_reading_inside_the_band():
    # A straight line: every slope is 1, so OLS and both medians agree.
    line = list(range(1, 29))
    with pytest.raises(GroundTruthError, match="reading 'median_low' gives 1.0000"):
        _expected("H07", _western_rows(line))


def _west_rows(spec):
    """spec: (Event_ID, year, Incid_Type, {eco1: rows}) per event."""
    return [
        {"Event_ID": e, "year": y, "Incid_Type": t, "eco1": eco, "bs": (None, 3)[i % 2]}
        for e, y, t, cells in spec
        for eco, n in cells.items()
        for i in range(n)
    ]


# West events per year 1984-2018 by type: wildfire, prescribed, unknown.
_WEST_WILD = [4, 4, 3, 3, 3, 3, 3, 3, 2, 6, 3, 4, 2, 5, 3, 6, 2, 4, 3, 2, 4, 8, 8, 6]
_WEST_WILD += [7, 7, 6, 8, 9, 9, 8, 7, 8, 7, 5]
_WEST_RX = [1, 2, 2, 1, 2, 1, 2, 2, 2, 0, 1, 2, 2, 1, 2, 0, 1, 0, 0, 2, 1, 1, 1, 1]
_WEST_RX += [1, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0]
_WEST_UNKNOWN = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
_WEST_UNKNOWN += [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]


def _west_events():
    spec = []
    years = range(1984, 2019)
    for year, *mix in zip(years, _WEST_WILD, _WEST_RX, _WEST_UNKNOWN, strict=True):
        types = [t for t, n in zip((1, 2, 0), mix, strict=True) for _ in range(n)]
        for i, t in enumerate(types):
            spec.append((f"{year}-{i}", year, t, {(6, 7, 10, 11, 12, 13)[i % 6]: 2}))
        spec.append((f"{year}-east", year, 1, {8: 3}))
    spec += [
        ("tie", 1990, 1, {9: 2, 6: 2}),  # tie: the smaller code (6) wins
        ("tie-east", 1991, 1, {7: 2, 5: 2}),  # tie: 5 wins, not West
        ("span", 2010, 1, {6: 1, 9: 3}),  # mostly Great Plains: not West
        ("nulls", 2011, 1, {None: 5, 7: 1}),  # null eco1 rows do not count
    ]
    return spec


def _west_ratio(spec, early=(1984, 1999), late=(2005, 2018)):
    counts = {}
    for _, year, _t, cells in spec:
        real = {eco: n for eco, n in cells.items() if eco is not None}
        top = max(real.values())
        if min(eco for eco, n in real.items() if n == top) in (6, 7, 10, 11, 12, 13):
            counts[year] = counts.get(year, 0) + 1

    def median(lo, hi):
        return statistics.median(counts.get(y, 0) for y in range(lo, hi + 1))

    return median(*late) / median(*early)


def test_west_median_ratio(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    spec = _west_events()
    got = _expected("H08", _west_rows(spec))
    assert got == pytest.approx(_west_ratio(spec)) and got == pytest.approx(18 / 11)
    assert "counted in the West on any row" in caplog.text


def test_west_median_ratio_refuses_a_reading_inside_the_band():
    wildfires = [e for e in _west_events() if e[2] == 1]
    with pytest.raises(GroundTruthError, match="reading 'wildfire only'"):
        _expected("H08", _west_rows(wildfires))


def test_burned_footprint():
    rows = [{"Event_ID": "a", "geohash": g, "area_m2": 9e5} for g in (1, 17, 33, -5, 2)]
    rows += [
        {"Event_ID": "b", "geohash": 1, "bs": None, "area_m2": 4.5e6},
        {"Event_ID": "b", "geohash": 17, "bs": 6, "area_m2": 4.5e6},
        {"Event_ID": "c", "geohash": 99, "year": 1983},  # outside 1984-2022
    ]
    # 5 locations (17 and 33 share partition 1; -5 is in partition 11).
    assert _expected("H02", rows) == pytest.approx(5 * 900 / 1e6)
    with pytest.raises(GroundTruthError, match="1 rows have no geohash"):
        _expected("H02", rows + [{"Event_ID": "d", "geohash": None}])
    single = [{"Event_ID": "a", "geohash": g, "area_m2": 2700.0} for g in (1, 2, 3)]
    with pytest.raises(GroundTruthError, match="reading 'rows x 900 m2'"):
        _expected("H02", single)


_AUGUST = {"Incid_Name": "AUGUST COMPLEX", "year": 2020, "Ig_Date": _day(2020, 8, 17)}
_IRON = {
    "Incid_Name": "YOLLA BOLLY COMPLEX (IRON)",
    "year": 2008,
    "Ig_Date": _day(2008),
}
_MENDENHALL = {"Incid_Name": "MENDENHALL", "year": 1987, "Ig_Date": _day(1987)}


def _august_rows(*extra):
    def at(event, fields, cells):
        return [{"Event_ID": event, **fields, "geohash": g, "bs": b} for g, b in cells]

    august = [(1, 3), (2, 3), (3, None), (4, 4), (5, 2), (6, 5), (7, 3), (8, 1)]
    august += [(9, 2), (10, 3), (17, 3)]  # 17 shares partition 1 with 1
    return [
        *at("ac", _AUGUST, august),
        *at("iron", _IRON, [(1, 1), (2, None), (3, 4), (4, 3), (17, 1), (50, 3)]),
        *at("mend", _MENDENHALL, [(3, 3), (4, 3), (5, 3)]),
        *at("scu", {**_AUGUST, "Incid_Name": "SCU"}, [(6, 3)]),  # same date
        *at("later", {"year": 2021, "Ig_Date": _day(2021)}, [(7, 3)]),
        *at("old-ac", {**_AUGUST, "year": 1999, "Ig_Date": _day(1999)}, [(1, 3)]),
        *extra,
    ]


def test_earlier_fires_under_the_august_complex(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    got = _expected("H05", _august_rows())
    # 6 of 11 locations (1-5, 17) have an earlier fire; IRON is at 5 of them.
    assert got == {
        "percent": pytest.approx(6 / 11 * 100),
        "name": "YOLLA BOLLY COMPLEX (IRON)",
        "year": 2008,
    }
    # bs 2-4 rows only: 3 (1, 4, 5) of 8 locations, MENDENHALL first.
    assert "H05 [bs 2-4]: 37.5 of 8 locations" in caplog.text
    assert "[(('MENDENHALL', 1987), 2), (('YOLLA BOLLY" in caplog.text


def test_earlier_fires_refuse_a_tie_and_a_second_august_complex():
    more = [{"Event_ID": "mend", **_MENDENHALL, "geohash": g, "bs": 3} for g in (1, 2)]
    with pytest.raises(GroundTruthError, match="tie for most locations"):
        _expected("H05", _august_rows(*more))
    twin = {"Event_ID": "ac2", **_AUGUST, "geohash": 1}
    with pytest.raises(GroundTruthError, match="2 events named AUGUST COMPLEX"):
        _expected("H05", _august_rows(twin))


def _logged_readings(log: str, qid: str) -> dict:
    """The readings dict a reference logged on its summary line."""
    line = next(x for x in log.splitlines() if f"{qid} " in x and "readings {" in x)
    return ast.literal_eval(line.split("readings ", 1)[1])


def _nw(geohash, event, year, bs, **extra):
    """A row; by default a wildfire row in eco1 = 6 (a target row from 1995)."""
    row = {"geohash": geohash, "Event_ID": event, "year": year, "bs": bs}
    return {"Ig_Date": _day(year), "Incid_Type": 1, "eco1": 6, **row, **extra}


_REBURN_ROWS = [
    _nw(1, "t1", 2000, 4),
    _nw(1, "p1", 1990, 3, Incid_Type=2),  # prescribed, before 1995
    _nw(2, "t2", 2005, 2),
    _nw(2, "t2a", 1998, 3),  # itself a target row, with nothing before it
    _nw(3, "t3", 2010, 4),
    _nw(3, "p3", 1985, None),  # null bs counts as a row
    _nw(-4, "t4", 2003, 1),  # a negative geohash (partition 12)
    _nw(-4, "p4", 2001, 6, eco1=7),  # bs 6, outside the region
    _nw(5, "t5", 2012, 4),
    _nw(5, "p5", 1990, 3, Incid_Type=2),
    _nw(5, "t5a", 1995, 3),
    _nw(5, "t5b", 2000, 3),
    _nw(6, "t6", 2015, 3),
    _nw(6, "p6", 2015, 3, Incid_Type=2),  # same Ig_Date: not earlier
    _nw(7, "t7", 1996, 4),
    _nw(8, "t8", 2020, 2),
    _nw(8, "p8", 1984, 1, Incid_Type=0),
    _nw(9, "n9", 2005, 5),  # bs 5: not a target row
    _nw(10, "n10", 1994, 4),  # before 1995: not a target row
    _nw(10, "n10b", 2001, 4, Incid_Type=2),  # prescribed: not a target row
]


def test_reburn_severity_difference(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    # Reburns t1 t2 t3 t4 t5 t5a t5b t8: 3 of 8 at bs 4; others t2a t6 t7: 1 of 3.
    assert _expected("H09", _REBURN_ROWS) == pytest.approx(3 / 8 * 100 - 1 / 3 * 100)
    readings = {
        # wildfire-only: t2 t3 t4 t5 t5b reburn (2 of 5); 2 of 6 others
        "wildfire-only lookback": 2 / 5 - 2 / 6,
        # within the target rows: t2 t5 t5b (1 of 3); 3 of 8 others
        "filtered-set lookback": 1 / 3 - 3 / 8,
        # from 1995 on: t2 t4 t5 t5b (1 of 4); 3 of 7 others
        "1995-only lookback": 1 / 4 - 3 / 7,
        # bs 2-4 rows only: t1 t2 t5 t5a t5b (2 of 5); 2 of 6 others
        "bs 2-4 lookback": 2 / 5 - 2 / 6,
        # each reburn row once per earlier row: t5 3 times, t5b twice
        "row-duplicating join": 5 / 11 - 1 / 3,
    }
    assert _logged_readings(caplog.text, "H09") == pytest.approx(
        {name: value * 100 for name, value in readings.items()}
    )


def test_reburn_severity_refuses_an_empty_group():
    firsts = [r for r in _REBURN_ROWS if r["geohash"] == 7]
    with pytest.raises(GroundTruthError, match=r"an empty group \(reburn 0, other 1\)"):
        _expected("H09", firsts)


def _location(geohash, bss, start=1990, event=None):
    """Rows of one location: one fire per bs value, a year apart."""
    return [
        {
            "geohash": geohash,
            "Event_ID": f"{event or geohash}-{i}",
            "Ig_Date": _day(start + i),
            "bs": bs,
        }
        for i, bs in enumerate(bss)
    ]


# (bs sequence, locations, pairs from bs 4, of those bs 4 again)
_SEQUENCES = [
    ((4, 4), 30, 1, 1),
    ((4, 3), 40, 1, 0),
    ((4, None, 4), 20, 1, 0),  # bs 2-4 and 1-4 readings see 4 -> 4
    ((4, 1, 4), 20, 1, 0),  # bs 2-4 reading sees 4 -> 4
    ((4, 6), 15, 1, 0),  # bs 1-4 and 2-4 readings see no pair
    ((2, 4, 4, 4), 10, 2, 2),
    ((3, 2), 30, 0, 0),
]


def _pairs_rows():
    rows, geohash = [], 0
    for bss, n, _, _ in _SEQUENCES:
        for _ in range(n):
            geohash += 1
            rows += _location(geohash, bss)
    return rows


def test_high_severity_after_high_severity(caplog):
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    rows = _pairs_rows()
    # Two fires on one date after a bs 4 fire, both bs 4: either order gives
    # the same pairs, and 1 same-date pair of 227 is under 0.5%.
    rows += _location(-1, [4], start=2000, event="tie") + [
        {"geohash": -1, "Event_ID": e, "Ig_Date": _day(2001), "bs": 4}
        for e in ("tie-a", "tie-b")
    ]
    from_high = sum(n * f for _, n, f, _ in _SEQUENCES) + 2
    again = sum(n * a for _, n, _, a in _SEQUENCES) + 2
    assert _expected("H03", rows) == pytest.approx(again / from_high * 100)
    assert "227 pairs, 147 from bs 4, 52 bs 4 again" in caplog.text
    assert "same-date pairs 0.441%" in caplog.text
    assert _logged_readings(caplog.text, "H03") == pytest.approx(
        {
            "same-date ties reversed": 52 / 147 * 100,
            "bs 2-4 fires": 92 / 132 * 100,
            "bs 1-4 fires": 72 / 132 * 100,
            "every later fire": 103 / 198 * 100,
        }
    )


def test_high_severity_pairs_refuse_ties_and_inside_readings():
    rows = _pairs_rows()

    def same_day(geohash, bss):
        return [
            {
                "geohash": geohash,
                "Event_ID": f"{geohash}-tie-{i}",
                "Ig_Date": _day(2001),
                "bs": b,
            }
            for i, b in enumerate(bss)
        ]

    # 2 same-date pairs of 227: 0.88%.
    with pytest.raises(GroundTruthError, match="0.88% of consecutive pairs share"):
        _expected("H03", rows + same_day(-1, [4, 3]) + same_day(-2, [3, 3]))
    # One same-date pair after a bs 4 fire, with different bs: the order of
    # the two fires moves the answer 0.44 pp.
    flips = _location(-1, [4], start=2000, event="first") + same_day(-1, [4, 3])
    with pytest.raises(GroundTruthError, match="tie order moves the answer 0.44"):
        _expected("H03", rows + flips)
    # Every fire bs 2-4 and no 1, 5, 6 or null: the bs 2-4 reading is the answer.
    plain = [r for r in rows if r["bs"] in (2, 3, 4)]
    with pytest.raises(GroundTruthError, match="reading 'bs 2-4 fires'"):
        _expected("H03", plain)
    with pytest.raises(GroundTruthError, match="two rows at one location"):
        _expected("H03", rows + [rows[0]])
    with pytest.raises(GroundTruthError, match="no location has two fire events"):
        _expected("H03", [r for r in rows if r["Event_ID"].endswith("-0")])


def test_high_severity_pairs_log_a_close_bs_1_4_reading(caplog):
    # Only a bs null, 5 or 6 fire after a bs 4 one tells the bs 1-4 reading
    # from the answer; with one such location the reading lands within 0.5 pp
    # (45.45% against 45.05%) and is only logged.
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    kept = {(4, None, 4): 0, (4, 6): 1}
    rows, geohash = [], 0
    for bss, n, _, _ in _SEQUENCES:
        for _ in range(kept.get(bss, n)):
            geohash += 1
            rows += _location(geohash, bss)
    expected = _expected("H03", rows)
    readings = _logged_readings(caplog.text, "H03")
    assert expected == pytest.approx(50 / 111 * 100)
    assert abs(readings["bs 1-4 fires"] - expected) < 0.5
    assert abs(readings["bs 2-4 fires"] - expected) > 0.5


def test_high_severity_pairs_log_a_bs_1_4_reading_without_pairs(caplog):
    # The logged-only reading never refuses: with no pair from a bs = 4 fire
    # (forced here by reading only bs = 1 fires) it is logged as None.
    caplog.set_level(logging.INFO, logger="bench.ground_truth")
    swap = (
        'ordered.filter(pl.col("bs").is_between(1, 4))',
        'ordered.filter(pl.col("bs") == 1)',
    )
    expected = _expected("H03", _pairs_rows(), swap=swap)
    assert expected == pytest.approx(_expected("H03", _pairs_rows()))
    assert _logged_readings(caplog.text, "H03")["bs 1-4 fires"] is None


# ---------------------------------------------------------------------------
# the 1/1000-location sample (every row at geohash % 1000 == 0)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE.exists(), reason="no .eval-runs/sample.parquet")
def test_references_on_the_sample():
    from app.query_engine import _build_lazyframe

    lf = _build_lazyframe(SAMPLE)

    def run(qid, swap=None):
        q = BY_ID[qid]
        if swap:
            q = dataclasses.replace(q, reference_code=q.reference_code.replace(*swap))
        return compute_expected(q, lf)[0]

    assert run("H01") == {"state": "FL", "count": 1388}
    assert run("H02") == pytest.approx(508.1373)
    assert run("H04", (f"YEAR = {COTTONWOOD_YEAR}\n", "YEAR = 1994\n"))["count"] == 3
    assert run("H05") == {
        "percent": pytest.approx(24.6775, abs=1e-4),
        "name": "YOLLA BOLLY COMPLEX (IRON)",
        "year": 2008,
    }
    assert run("H06", (f"N = {ECO8_2020_WILDFIRE_EVENTS}\n", "N = 24\n")) == 9
    assert run("H07") == pytest.approx(3.539474, abs=1e-6)
    assert run("H08") == pytest.approx(1.406326, abs=1e-6)
    assert run("H09") == pytest.approx(-11.0590, abs=1e-4)
    # The bs 1-4 reading (16.955%) is 0.40 pp away: logged, not refused.
    assert run("H03") == pytest.approx(16.5531, abs=1e-4)

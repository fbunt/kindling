"""What SYSTEM_INSTRUCTION tells the model about execution and encodings."""

import re

from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION


def test_prompt_requires_streaming_engine():
    assert '.collect(engine="streaming")' in SYSTEM_INSTRUCTION
    assert "will time out" not in SYSTEM_INSTRUCTION
    # No example may show a bare collect the model could copy.
    assert ".collect()" not in SYSTEM_INSTRUCTION
    # The sampling-disclosure rule stays.
    assert "Never present a sampled result as if it covered the whole dataset" in (
        SYSTEM_INSTRUCTION
    )
    run_query = next(
        f for f in FIRE_DATA_TOOLS.function_declarations if f.name == "run_query"
    )
    assert 'engine="streaming"' in run_query.parameters.properties["code"].description


def test_prompt_states_row_cap():
    assert "at most 100 rows" in SYSTEM_INSTRUCTION
    assert "`truncated`" in SYSTEM_INSTRUCTION
    assert "Never present a truncated table as complete" in SYSTEM_INSTRUCTION


def test_prompt_does_not_claim_eco2_eco3_mappings():
    # Only the columns query_engine actually maps are named as mapped.
    m = re.search(
        r"authoritative integer-to-label mapping for ([^.]*)\.", SYSTEM_INSTRUCTION
    )
    assert m
    mapped = m.group(1)
    assert "`eco1`" in mapped and "`nlcd`" in mapped
    assert "eco2" not in mapped and "eco3" not in mapped
    assert "`eco2` and `eco3` have NO name mapping" in SYSTEM_INSTRUCTION
    assert "web_search" in SYSTEM_INSTRUCTION


def test_prompt_examples_deduplicate_fire_level_stats():
    # Fire-level examples must roll up per Event_ID (rows are pixels).
    assert '.unique("Event_ID").sort("area_m2", descending=True)' in (
        SYSTEM_INSTRUCTION
    )
    assert (
        'lf.group_by("Event_ID").agg(pl.col("year", "area_m2").first())'
        in SYSTEM_INSTRUCTION
    )
    assert 'lf.group_by("year").agg(pl.col("area_m2").sum())' not in (
        SYSTEM_INSTRUCTION
    )


def test_prompt_plot_feedback_claim_is_accurate():
    # chat_loop never feeds a plot back within the turn; routes/chat.py
    # re-embeds recent plots in history on later turns.
    assert "returned to you in conversation history for review" not in (
        SYSTEM_INSTRUCTION
    )
    assert "cannot see a plot's image while answering" in SYSTEM_INSTRUCTION

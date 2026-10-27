"""What SYSTEM_INSTRUCTION tells the model about execution and encodings."""

import re
from pathlib import Path

from app.query_engine import _KEY_COLUMNS
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


def test_prompt_names_ecoregions_through_the_lookup():
    # get_dataset_info maps only the non-ecoregion codes; names come from the
    # preloaded eco_names frame, whose every column the prompt lists.
    m = re.search(
        r"authoritative integer-to-label mapping for ([^.]*)\.", SYSTEM_INSTRUCTION
    )
    assert m
    mapped = m.group(1)
    assert "`nlcd`" in mapped and "eco" not in mapped
    csv = Path(__file__).resolve().parents[1] / "app/sandbox/ecoregions.csv"
    header = csv.read_text().splitlines()[0]
    for column in header.split(","):
        assert f"`{column}`" in SYSTEM_INSTRUCTION, column
    assert "`eco_names`" in SYSTEM_INSTRUCTION
    assert "eco2s" not in SYSTEM_INSTRUCTION and "eco1s" not in SYSTEM_INSTRUCTION


def test_prompt_lists_every_dataset_column():
    # The statistics table names each column the key-column descriptions cover.
    for column in _KEY_COLUMNS:
        assert f"| {column} |" in SYSTEM_INSTRUCTION, column


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

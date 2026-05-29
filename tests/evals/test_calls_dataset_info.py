"""Eval: model should consult get_dataset_info for Incid_Type questions and
should not claim Incid_Type=2 is Wildland Fire Use.
"""

import pytest

from tests.evals.judge import judge

PROMPTS = [
    "How many Wildland Fire Use events per year?",
    "Plot prescribed-fire-vs-wildfire trends since 2000.",
    "What's the breakdown of Incid_Type 1 vs 2 vs 3?",
    "Compare burn severity across ecoregions.",
]

N_TRIALS = 5
PASS_THRESHOLD = 4  # need >= 4/5 per variant

OUTCOME_CRITERION = (
    "The response does NOT claim that Incid_Type=2 is Wildland Fire Use. "
    "Saying Incid_Type=2 is Prescribed Fire, or not mentioning the specific "
    "Incid_Type=2 mapping at all, both satisfy the criterion. "
    "Only answer 'no' if the response explicitly equates Incid_Type=2 with "
    "Wildland Fire Use."
)


@pytest.mark.model_eval
@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p[:40])
@pytest.mark.asyncio
async def test_calls_dataset_info_and_correct_mapping(prompt, run_turn, genai_client):
    behavioral = []
    outcome = []
    for trial in range(N_TRIALS):
        result = await run_turn(prompt, trial)
        called = any(tc["name"] == "get_dataset_info" for tc in result.tool_calls)
        verdict = judge(
            genai_client,
            response_text=result.text,
            criterion=OUTCOME_CRITERION,
        )
        behavioral.append(called)
        outcome.append(verdict)

    b_passes = sum(behavioral)
    o_passes = sum(outcome)
    assert b_passes >= PASS_THRESHOLD, (
        f"behavioral: {b_passes}/{N_TRIALS} trials called get_dataset_info "
        f"for prompt {prompt!r} (trace per trial in .eval-runs/)"
    )
    assert o_passes >= PASS_THRESHOLD, (
        f"outcome: {o_passes}/{N_TRIALS} trials avoided the wrong claim "
        f"for prompt {prompt!r} (trace per trial in .eval-runs/)"
    )

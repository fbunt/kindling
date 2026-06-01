"""Eval: the model must not claim Incid_Type=2 is Wildland Fire Use.

We dropped the behavioral assertion (whether the model calls get_dataset_info)
because the Numeric encodings section of SYSTEM_INSTRUCTION now states the
Incid_Type=3 -> WFU mapping directly, so the correct, non-redundant behavior
is to skip the tool. The trial JSON dumps still record the tool-call trace
for inspection.
"""

import pytest

from tests.evals.conftest import append_trial_fields
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
async def test_correct_incid_type_mapping(prompt, run_turn, genai_client):
    outcome = []
    for trial in range(N_TRIALS):
        result, trial_path = await run_turn(prompt, trial)
        verdict = judge(
            genai_client,
            response_text=result.text,
            criterion=OUTCOME_CRITERION,
        )
        append_trial_fields(
            trial_path,
            outcome_criterion=OUTCOME_CRITERION,
            outcome_verdict=verdict,
        )
        outcome.append(verdict)

    o_passes = sum(outcome)
    assert o_passes >= PASS_THRESHOLD, (
        f"outcome: {o_passes}/{N_TRIALS} trials avoided the wrong claim "
        f"for prompt {prompt!r} (trace per trial in .eval-runs/)"
    )

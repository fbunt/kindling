"""Eval: the budget-exhausted final call is accepted by the real API and answers.

When run_chat_turn uses up its tool rounds it makes one more generate_content
call with tools disabled (FunctionCallingConfig(mode="NONE")) and a user
instruction appended directly after the user function-response turn, i.e. two
consecutive user turns. tests/test_chat_loop.py covers the request shape with a
fake client; this checks that real Gemini accepts it (no exception) and returns
real text rather than an empty reply that trips the fallback.

Exhaustion is forced with MAX_ROUNDS=1: any tool call in the first round
exhausts the budget. The prompt needs an Incid_Type lookup plus several
aggregations, so a no-tool first round is unlikely; if it happens the trial is
retried once with a harder prompt, then fails (never passes silently).

API calls per run: 2 models x 2 trials, each 2 chat calls (tool round + final
call) + 1 flash-lite code-judge per run_query in that round (0 if the round is
only get_dataset_info) + 1 flash-lite judge, so roughly 12-16 calls (more if a
round issues parallel queries or a trial is retried).
"""

import pytest

from app.config import CHAT_MODELS
from tests.evals.conftest import append_trial_fields
from tests.evals.judge import judge

MAX_ROUNDS = 1
N_TRIALS = 2

PROMPT = (
    "For each decade since the 1980s, compare the total burned area of "
    "wildfires versus prescribed fires, then compute the wildfire-to-prescribed "
    "ratio per decade and tell me whether that ratio is trending up or down."
)
# Used only if PROMPT gets answered without any tool call.
HARDER_PROMPT = (
    "Using the dataset, not general knowledge: first check the Incid_Type "
    "encoding with get_dataset_info, then query total burned area per year by "
    "Incid_Type, then per decade, then compute the wildfire-to-prescribed ratio "
    "per decade and fit a linear trend to it. Report the exact numbers."
)

# Leading text of run_chat_turn's fallback when the loop-exhaustion final call
# returns no text (inline in app/chat_loop.py; tests/test_chat_loop.py matches
# the same prefix).
FALLBACK_PREFIX = "I ran out of tool-use rounds"

OUTCOME_CRITERION = (
    "The response either answers the user's question (fully or partly) from "
    "data results, or says that some of the requested work is unfinished, "
    "unverified, or could not be completed. Answer 'no' only if the response "
    "is empty, off-topic, or only asks the user to retry."
)


@pytest.mark.model_eval
@pytest.mark.parametrize(
    "model", [m["id"] for m in CHAT_MODELS], ids=[m["id"] for m in CHAT_MODELS]
)
async def test_exhausted_final_call_answers(model, run_turn, genai_client):
    verdicts = []
    trace_idx = 0
    for trial in range(N_TRIALS):
        for prompt in (PROMPT, HARDER_PROMPT):
            # An API rejection of the final call raises here and fails the test.
            result, trial_path = await run_turn(
                prompt, trace_idx, max_rounds=MAX_ROUNDS, model=model
            )
            trace_idx += 1
            if result.loop_exhausted:
                break
        assert result.loop_exhausted, (
            f"trial {trial}: {model} answered both prompts without a tool call, "
            f"so the exhausted path never ran (traces in .eval-runs/)"
        )
        assert result.text.strip(), f"trial {trial}: empty final text"
        assert not result.text.startswith(FALLBACK_PREFIX), (
            f"trial {trial}: final no-tools call returned no text, got the "
            f"fallback instead: {result.text!r}"
        )

        verdict = judge(
            genai_client, response_text=result.text, criterion=OUTCOME_CRITERION
        )
        append_trial_fields(
            trial_path,
            outcome_criterion=OUTCOME_CRITERION,
            outcome_verdict=verdict,
        )
        verdicts.append(verdict)

    # Lenient: the hard assertions above are the point; the judge only has to
    # agree once that the answer engages with the unfinished work or results.
    assert any(verdicts), (
        f"judge rejected every exhausted-turn answer from {model} "
        f"(traces in .eval-runs/)"
    )

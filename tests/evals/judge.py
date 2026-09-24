"""Judge wrapper using the flash-lite model for outcome assertions."""

from google import genai

from app.config import LITE_MODEL

JUDGE_MODEL = LITE_MODEL


def judge(client: genai.Client, *, response_text: str, criterion: str) -> bool:
    """Ask the judge a yes/no question about a response. Returns True on 'yes'."""
    prompt = (
        "Answer ONLY 'yes' or 'no'.\n\n"
        "Response from model under test:\n"
        "---\n"
        f"{response_text}\n"
        "---\n\n"
        f"Criterion: {criterion}\n"
        "Does the response satisfy the criterion?"
    )
    r = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    verdict = (r.text or "").strip().lower()
    return verdict.startswith("yes")

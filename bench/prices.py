"""Per-model token prices for the report's dollar estimate.

USD per 1M tokens. Prices as of <date>: NOT FILLED IN -- verify against the
current Gemini API / Vertex pricing page before relying on any dollar figure.
Every value is None until then; the report prints token totals and says the
prices are missing. Output price applies to candidates + thoughts tokens
(thinking is billed as output). Tiered (long-context) pricing is not modelled.
"""

PRICES_AS_OF: str | None = None

# model id -> {"input": $/1M, "cached_input": $/1M, "output": $/1M}
PRICES: dict[str, dict[str, float | None]] = {
    "gemini-3.1-pro-preview": {"input": None, "cached_input": None, "output": None},
    "gemini-3.8-flash": {"input": None, "cached_input": None, "output": None},
    "gemini-3.5-flash-lite": {"input": None, "cached_input": None, "output": None},
}


def estimate_usd(model: str, usage: dict) -> float | None:
    """Dollar cost of one model's summed usage, or None if any price is unset."""
    price = PRICES.get(model)
    if not price or any(v is None for v in price.values()):
        return None
    prompt = usage.get("prompt_token_count", 0) + usage.get(
        "tool_use_prompt_token_count", 0
    )
    cached = usage.get("cached_content_token_count", 0)
    output = usage.get("candidates_token_count", 0) + usage.get(
        "thoughts_token_count", 0
    )
    return (
        (prompt - cached) * price["input"]
        + cached * price["cached_input"]
        + output * price["output"]
    ) / 1e6

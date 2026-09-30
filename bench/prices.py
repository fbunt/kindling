"""Per-model token prices for the report's dollar estimate.

USD per 1M tokens, Gemini Developer API paid tier (Standard), from
https://ai.google.dev/gemini-api/docs/pricing (page last updated 2026-09-24).
Output price applies to candidates + thoughts tokens (thinking is billed as
output). Context-cache storage ($/1M tokens/hour) is not modelled; the app
uses no explicit cache, so cached tokens are implicit-cache hits only.

Two things the estimate does not model:
- 3.1 Pro's long-context tier: prompts over LONG_CONTEXT_THRESHOLD tokens bill
  at $4 in / $18 out / $0.40 cached. The report counts calls over the
  threshold so an undercount is visible.
- 3.8 Flash's price doubles on 2027-01-01 ($1.50 in / $7.50 out / $0.15
  cached). Update this table for any run after that date.
"""

PRICES_AS_OF: str | None = "2026-09-24"

# Prompt size (per call) above which TIERED_MODELS bill at a higher rate.
LONG_CONTEXT_THRESHOLD = 200_000
TIERED_MODELS: frozenset[str] = frozenset({"gemini-3.1-pro-preview"})

# model id -> {"input": $/1M, "cached_input": $/1M, "output": $/1M}
PRICES: dict[str, dict[str, float | None]] = {
    "gemini-3.1-pro-preview": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gemini-3.8-flash": {"input": 0.75, "cached_input": 0.075, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "cached_input": 0.03, "output": 2.50},
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


def long_context_calls(calls: list[dict]) -> int:
    """Calls to a tiered model whose prompt exceeded LONG_CONTEXT_THRESHOLD."""
    n = 0
    for c in calls:
        if c.get("model") not in TIERED_MODELS or "usage" not in c:
            continue
        u = c["usage"]
        prompt = (u.get("prompt_token_count") or 0) + (
            u.get("tool_use_prompt_token_count") or 0
        )
        n += prompt > LONG_CONTEXT_THRESHOLD
    return n

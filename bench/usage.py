"""Bench-only genai client proxy: per-call token usage, finish reason, latency.

RecordingClient wraps the real client and is passed to run_chat_turn in its
place, so the production chat loop, code-judge and system prompt run unchanged
while every generate_content call (chat rounds and code-judge verdicts alike)
is logged into the sink of the trial that made it.
"""

import time

USAGE_FIELDS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "tool_use_prompt_token_count",
    "total_token_count",
)

# A final chat response with one of these finish reasons is a model-caused turn
# failure (the chat loop would otherwise paper over it with fallback text).
MODEL_FAILURE_FINISH = frozenset(
    {
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
    }
)
# ...and these only when the response also carried no text. Any other finish
# reason (STOP included) is a failure when the response carried neither text
# nor a function call: an empty answer.
EMPTY_FAILURE_FINISH = frozenset({"MAX_TOKENS", "OTHER", "LANGUAGE"})


def _enum_name(value) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", None) or str(value)


def describe_response(response) -> dict:
    """The per-call facts worth keeping from a GenerateContentResponse."""
    candidates = response.candidates or []
    first = candidates[0] if candidates else None
    content = first.content if first is not None else None
    parts = (content.parts or []) if content is not None else []
    meta = response.usage_metadata
    feedback = response.prompt_feedback
    return {
        "model_version": response.model_version,
        "finish_reason": _enum_name(first.finish_reason) if first else None,
        "block_reason": _enum_name(feedback.block_reason) if feedback else None,
        "n_candidates": len(candidates),
        "has_content": content is not None,
        "has_text": any(getattr(p, "text", None) for p in parts),
        "n_function_calls": sum(1 for p in parts if getattr(p, "function_call", None)),
        "usage": {
            f: (getattr(meta, f, None) or 0) if meta is not None else 0
            for f in USAGE_FIELDS
        },
    }


class _RecordingModels:
    def __init__(self, owner: "RecordingClient"):
        self._owner = owner

    def generate_content(self, **kwargs):
        owner = self._owner
        sink = owner.sink  # bound at call start: a late thread can't leak
        model = kwargs.get("model")
        entry = {
            "model": model,
            "purpose": "chat" if model == owner.chat_model else "judge",
            "attempt": owner.attempt,
        }
        t0 = time.monotonic()
        try:
            response = owner.inner.models.generate_content(**kwargs)
        except Exception as e:
            entry["elapsed_s"] = round(time.monotonic() - t0, 3)
            entry["error"] = f"{type(e).__name__}: {e}"[:500]
            entry["error_code"] = getattr(e, "code", None)
            sink.append(entry)
            raise
        entry["elapsed_s"] = round(time.monotonic() - t0, 3)
        entry.update(describe_response(response))
        sink.append(entry)
        return response

    def __getattr__(self, name):
        return getattr(self._owner.inner.models, name)


class RecordingClient:
    """Duck-types genai.Client for run_chat_turn / execute_function_call_async.

    Set `sink` (a fresh list), `chat_model` and `attempt` before each attempt."""

    def __init__(self, inner):
        self.inner = inner
        self.sink: list[dict] = []
        self.chat_model: str | None = None
        self.attempt = 0
        self.models = _RecordingModels(self)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def chat_calls(calls: list[dict]) -> list[dict]:
    return [c for c in calls if c.get("purpose") == "chat"]


def terminal_model_error(calls: list[dict]) -> str | None:
    """Model-caused failure of a completed turn, judged from its LAST chat call:
    no candidate / no content / a blocked prompt, a failure finish reason, or
    an empty response (no text and no function call, whatever the reason)."""
    chats = [c for c in chat_calls(calls) if "error" not in c]
    if not chats:
        return None
    last = chats[-1]
    reason = last.get("finish_reason")
    if last.get("block_reason"):
        return f"prompt blocked: {last['block_reason']}"
    if not last.get("n_candidates"):
        return "no candidates in final response"
    if reason in MODEL_FAILURE_FINISH:
        return f"final response finish_reason={reason}"
    if not last.get("has_content"):
        return f"final response has no content (finish_reason={reason})"
    if not last.get("has_text") and (
        reason in EMPTY_FAILURE_FINISH or not last.get("n_function_calls")
    ):
        return f"final response empty (finish_reason={reason})"
    return None


def call_anomalies(calls: list[dict]) -> list[str]:
    """Every non-STOP / contentless / empty chat call, terminal or not
    (informational)."""
    out = []
    for i, c in enumerate(chat_calls(calls)):
        if "error" in c:
            continue
        reason = c.get("finish_reason")
        if reason not in (None, "STOP") or not c.get("has_content"):
            out.append(f"chat call {i}: finish_reason={reason}")
        elif not c.get("has_text") and not c.get("n_function_calls"):
            out.append(f"chat call {i}: empty (finish_reason={reason})")
    return out


def _empty_totals() -> dict:
    return {"calls": 0, **{f: 0 for f in USAGE_FIELDS}}


def aggregate_usage(calls: list[dict]) -> dict[str, dict]:
    """{model: {calls, <usage fields summed>}} over successful calls."""
    out: dict[str, dict] = {}
    for c in calls:
        if "usage" not in c:
            continue
        tot = out.setdefault(c.get("model") or "?", _empty_totals())
        tot["calls"] += 1
        for f in USAGE_FIELDS:
            tot[f] += int(c["usage"].get(f) or 0)
    return out


def merge_usage(*aggregates: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for agg in aggregates:
        for model, tot in (agg or {}).items():
            acc = out.setdefault(model, _empty_totals())
            for k in acc:
                acc[k] += int(tot.get(k) or 0)
    return out


def total_tokens(agg: dict[str, dict]) -> int:
    return sum(t.get("total_token_count", 0) for t in (agg or {}).values())

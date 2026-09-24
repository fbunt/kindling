"""Single source of truth for model names and turn limits.

Every model string in the app, bench, and evals imports from here. The chat
model list is served to the frontend via GET /api/config and doubles as the
allowlist for the `model` field on POST /api/chat.
"""

# Ordered: the first entry is the default the UI shows.
CHAT_MODELS: list[dict[str, str]] = [
    {"id": "gemini-3.1-pro-preview", "label": "3.1 Pro Preview"},
    {"id": "gemini-3.8-flash", "label": "3.8 Flash"},
]
CHAT_MODEL_IDS: frozenset[str] = frozenset(m["id"] for m in CHAT_MODELS)
DEFAULT_CHAT_MODEL = "gemini-3.1-pro-preview"
assert DEFAULT_CHAT_MODEL in CHAT_MODEL_IDS

# Cheap classifier model: prompt-guard, code-judge, key validation, eval judge.
# Pinned deliberately (not an alias) -- see PROGRESS.md 2026-09-22.
LITE_MODEL = "gemini-3.5-flash-lite"

# Max Gemini/tool round-trips per chat turn (POST /api/chat).
MAX_TOOL_ROUNDS = 20

"""Central google-genai client factory.

One place decides which backend every client talks to, controlled by the
KINDLING_USE_VERTEX env toggle:

- unset / false: Gemini Developer API (generativelanguage.googleapis.com). Uses a
  standard `AIza…` Developer API key.
- true: Vertex AI **express mode** (aiplatform.googleapis.com). Uses a Vertex
  express API key (`AQ.…`). Note `models.list()` is NOT supported with an API key
  under Vertex express mode (it 401s) — validate keys with a `generate_content`
  call instead (see app/routes/auth.py).

Both backends authenticate with an API key passed by the caller; only the
endpoint and key type differ.
"""

import os

from google import genai

_TRUTHY = {"1", "true", "yes", "on"}


def use_vertex() -> bool:
    """True if the app should route to Vertex AI instead of the Developer API."""
    return os.environ.get("KINDLING_USE_VERTEX", "").strip().lower() in _TRUTHY


def make_client(api_key: str) -> genai.Client:
    """Build a genai client for the configured backend."""
    if use_vertex():
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(api_key=api_key)

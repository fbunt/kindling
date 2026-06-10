"""Server-side API-key store.

Starlette's SessionMiddleware is a client-side signed (NOT encrypted) cookie,
so the Gemini key must never go in the session itself. Keys live in this
in-memory dict; the session cookie carries only the opaque token. Entries are
dropped on re-login/logout; a server restart clears the store, which just sends
the client back through /api/auth/status (env-key fallback or login screen).
"""

import secrets

_KEYS: dict[str, str] = {}


def put_key(api_key: str) -> str:
    """Store a key and return a fresh opaque token for the session cookie."""
    token = secrets.token_urlsafe(32)
    _KEYS[token] = api_key
    return token


def get_key(token: str | None) -> str | None:
    return _KEYS.get(token) if token else None


def drop_key(token: str | None) -> None:
    _KEYS.pop(token, None)

import asyncio
import os

from fastapi import APIRouter, HTTPException, Request
from google.genai import types
from pydantic import BaseModel

from app.config import LITE_MODEL
from app.genai_client import make_client
from app.keystore import drop_key, get_key, put_key

router = APIRouter()

# Validate keys with a tiny generation, NOT models.list(): under Vertex express
# mode list() rejects API keys with 401 UNAUTHENTICATED, but generate_content
# works (and works on the Developer API too). flash-lite keeps it cheap.
_VALIDATION_MODEL = LITE_MODEL


def _validate_key(api_key: str) -> None:
    """Raise if the key can't make a real call (auth/quota/etc.)."""
    client = make_client(api_key)
    client.models.generate_content(
        model=_VALIDATION_MODEL,
        contents="ping",
        config=types.GenerateContentConfig(max_output_tokens=16),
    )


class AuthRequest(BaseModel):
    api_key: str


@router.post("/auth")
async def authenticate(req: AuthRequest, request: Request):
    try:
        # to_thread: a sync Gemini round-trip here would stall the event loop
        # (and every concurrent SSE stream) for the network call's duration.
        await asyncio.to_thread(_validate_key, req.api_key)
    except Exception as e:
        return {"ok": False, "error": f"Invalid API key: {e}"}

    drop_key(request.session.get("token"))
    request.session["token"] = put_key(req.api_key)
    return {"ok": True}


@router.get("/auth/status")
async def auth_status(request: Request):
    # Read-only: never mints a token or touches the session. Using the
    # server's GEMINI_API_KEY is an explicit POST /auth/env, so a cookieless
    # GET cannot spend the operator's key and logout actually logs out.
    # No re-validation either: the key was validated at login (POST /auth)
    # and a revoked key surfaces as an error on the next chat turn. Keeps page
    # loads free of paid LLM calls and immune to transient quota/network
    # failures logging the user out.
    return {
        "authenticated": bool(get_key(request.session.get("token"))),
        "env_key_available": bool(os.environ.get("GEMINI_API_KEY")),
    }


@router.post("/auth/env")
async def use_env_key(request: Request):
    """Log in with the server's GEMINI_API_KEY (explicit user action)."""
    env_key = os.environ.get("GEMINI_API_KEY")
    if not env_key:
        raise HTTPException(status_code=404, detail="No server API key configured")
    # Not validated with a paid call: the env key is the operator's own and
    # was never validated on this path before either; a bad key surfaces as an
    # error on the first chat turn.
    drop_key(request.session.get("token"))
    request.session["token"] = put_key(env_key)
    return {"ok": True}


@router.post("/auth/logout")
async def logout(request: Request):
    drop_key(request.session.get("token"))
    request.session.clear()
    return {"ok": True}

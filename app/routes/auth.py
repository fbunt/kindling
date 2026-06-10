import asyncio
import os

from fastapi import APIRouter, Request
from google.genai import types
from pydantic import BaseModel

from app.genai_client import make_client
from app.keystore import drop_key, get_key, put_key

router = APIRouter()

# Validate keys with a tiny generation, NOT models.list(): under Vertex express
# mode list() rejects API keys with 401 UNAUTHENTICATED, but generate_content
# works (and works on the Developer API too). flash-lite keeps it cheap.
_VALIDATION_MODEL = "gemini-3.1-flash-lite-preview"


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
    api_key = get_key(request.session.get("token"))
    if not api_key:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            request.session["token"] = put_key(env_key)
            return {"authenticated": True}
        return {"authenticated": False}

    try:
        await asyncio.to_thread(_validate_key, api_key)
    except Exception:
        return {"authenticated": False}
    return {"authenticated": True}


@router.post("/auth/logout")
async def logout(request: Request):
    drop_key(request.session.get("token"))
    request.session.clear()
    return {"ok": True}

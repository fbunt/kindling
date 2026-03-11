import os

from fastapi import APIRouter, Request
from pydantic import BaseModel
from google import genai

router = APIRouter()


class AuthRequest(BaseModel):
    api_key: str


@router.post("/auth")
async def authenticate(req: AuthRequest, request: Request):
    try:
        client = genai.Client(api_key=req.api_key)
        # Validate key with a lightweight call
        next(iter(client.models.list()))
    except Exception as e:
        return {"ok": False, "error": f"Invalid API key: {e}"}

    request.session["api_key"] = req.api_key
    return {"ok": True}


@router.get("/auth/status")
async def auth_status(request: Request):
    api_key = request.session.get("api_key")
    if not api_key:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            request.session["api_key"] = env_key
            return {"authenticated": True}
        return {"authenticated": False}

    try:
        client = genai.Client(api_key=api_key)
        next(iter(client.models.list()))
    except Exception:
        return {"authenticated": False}
    return {"authenticated": True}


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}

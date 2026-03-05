import os

from fastapi import APIRouter, Request
from pydantic import BaseModel
from google import genai

router = APIRouter()


def _list_models(client):
    models = []
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            models.append(m.name)
    models.sort()
    return models


class AuthRequest(BaseModel):
    api_key: str


@router.post("/auth")
async def authenticate(req: AuthRequest, request: Request):
    try:
        client = genai.Client(api_key=req.api_key)
        models = _list_models(client)
    except Exception as e:
        return {"ok": False, "error": f"Invalid API key: {e}"}

    request.session["api_key"] = req.api_key
    return {"ok": True, "models": models}


@router.get("/auth/status")
async def auth_status(request: Request):
    api_key = request.session.get("api_key")
    if not api_key:
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            try:
                client = genai.Client(api_key=env_key)
                models = _list_models(client)
                request.session["api_key"] = env_key
                return {"authenticated": True, "models": models}
            except Exception:
                pass
        return {"authenticated": False, "models": []}

    try:
        client = genai.Client(api_key=api_key)
        models = _list_models(client)
    except Exception:
        return {"authenticated": False, "models": []}
    return {"authenticated": True, "models": models}


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}

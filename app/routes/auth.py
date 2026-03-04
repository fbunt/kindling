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
        models = _list_models(client)
    except Exception as e:
        return {"ok": False, "error": f"Invalid API key: {e}"}

    request.session["api_key"] = req.api_key
    return {"ok": True, "models": models}


@router.get("/auth/status")
async def auth_status(request: Request):
    authenticated = "api_key" in request.session
    models = []
    if authenticated:
        client = genai.Client(api_key=request.session["api_key"])
        models = _list_models(client)
    return {"authenticated": authenticated, "models": models}


def _list_models(client):
    models = []
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            models.append(m.name)
    models.sort()
    return models


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}

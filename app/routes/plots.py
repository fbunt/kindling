"""Authenticated plot serving.

Plots were previously a public StaticFiles mount; counter-based names
(plot-000.png, ...) are trivially enumerable, so serving requires the same
session as /api/chat. Same-origin <img>/fetch requests carry the cookie
automatically — the frontend needs no changes.
"""

import re

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import FileResponse

from app.keystore import get_key
from app.query_engine import PLOTS_DIR

router = APIRouter()

_NAME_RE = re.compile(r"plot-\d{3,}\.png")


@router.get("/plots/{name}")
async def serve_plot(name: str, request: Request):
    if not get_key(request.session.get("token")):
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Strict allowlist of generated names; also forecloses any traversal.
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(status_code=404)
    path = PLOTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/png")

import asyncio
import base64
import binascii
import json
import logging
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.chat_loop import (  # noqa: E402
    DoneEvent,
    RejectedEvent,
    RunningQueryEvent,
    ThinkingEvent,
    run_chat_turn,
)
from app.config import (  # noqa: E402
    CHAT_MODEL_IDS,
    CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    MAX_TOOL_ROUNDS,
)
from app.genai_client import make_client  # noqa: E402
from app.guards import guard_prompt  # noqa: E402
from app.keystore import get_key  # noqa: E402
from app.query_engine import PLOTS_DIR  # noqa: E402
from app.sandbox.pool import PLOT_EPOCH, SandboxBusy  # noqa: E402
from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION  # noqa: E402

router = APIRouter()

# Request caps. HTTP-layer checks are defense in depth (decision 0002): the
# container contains the code regardless; these bound memory and keep a stale
# tab's oversized history from stranding the user with a hung spinner.
_MAX_BODY = 48 * 1024 * 1024  # Content-Length precheck -> 413
_MAX_FORM_PART = 32 * 1024 * 1024  # request.form(max_part_size=); text parts only
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # current upload -> 413; history image -> stub
_IMAGE_MAGIC = {
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}
# Above this the oldest entries are dropped (whole turns), never a 422: a long
# session must keep working.
_MAX_HISTORY_MSGS = 400
# Mirrors plots._NAME_RE (minus .png); fullmatch before any disk access.
_PLOT_NAME_RE = re.compile(r"plot-\d{3,}")

# Image bytes (plots re-read from PLOTS_DIR, uploads re-sent by the client) are
# embedded only for the last N user+assistant turns; older refs become text
# stubs. app.js mirrors this as IMAGE_WINDOW_TURNS so it stops sending upload
# bytes the server would stub anyway. The server enforces it regardless.
HISTORY_IMAGE_WINDOW = 2

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


# --- history schema (client -> server, JSON in the `history` form field) ---
#
# Structure only: types, the role literal, required content. Every per-entry
# value that can be bad in a stale tab's history (plot name, image mime, and
# especially image data) degrades to a text stub in the helpers below, never a
# 422 -- the user cannot edit history, so a 422 would strand them until Clear.
# ImageRef.data deliberately has no max_length: a pydantic cap would 422 the
# whole array on one oversized legacy upload. The body is already bounded by
# _MAX_BODY/_MAX_FORM_PART before pydantic runs.


class ImageRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mime: str = Field(max_length=64)
    name: str = Field(default="image", max_length=200)
    data: str | None = None  # base64; absent outside the client's window


class PlotRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(max_length=64)
    epoch: str | None = None  # PLOT_EPOCH of the process that made the plot


class HistoryMsg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant"]
    content: str
    image: ImageRef | None = None
    plots: list[PlotRef] = Field(default_factory=list)
    # Legacy key from pre-2026-09-23 tabs: [{data, mime, name}]. data/mime are
    # ignored and, lacking an epoch, every legacy plot renders as a stub.
    plot_images: list[PlotRef] = Field(default_factory=list)


HISTORY_ADAPTER = TypeAdapter(list[HistoryMsg])


@router.get("/config")
async def get_config():
    # Unauthenticated on purpose: model ids/labels are not sensitive, and the
    # UI needs them before login to render the selector.
    return {"models": CHAT_MODELS, "default_model": DEFAULT_CHAT_MODEL}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _clean_label(s: str) -> str:
    """Name strings from the client go into prompt text: one line, <= 100 chars."""
    return _CONTROL_RE.sub("", s or "")[:100]


def _require_text(form, key: str) -> str | None:
    """A form field as str, None if absent; 422 if the client sent it as a file."""
    value = form.get(key)
    if value is None or isinstance(value, str):
        return value
    raise RequestValidationError(
        [
            {
                "type": "string_type",
                "loc": ("body", key),
                "msg": "Input should be a valid string",
                "input": getattr(value, "filename", None),
            }
        ]
    )


def _decode_history_image(ref: ImageRef) -> bytes | None:
    """Validated bytes of a re-sent upload, or None if unusable (caller stubs).

    The size check runs on the base64 string before decoding, so an oversized
    legacy upload costs no decode. Reads _MAX_UPLOAD_BYTES at call time.
    """
    if ref.data is None:
        return None
    if len(ref.data) > _MAX_UPLOAD_BYTES * 4 // 3 + 8:
        logger.warning("history image %r over size cap; stubbed", ref.name[:40])
        return None
    magic = _IMAGE_MAGIC.get(ref.mime)
    if magic is None:
        logger.warning("history image %r has mime %r; stubbed", ref.name[:40], ref.mime)
        return None
    try:
        raw = base64.b64decode(ref.data, validate=True)
    except (binascii.Error, ValueError):
        logger.warning("history image %r is not base64; stubbed", ref.name[:40])
        return None
    if len(raw) > _MAX_UPLOAD_BYTES or not raw.startswith(magic):
        logger.warning(
            "history image %r failed size/magic check; stubbed", ref.name[:40]
        )
        return None
    return raw


def _plot_parts(ref: PlotRef, in_window: bool) -> list[types.Part]:
    """Label + inline PNG for a plot ref, or a single text stub.

    Lookup is by name in PLOTS_DIR (never by URL: the /plots URL carries a ?t=
    cache-buster). A ref is embedded only if it is inside the window, its name
    matches the generated pattern, its epoch is this process's, and the file is
    still on disk (main.py wipes plots/ at startup; _prune_plots caps the dir).
    """
    label = _clean_label(ref.name)
    if not in_window:
        return [
            types.Part(text=f"[Generated plot: {label} (image omitted from context)]")
        ]
    if _PLOT_NAME_RE.fullmatch(ref.name) and ref.epoch == PLOT_EPOCH:
        path = PLOTS_DIR / f"{ref.name}.png"
        if path.is_file():
            return [
                types.Part(text=f"[Generated plot: {label}]"),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png", data=path.read_bytes()
                    )
                ),
            ]
    return [types.Part(text=f"[Generated plot: {label} (image no longer available)]")]


def _history_to_contents(msgs: list[HistoryMsg]) -> list[types.Content]:
    if len(msgs) > _MAX_HISTORY_MSGS:
        kept = msgs[-_MAX_HISTORY_MSGS:]
        # Keep whole turns: the kept prefix must start with a user entry.
        while kept and kept[0].role != "user":
            kept = kept[1:]
        logger.info(
            "history truncated: %d entries -> %d (cap %d)",
            len(msgs),
            len(kept),
            _MAX_HISTORY_MSGS,
        )
        msgs = kept
    cutoff = len(msgs) - 2 * HISTORY_IMAGE_WINDOW
    contents = []
    n_inline = n_stubs = 0
    for i, msg in enumerate(msgs):
        in_window = i >= cutoff
        role = "user" if msg.role == "user" else "model"
        parts = [types.Part(text=msg.content)]
        if msg.image is not None:
            name = _clean_label(msg.image.name)
            # Three outcomes: inline bytes; "unavailable" (data sent but bad);
            # "omitted" (outside the window, or the client dropped the bytes).
            has_data = in_window and msg.image.data is not None
            raw = _decode_history_image(msg.image) if has_data else None
            if raw is not None:
                parts.insert(
                    0,
                    types.Part(
                        inline_data=types.Blob(mime_type=msg.image.mime, data=raw)
                    ),
                )
                n_inline += 1
            else:
                why = "unavailable" if has_data else "omitted from context"
                parts.append(types.Part(text=f"[Attached image: {name} ({why})]"))
                n_stubs += 1
        for ref in msg.plots or msg.plot_images:
            plot_parts = _plot_parts(ref, in_window)
            if len(plot_parts) == 2:
                n_inline += 1
            else:
                n_stubs += 1
            parts.extend(plot_parts)
        contents.append(types.Content(role=role, parts=parts))
    logger.debug(
        "history contents: %d entries, %d inline images, %d stubs",
        len(msgs),
        n_inline,
        n_stubs,
    )
    return contents


@router.post("/chat")
async def chat(request: Request):
    api_key = get_key(request.session.get("token"))
    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # File parts are spooled before any size check is possible, so bound the
    # whole body up front (browsers always send Content-Length for FormData).
    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > _MAX_BODY:
        raise HTTPException(
            status_code=413,
            detail="Request too large; clear the chat or remove attached images.",
        )

    # The form is read by hand (no Form()/File() params) so max_part_size can be
    # raised above Starlette's 1 MB default. An oversized text part surfaces
    # inside request.form() as Starlette's own HTTPException(400); rewrap it
    # with an actionable message. Never catch/raise MultiPartException here: it
    # is only converted to 400 inside request.form(), elsewhere it is a 500.
    try:
        form = await request.form(max_part_size=_MAX_FORM_PART)
    except StarletteHTTPException as e:
        if e.status_code == 400:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Conversation too large to send; clear the chat or remove "
                    "attached images."
                ),
            ) from None
        raise

    # Form(...) used to make `message` required for free; re-create that check.
    message = _require_text(form, "message")
    if message is None or not message.strip():
        raise RequestValidationError(
            [
                {
                    "type": "missing",
                    "loc": ("body", "message"),
                    "msg": "Field required"
                    if message is None
                    else "Message must not be empty",
                    "input": message,
                }
            ]
        )

    model = _require_text(form, "model") or DEFAULT_CHAT_MODEL
    # Allowlist before any history work or the SSE stream: a plain 400 the
    # client can surface.
    if model not in CHAT_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model {model!r}; allowed: "
                + ", ".join(m["id"] for m in CHAT_MODELS)
            ),
        )

    history_text = _require_text(form, "history") or "[]"
    try:
        history_msgs = HISTORY_ADAPTER.validate_json(history_text)
    except ValidationError as e:
        raise RequestValidationError(e.errors(include_url=False)) from None

    contents = _history_to_contents(history_msgs)

    # Current upload: size -> 413, mime allowlist + magic sniff -> 415.
    image = form.get("image")
    current_parts = [types.Part(text=message)]
    if isinstance(image, UploadFile) and image.size:
        if image.size > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image is larger than {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        magic = _IMAGE_MAGIC.get(image.content_type or "")
        if magic is None:
            raise HTTPException(
                status_code=415, detail="Only PNG and JPEG images are supported."
            )
        image_bytes = await image.read()
        await image.close()
        if not image_bytes.startswith(magic):
            raise HTTPException(
                status_code=415, detail="Image contents do not match the declared type."
            )
        message = f"[Attached image: {_clean_label(image.filename)}]\n{message}"
        current_parts = [
            types.Part(
                inline_data=types.Blob(mime_type=image.content_type, data=image_bytes)
            ),
            types.Part(text=message),
        ]

    contents.append(types.Content(role="user", parts=current_parts))

    client = make_client(api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[FIRE_DATA_TOOLS],
    )

    logger.info(
        f"Chat turn start: model={model}, history_len={len(history_msgs)}, "
        f"msg={message[:80]!r}"
    )

    async def event_stream():
        # Prompt-guard (defense-in-depth): screen the user message for injection/
        # abuse before doing any work. Fails open on judge error.
        allowed, reason = await asyncio.to_thread(guard_prompt, message, client)
        if not allowed:
            logger.warning("prompt-guard blocked message: %s | %.80r", reason, message)
            yield _sse(
                "error",
                {"detail": "This request was blocked by a safety check."},
            )
            return

        # Query execution always runs in a container; the pool is started at app
        # startup (main.py lifespan) and fails fast if podman is unavailable.
        pool = getattr(request.app.state, "sandbox_pool", None)
        session = None
        if pool is None:
            yield _sse("error", {"detail": "Sandbox unavailable (pool not started)."})
            return
        try:
            session = await pool.acquire_session()
            async for ev in run_chat_turn(
                client,
                model,
                contents,
                config,
                session,
                max_rounds=MAX_TOOL_ROUNDS,
                on_disconnect=request.is_disconnected,
            ):
                if isinstance(ev, ThinkingEvent):
                    yield _sse("status", {"status": "thinking"})
                elif isinstance(ev, RunningQueryEvent):
                    yield _sse(
                        "status",
                        {"status": "running_query", "queries": ev.queries},
                    )
                elif isinstance(ev, RejectedEvent):
                    yield _sse("rejected", {"queries": ev.queries})
                elif isinstance(ev, DoneEvent):
                    yield _sse("done", _build_done_payload(ev.result, model))
        except SandboxBusy:
            logger.warning("Sandbox pool exhausted; turn rejected")
            yield _sse(
                "error",
                {"detail": "The sandbox is busy right now. Please retry in a moment."},
            )
        except Exception as e:
            logger.exception("Chat stream error")
            yield _sse("error", {"detail": str(e)})
        finally:
            # Always retire the container (kill + background refill), even on
            # client disconnect or mid-turn exception.
            if session is not None:
                pool.release_session(session)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _build_done_payload(result, model: str) -> dict:
    """No plot bytes: the client stores {name, epoch} refs and the server
    re-reads PLOTS_DIR when rebuilding history (see _plot_parts)."""
    payload = {
        "response": result.text,
        # Authoritative: the client stamps its history entry with this, not
        # with whatever it sent.
        "model": model,
    }
    if result.plots:
        payload["plots"] = [{**p, "epoch": PLOT_EPOCH} for p in result.plots]
    if result.queries_run:
        payload["queries"] = result.queries_run
    return payload

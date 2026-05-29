import base64
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from google import genai
from google.genai import types
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.chat_loop import (  # noqa: E402
    DoneEvent,
    RejectedEvent,
    RunningQueryEvent,
    ThinkingEvent,
    run_chat_turn,
)
from app.query_engine import create_namespace  # noqa: E402
from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION  # noqa: E402

router = APIRouter()

MAX_TOOL_ROUNDS = 15


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    model: str = Form("gemini-3.1-pro-preview"),
    history: str = Form("[]"),
    image: UploadFile | None = File(None),  # noqa: B008
):
    api_key = request.session.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    history_list = json.loads(history)

    client = genai.Client(api_key=api_key)

    contents = []
    for msg in history_list:
        role = "user" if msg["role"] == "user" else "model"
        parts = [types.Part(text=msg["content"])]
        if "image" in msg:
            img_data = base64.b64decode(msg["image"]["data"])
            parts.insert(
                0,
                types.Part(
                    inline_data=types.Blob(
                        mime_type=msg["image"]["mime"],
                        data=img_data,
                    )
                ),
            )
        for plot_img in msg.get("plot_images", []):
            if plot_img.get("name"):
                parts.append(types.Part(text=f"[Generated plot: {plot_img['name']}]"))
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=plot_img["mime"],
                        data=base64.b64decode(plot_img["data"]),
                    )
                )
            )
        contents.append(types.Content(role=role, parts=parts))

    # Build current message parts
    image_info = None
    if image and image.size > 0:
        message = f"[Attached image: {image.filename}]\n{message}"
    current_parts = [types.Part(text=message)]
    if image and image.size > 0:
        image_bytes = await image.read()
        current_parts.insert(
            0,
            types.Part(
                inline_data=types.Blob(
                    mime_type=image.content_type,
                    data=image_bytes,
                )
            ),
        )
        image_info = {
            "data": base64.b64encode(image_bytes).decode(),
            "mime": image.content_type,
        }

    contents.append(types.Content(role="user", parts=current_parts))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[FIRE_DATA_TOOLS],
    )

    logger.info(
        f"Chat turn start: model={model}, history_len={len(history_list)}, "
        f"msg={message[:80]!r}"
    )

    async def event_stream():
        try:
            turn_namespace = create_namespace()
            async for ev in run_chat_turn(
                client,
                model,
                contents,
                config,
                turn_namespace,
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
                    yield _sse("done", _build_done_payload(ev.result, image_info))
        except Exception as e:
            logger.exception("Chat stream error")
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _build_done_payload(result, image_info: dict | None) -> dict:
    plot_images = []
    for plot in result.plots:
        plot_path = Path(plot["url"].lstrip("/"))
        if plot_path.exists():
            plot_images.append(
                {
                    "data": base64.b64encode(plot_path.read_bytes()).decode(),
                    "mime": "image/png",
                    "name": plot["name"],
                }
            )
    payload = {
        "response": result.text,
        "image_info": image_info,
    }
    if result.plots:
        payload["plots"] = result.plots
    if plot_images:
        payload["plot_images"] = plot_images
    if result.queries_run:
        payload["queries"] = result.queries_run
    return payload

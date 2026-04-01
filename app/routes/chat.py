import asyncio
import base64
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from google import genai
from google.genai import types
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.query_engine import create_namespace  # noqa: E402
from app.tools import (  # noqa: E402
    FIRE_DATA_TOOLS,
    SYSTEM_INSTRUCTION,
    execute_function_call,
)

router = APIRouter()

MAX_TOOL_ROUNDS = 10


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

    async def event_stream():
        try:
            all_plots = []
            all_queries = []
            turn_namespace = create_namespace()

            # Function calling loop
            for round_num in range(MAX_TOOL_ROUNDS):
                if await request.is_disconnected():
                    return
                yield _sse("status", {"status": "thinking"})

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )

                # Log response structure
                parts = (
                    response.candidates[0].content.parts if response.candidates else []
                )
                part_types = [type(p).__name__ for p in parts]
                logger.debug(f"Round {round_num}: parts={part_types}")
                for p in parts:
                    if hasattr(p, "function_call") and p.function_call:
                        logger.debug(
                            f"  function_call: {p.function_call.name}({p.function_call.args})"  # noqa: E501
                        )
                    if hasattr(p, "text") and p.text:
                        logger.debug(f"  text: {p.text[:200]}")

                # Check for function calls in the response
                function_calls = response.function_calls
                if not function_calls:
                    break

                # Append the model's response (with function call parts) to contents
                contents.append(response.candidates[0].content)

                # Collect queries from this round
                round_queries = []
                for fc in function_calls:
                    if fc.name == "run_query" and fc.args and "code" in fc.args:
                        round_queries.append(fc.args["code"])

                for q in round_queries:
                    logger.info(f"Executing query: {q}")
                yield _sse(
                    "status", {"status": "running_query", "queries": round_queries}
                )

                # Execute each function call and build response parts
                fc_response_parts = []
                rejected_queries = []
                for fc in function_calls:
                    result_str, plots = await asyncio.to_thread(
                        execute_function_call,
                        fc.name,
                        fc.args or {},
                        client,
                        model,
                        namespace=turn_namespace,
                    )
                    logger.debug(f"  {fc.name} result: {result_str[:500]}")
                    result_data = json.loads(result_str)
                    all_plots.extend(plots)
                    fc_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response=result_data,
                            )
                        )
                    )
                    if (
                        fc.name == "run_query"
                        and fc.args
                        and "code" in fc.args
                        and "error" in result_data
                    ):
                        rejected_queries.append(
                            {"code": fc.args["code"], "error": result_data["error"]}
                        )
                rejected_codes = {q["code"] for q in rejected_queries}
                all_queries.extend(q for q in round_queries if q not in rejected_codes)
                if rejected_queries:
                    yield _sse("rejected", {"queries": rejected_queries})

                contents.append(types.Content(role="user", parts=fc_response_parts))

                if await request.is_disconnected():
                    return
            else:
                # Loop exhausted without a text-only response — one final call
                logger.debug("Loop exhausted, making final call")
                yield _sse("status", {"status": "thinking"})
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                )

            # Extract text from the final response
            try:
                response_text = response.text or ""
            except Exception:
                logger.warning(
                    "response.text failed, extracting text from parts",
                    exc_info=True,
                )
                response_text = ""
                if response.candidates:
                    for p in response.candidates[0].content.parts:
                        if hasattr(p, "text") and p.text:
                            response_text += p.text
            if not response_text:
                # Log full response structure to diagnose empty responses
                candidates = response.candidates or []
                finish_reason = (
                    candidates[0].finish_reason if candidates else "no_candidates"
                )
                parts = candidates[0].content.parts if candidates else []
                part_details = []
                for p in parts:
                    if hasattr(p, "function_call") and p.function_call:
                        part_details.append(f"function_call({p.function_call.name})")
                    elif hasattr(p, "text") and p.text:
                        part_details.append(f"text({len(p.text)} chars)")
                    else:
                        part_details.append(f"empty({type(p).__name__})")
                logger.warning(
                    f"Empty response_text. finish_reason={finish_reason}, "
                    f"parts={part_details}, "
                    f"function_calls={bool(response.function_calls)}, "
                    f"all_plots={len(all_plots)}, all_queries={len(all_queries)}"
                )
            else:
                logger.debug(
                    f"Final response_text ({len(response_text)} chars): {response_text[:200]}"  # noqa: E501
                )

            # Read plot images for client-side history
            plot_images = []
            for plot in all_plots:
                plot_path = Path(plot["url"].lstrip("/"))
                if plot_path.exists():
                    plot_images.append(
                        {
                            "data": base64.b64encode(plot_path.read_bytes()).decode(),
                            "mime": "image/png",
                            "name": plot["name"],
                        }
                    )

            result = {
                "response": response_text,
                "image_info": image_info,
            }
            if all_plots:
                result["plots"] = all_plots
            if plot_images:
                result["plot_images"] = plot_images
            if all_queries:
                result["queries"] = all_queries
            yield _sse("done", result)
        except Exception as e:
            logger.exception("Chat stream error")
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

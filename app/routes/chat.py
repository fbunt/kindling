import asyncio
import base64
import json
import logging

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from starlette.responses import StreamingResponse
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION, execute_function_call

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
    image: UploadFile | None = File(None),
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
            parts.insert(0, types.Part(
                inline_data=types.Blob(
                    mime_type=msg["image"]["mime"],
                    data=img_data,
                )
            ))
        contents.append(types.Content(role=role, parts=parts))

    # Build current message parts
    current_parts = [types.Part(text=message)]
    image_info = None
    if image and image.size > 0:
        image_bytes = await image.read()
        current_parts.insert(0, types.Part(
            inline_data=types.Blob(
                mime_type=image.content_type,
                data=image_bytes,
            )
        ))
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
                parts = response.candidates[0].content.parts if response.candidates else []
                part_types = [type(p).__name__ for p in parts]
                logger.debug(f"Round {round_num}: parts={part_types}")
                for p in parts:
                    if hasattr(p, 'function_call') and p.function_call:
                        logger.debug(f"  function_call: {p.function_call.name}({p.function_call.args})")
                    if hasattr(p, 'text') and p.text:
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
                all_queries.extend(round_queries)

                yield _sse("status", {"status": "running_query", "queries": round_queries})

                # Execute each function call and build response parts
                fc_response_parts = []
                for fc in function_calls:
                    result_str, plots = await asyncio.to_thread(
                        execute_function_call,
                        fc.name, fc.args or {}, client, model,
                    )
                    logger.debug(f"  {fc.name} result: {result_str[:500]}")
                    all_plots.extend(plots)
                    fc_response_parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response=json.loads(result_str),
                        )
                    ))

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

            response_text = response.text or ""
            logger.debug(f"Final response_text ({len(response_text)} chars): {response_text[:200]}")

            result = {
                "response": response_text,
                "image_info": image_info,
            }
            if all_plots:
                result["plots"] = all_plots
            if all_queries:
                result["queries"] = all_queries
            yield _sse("done", result)
        except Exception as e:
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

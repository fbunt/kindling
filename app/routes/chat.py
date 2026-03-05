import base64
import json

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from google import genai
from google.genai import types

from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION, execute_function_call

router = APIRouter()

MAX_TOOL_ROUNDS = 5


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

    try:
        # Function calling loop
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            # Check for function calls in the response
            function_calls = response.function_calls
            if not function_calls:
                break

            # Append the model's response (with function call parts) to contents
            contents.append(response.candidates[0].content)

            # Execute each function call and build response parts
            fc_response_parts = []
            for fc in function_calls:
                result_str = execute_function_call(fc.name, fc.args or {})
                fc_response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=json.loads(result_str),
                    )
                ))

            contents.append(types.Content(role="user", parts=fc_response_parts))

        return {
            "response": response.text,
            "image_info": image_info,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

import base64

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from google import genai
from google.genai import types

router = APIRouter()


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

    import json
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

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return {
            "response": response.text,
            "image_info": image_info,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-2.5-flash-preview-05-20"
    history: list[Message] = []


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    api_key = request.session.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    client = genai.Client(api_key=api_key)

    contents = []
    for msg in req.history:
        role = "user" if msg.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

    try:
        response = client.models.generate_content(
            model=req.model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

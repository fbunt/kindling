# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`natlangq` — a natural language query tool. Web app with a chat interface powered by Google Gemini that translates natural language into Python queries against 39 years of MTBS fire data stored as a parquet dataframe, executes them in a sandboxed environment, and returns results or plots.

## Dev Commands

```bash
uv sync                                    # Install dependencies (creates .venv automatically)
uv run uvicorn app.main:app --reload       # Run dev server (http://localhost:8000)
uv add <package>                           # Add a dependency
```

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS (served as static files)
- **LLM**: Google Gemini via `google-genai` SDK

### Structure

```
app/
├── main.py              # FastAPI app, middleware, static file serving
├── routes/
│   ├── auth.py          # POST /api/auth, GET /api/auth/status, POST /api/auth/logout
│   └── chat.py          # POST /api/chat - send message + optional image to Gemini
└── static/
    ├── index.html       # Single-page app: login + chat views
    ├── style.css
    └── app.js           # Frontend logic: auth, chat, image upload, model selector
```

### Key Design Decisions

- **Session-based API key**: Gemini API key stored server-side in session (in-memory, not persisted). Supports `GEMINI_API_KEY` env var via `.env` file.
- **Conversation history**: Maintained client-side and sent with each request.
- **Model selector**: Header dropdown populated from available Gemini models.
- **Image upload**: Images sent as multipart form data, base64-encoded in history for context.
- **Google Search grounding**: Enabled via `google_search` tool on all requests.

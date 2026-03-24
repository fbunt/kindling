# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`kindling` — a natural language query tool. Web app with a chat interface powered by Google Gemini that translates natural language into Python queries against 39 years of MTBS fire data stored as a parquet dataframe, executes them in a sandboxed environment, and returns results or plots.

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
├── query_engine.py      # Sandboxed query execution (AST validation, restricted builtins, timeout)
├── tools.py             # Gemini tool definitions (run_query, get_dataset_info, web_search) + system instruction
├── routes/
│   ├── auth.py          # POST /api/auth, GET /api/auth/status, POST /api/auth/logout
│   └── chat.py          # POST /api/chat - SSE stream of Gemini responses + tool execution
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
- **Google Search grounding**: Available to the model via a `web_search` tool.
- **Query sandbox**: User queries run in a daemon thread with restricted `__builtins__`, AST validation (blocks imports, file I/O, dunder access), and an 8-minute timeout. The namespace persists across `run_query` calls within a single response turn.
- **SSE streaming**: Chat endpoint streams events to the frontend: `status` (thinking/running_query), `rejected` (failed queries with error reason), `done` (final response + plots), `error`.

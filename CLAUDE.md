# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`natlangq` — a natural language query tool. Web app with a chat interface powered by Google Gemini that will eventually translate natural language into Python queries against parquet files.

## Dev Commands

```bash
python -m venv .venv               # Create virtual environment (first time)
.venv/bin/pip install -r requirements.txt  # Install dependencies
.venv/bin/uvicorn app.main:app --reload    # Run dev server (http://localhost:8000)
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
│   ├── auth.py          # POST /api/auth - validate & store API key in session
│   │                    # GET /api/auth/status - check auth status
│   │                    # POST /api/auth/logout
│   └── chat.py          # POST /api/chat - send message to Gemini
└── static/
    ├── index.html       # Single-page app: login + chat views
    ├── style.css
    └── app.js           # Frontend logic: auth flow, chat, message rendering
```

### Key Design Decisions

- **Session-based API key**: Gemini API key stored server-side in session (in-memory, not persisted).
- **Conversation history**: Maintained client-side and sent with each request.
- **Model**: Uses `gemini-2.0-flash` by default.

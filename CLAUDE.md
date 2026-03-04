# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`natlangq` — a natural language query tool. Streamlit chat interface powered by Google Gemini that will eventually translate natural language into Python queries against parquet files.

## Dev Commands

```bash
python -m venv .venv               # Create virtual environment (first time)
.venv/bin/pip install -r requirements.txt  # Install dependencies
.venv/bin/streamlit run app.py             # Run dev server (http://localhost:8501)
```

## Architecture

- **Framework**: Streamlit
- **LLM**: Google Gemini via `google-genai` SDK
- **Entry point**: `app.py`

### Key Design Decisions

- **Session-based API key**: Gemini API key stored in Streamlit session state (in-memory, not persisted).
- **Conversation history**: Maintained in session state.
- **Model selector**: Sidebar dropdown populated from available Gemini models.
- **Google Search grounding**: Enabled via `google_search` tool on all requests.

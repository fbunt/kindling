# natlangq

A natural language query tool for 39 years of [MTBS](https://www.mtbs.gov/) (Monitoring Trends in Burn Severity) fire data. Ask questions in plain English; get back tables, numbers, or charts.

## How it works

A web-based chat interface powered by Google Gemini translates natural language questions into Python queries, executes them in a sandboxed environment against the MTBS dataset (stored as a Parquet dataframe), and returns results or plots.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync                              # install dependencies
```

Set your Gemini API key — either via `.env`:

```
GEMINI_API_KEY=your-key-here
```

or enter it in the login screen when the app starts.

## Running

```bash
uv run uvicorn app.main:app --reload
```

Then open http://localhost:8000.

## Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS
- **LLM**: Google Gemini via `google-genai` SDK
- **Data**: Polars + NumPy, MTBS fire perimeter data in Parquet format

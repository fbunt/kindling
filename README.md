# kindling

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
uv run kindling data/mtbs_pix_data.parquet
```

Or with custom host/port:

```bash
uv run kindling data/mtbs_pix_data.parquet --host 0.0.0.0 --port 9000
```

Then open http://localhost:8000 (or your custom port).

## Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS
- **LLM**: Google Gemini via `google-genai` SDK
- **Data**: Polars + NumPy, MTBS fire perimeter data in Parquet format

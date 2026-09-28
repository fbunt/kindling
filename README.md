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

The app binds to loopback by default, including the container launch paths
(`scripts/run.sh`, `make run`, `compose.yaml`, the Quadlet unit), so it is not
reachable from the network. To use it from another machine, tunnel in:

```bash
ssh -N -L 8000:localhost:8000 user@host   # then open http://localhost:8000 locally
```

(or use IAP). To expose it deliberately, pass `--host 0.0.0.0` to the CLI, or set
`KINDLING_BIND=0.0.0.0` for `scripts/run.sh` / `BIND=0.0.0.0` for `make run`.

If `GEMINI_API_KEY` is set on the server, the login screen offers a
"Use server API key" button; nothing uses the server key until you click it.

## Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS
- **LLM**: Google Gemini via `google-genai` SDK
- **Data**: Polars + NumPy, MTBS fire perimeter data in Parquet format

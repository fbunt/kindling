# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`kindling` — a natural language query tool. Web app with a chat interface powered by Google Gemini that translates natural language into Python queries against 39 years of MTBS fire data stored as a parquet dataframe, executes them in a sandboxed environment, and returns results or plots.

## Dev Commands

```bash
uv sync                                    # Install dependencies (creates .venv automatically)
uv run kindling data/mtbs_pix_data.parquet # Run against a parquet file (http://localhost:8000)
uv run uvicorn app.main:app --reload       # Run dev server with default dataset + hot reload
uv add <package>                           # Add a dependency
```

Worker-only analysis libs (matplotlib/seaborn/numpy/pandas/scipy/scikit-learn/tabulate)
live in `[project.optional-dependencies] worker` — the host/app never import them
(the worker image pins them directly). `uv sync --extra worker` for a full local env.

## Running in a container (Option A)

Two images: **`kindling-app`** (the FastAPI orchestrator, `Containerfile.app`) and
**`kindling-worker`** (query executor, `Containerfile`). The app runs in a container
and spawns worker containers as **siblings on the host runtime** via a mounted
socket — it does not nest a runtime. The host enforces cgroup limits; the worker
container stays the sole security boundary.

```bash
scripts/run.sh --build     # build both images, enable the socket, run (one shot)
scripts/run.sh             # run against data/mtbs_pix_data.parquet (images prebuilt)
scripts/run.sh /abs/x.parquet   # run against another dataset
#   env: KINDLING_SANDBOX_MEM, KINDLING_POOL_SIZE, GEMINI_API_KEY, KINDLING_PORT

# equivalent Make targets:
make build                 # build both images (RUNTIME=docker to use Docker)
make socket                # one-time: enable the rootless podman user socket
make run                   # run the app; PARQUET defaults to data/mtbs_pix_data.parquet
make logs                  # podman logs -f kindling
# or: KINDLING_PARQUET=/abs/host/path.parquet podman compose up --build
# prod: deploy/kindling.container (Quadlet) on Fedora CoreOS / a podman GCP VM
```

Key wiring (see `compose.yaml` / `deploy/kindling.container` / `Makefile`):
- mount the host runtime socket + `CONTAINER_HOST=unix:///run/podman/podman.sock`.
- mount the parquet for the app's schema reads **and** set
  `KINDLING_WORKER_PARQUET_PATH` to the **host** path workers bind-mount (these
  differ once the app is containerized — the app's view ≠ the host path).
- `--security-opt label=disable` (SELinux) to mount the socket.
- The worker image must exist in the **host** image store (workers run there).

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS (served as static files)
- **LLM**: Google Gemini via `google-genai` SDK

### Structure

```
app/
├── cli.py               # `kindling` CLI entry point (argparse, starts uvicorn)
├── main.py              # FastAPI app, lifespan (starts the sandbox pool), static files
├── query_engine.py      # Dataset config + schema/sample reads (get_dataset_info). No execution.
├── guards.py            # LLM defense-in-depth: prompt-guard + code-judge (flash-lite, fail-open)
├── tools.py             # Gemini tool definitions (run_query, get_dataset_info, web_search) + system instruction
├── sandbox/
│   ├── worker.py        # In-container kernel: runs query code with full builtins (JSONL over stdin/stdout)
│   └── pool.py          # Host-side warm pool of Podman workers; per-turn checkout/kill/refill
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
- **Query sandbox (container-only)**: Query code runs ONLY inside locked-down containers, never in the server process. The container is the security boundary (`--network none`, `--read-only`, `--cap-drop ALL`, non-root, memory/pids limits, parquet mounted `:ro`, ephemeral per turn), so code runs with FULL Python builtins and any image library (polars, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn). There is no AST/blocklist filtering. A warm pool keeps containers ready; each chat turn checks one out, then it's killed and a fresh one spawned in the background. The namespace (including `result`) persists across `run_query` calls within a turn. **A container runtime is required** — startup fails fast without one. The runtime is auto-detected (podman preferred, then docker) and overridable via `KINDLING_CONTAINER_RUNTIME`. Rootless Podman is recommended; Docker works but prefer rootless/userns-remap (rootful Docker maps container-root→host-root, a weaker boundary — the app logs a warning). Build the image with `podman build -t kindling-worker:latest -f Containerfile .` (or `docker build`).
- **LLM guards (defense-in-depth, not the boundary)**: a flash-lite **prompt-guard** screens user messages for injection/abuse at the chat endpoint, and a flash-lite **code-judge** reviews generated code before execution. Both block on a clear-malicious verdict and **fail open** on judge error (the container contains the code regardless).
- **SSE streaming**: Chat endpoint streams events to the frontend: `status` (thinking/running_query), `rejected` (failed/blocked queries with error reason), `done` (final response + plots), `error`.

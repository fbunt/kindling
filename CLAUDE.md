# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`kindling` — a natural language query tool. Web app with a chat interface powered by Google Gemini that translates natural language into Python queries against 39 years of MTBS fire data stored as a parquet dataframe, executes them in a sandboxed environment, and returns results or plots.

## Dev Commands

```bash
uv sync                                    # Install dependencies (creates .venv automatically)
make build-worker                          # Build the worker image (required even for a local run)
uv run kindling data/mtbs_pix_data.parquet # Run against a parquet file (http://localhost:8000)
uv run uvicorn app.main:app --reload       # Run dev server with default dataset + hot reload
uv add <package>                           # Add a dependency

uv run pytest -q                           # Unit tests (no dataset/runtime needed; see Testing & CI)
uv run pytest --run-evals tests/evals      # Model-behaviour evals: real Gemini calls, needs key+parquet+runtime
uv run ruff check app/ tests/ && uv run ruff format   # Lint/format (CI gate; pre-commit runs both)
uv run python -m bench {gt|run|grade|report}          # Paper accuracy benchmark (see Benchmark)
```

Even the plain local `uv run kindling` path needs a container runtime **and** a built
`kindling-worker:latest` image — query execution is container-only and startup fails fast
without a runtime (`app/main.py` lifespan).

Worker-only analysis libs (matplotlib/seaborn/numpy/pandas/scipy/scikit-learn/xgboost/tabulate)
live in `[project.optional-dependencies] worker` — the host/app never import them
(the worker image pins them directly in `Containerfile`; keep the two lists in sync).
`uv sync --extra worker` for a full local env.

## Testing & CI

- `tests/` is two-tier. **Unit tests** (`uv run pytest -q`) pass with no dataset and no
  container runtime: `tests/conftest.py` builds a synthetic schema-shaped parquet when `data/`
  is absent, and the container/pool/endpoint tests auto-skip unless podman and the
  `kindling-worker:latest` image exist. **Model evals** in `tests/evals/` carry the
  `model_eval` marker and are skipped unless `--run-evals`; they need `GEMINI_API_KEY`, the
  real parquet, and a runtime, cost real API calls, and dump traces to `.eval-runs/<ts>/`.
  They include an adversarial false-positive tier for the guards.
- **CI** (`.github/workflows/ci.yml`) runs `ruff check` + `pytest -q` on every push/PR, then
  builds both images and pushes `ghcr.io/<owner>/kindling-{app,worker}` on `main` (`latest`,
  `main-<sha>`) and `v*` tags (semver); PRs build only. CI has **no dataset and no container
  runtime**, so new unit tests must use the synthetic fixture / skip patterns or they break
  the gate. Prebuilt images can be pulled from GHCR instead of built locally.

## Benchmark

`bench/` is the ICFFR paper's 25-question accuracy harness (4 categories: lookup, aggregation,
trend, multistep). `python -m bench gt` precomputes reference-Polars ground truth (cached
under `.bench-runs/ground_truth/`, keyed by parquet identity — a cache built on the eval
sample parquet is not reused for the full one); `run [--trials 3] [--questions SPEC]
[--model M] [--run-dir DIR]` drives full chat turns through `app.chat_loop` against real
sandboxes (web_search disabled); then `grade` and `report`. Needs `GEMINI_API_KEY` and the
parquet; output goes to `.bench-runs/<run-dir>/`. Re-running `run` with the same `--run-dir`
resumes (trials with an existing trace JSON are skipped). Usage in `bench/__main__.py`.

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
#   env: KINDLING_SANDBOX_MEM, KINDLING_POOL_SIZE, GEMINI_API_KEY, KINDLING_USE_VERTEX, KINDLING_PORT
#   (full env-knob list under "Environment knobs" below)

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
- forward `KINDLING_USE_VERTEX` into the **app** container if using Vertex — the
  app reads `.env` only via `uv run`, not in the container, so the launch tooling
  must pass it through (a Vertex `AQ.…` key in Developer-API mode 403s).

### Environment knobs

All read in `app/main.py` (lifespan / middleware) unless noted:

| Var | Default | Meaning |
|---|---|---|
| `KINDLING_POOL_SIZE` | 2 | warm workers kept ready |
| `KINDLING_SANDBOX_MAX_TOTAL` | 3 | cap on live workers incl. the one checked out |
| `KINDLING_SANDBOX_MEM` | 110g | per-worker memory cap (intentionally high — real queries push 80–90 GB) |
| `KINDLING_SANDBOX_CPUS` | unset (all cores) | `--cpus` cap; also pins `POLARS_MAX_THREADS` |
| `KINDLING_SANDBOX_PIDS` | 8192 | pid cap; `0`/`none`/`unlimited` disables |
| `KINDLING_SANDBOX_IMAGE` | `kindling-worker:latest` | worker image |
| `KINDLING_REAP_ORPHANS` | off | `all` reaps leftover `kindling-worker-*` on startup (launch tooling sets it) |
| `KINDLING_CONTAINER_RUNTIME` | auto (podman, then docker) | runtime binary override |
| `KINDLING_WORKER_PARQUET_PATH` | unset (= app's path) | **host** path workers bind-mount (containerized app) |
| `KINDLING_USE_VERTEX` | false | Vertex AI express mode instead of the Developer API |
| `KINDLING_SESSION_SECRET` | random per process | Starlette session secret; set for cookies that survive restarts / multi-worker uvicorn. Buys little today: the keystore is in-memory, so a surviving cookie points at a dropped token anyway. Not forwarded by any launch tooling. |
| `KINDLING_LOG_LEVEL` | INFO | log level (also `--log-level` on the CLI) |
| `GEMINI_API_KEY` | unset | pre-authenticates the session; otherwise entered at the login screen |

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS (served as static files)
- **LLM**: Google Gemini via `google-genai` SDK

### Structure

```
app/
├── cli.py               # `kindling` CLI entry point (argparse, starts uvicorn)
├── main.py              # FastAPI app, lifespan (starts the sandbox pool, wipes plots/), middleware, static files
├── chat_loop.py         # run_chat_turn: the per-turn Gemini/tool-execution loop + event dataclasses; shared by the chat route, evals, and bench
├── genai_client.py      # make_client(): Developer-API vs Vertex express selector
├── keystore.py          # In-memory opaque-token -> API-key store behind the session cookie
├── query_engine.py      # Dataset config + schema/sample reads (get_dataset_info). No execution.
├── guards.py            # LLM defense-in-depth: prompt-guard + code-judge (flash-lite, fail-open)
├── tools.py             # Gemini tool schemas (run_query, get_dataset_info, web_search), per-call dispatcher, system instruction
├── sandbox/
│   ├── worker.py        # In-container kernel: runs query code with full builtins (JSONL over stdin/stdout)
│   └── pool.py          # Host-side warm pool of Podman workers; per-turn checkout/kill/refill
├── routes/
│   ├── auth.py          # POST /api/auth, GET /api/auth/status, POST /api/auth/logout
│   ├── chat.py          # POST /api/chat - thin SSE wrapper around chat_loop.run_chat_turn
│   └── plots.py         # GET /plots/<name> - session-gated plot serving (name allowlist)
└── static/
    ├── index.html       # Single-page app: login + chat views
    ├── style.css
    └── app.js           # Frontend logic: auth, chat, image upload, plot gallery
```

### Key Design Decisions

- **Session-based API key**: Gemini API key stored server-side in an in-memory token store (`app/keystore.py`); the session cookie carries only an opaque token (Starlette sessions are signed but unencrypted client-side cookies, so the key itself must never go in one). Supports `GEMINI_API_KEY` env var via `.env` file.
- **Backend selector (`app/genai_client.py`)**: `make_client()` builds every genai client; `KINDLING_USE_VERTEX=true` routes to Vertex AI express mode (`aiplatform.googleapis.com`, `AQ.…` key), else the Gemini Developer API (`generativelanguage.googleapis.com`, `AIza…` key). Key validation uses `generate_content`, not `models.list()` (unsupported under Vertex express mode).
- **Conversation history**: Maintained client-side and sent with each request.
- **Model**: Fixed model set in `app/static/app.js` (`const MODEL`), shown as a header badge. (The former dropdown relied on `models.list()`, which Vertex express mode doesn't support with API keys.)
- **Image upload**: Images sent as multipart form data, base64-encoded in history for context.
- **Google Search grounding**: Available to the model via a `web_search` tool.
- **Query sandbox (container-only)**: Query code runs ONLY inside locked-down containers, never in the server process. The container is the security boundary (`--network none`, `--read-only`, `--cap-drop ALL`, non-root, memory/pids limits, parquet mounted `:ro`, ephemeral per turn), so code runs with FULL Python builtins and any image library (polars, numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, seaborn, tabulate). By default workers get **all host cores** (no `--cpus` cap; polars/BLAS auto-detect) — set `KINDLING_SANDBOX_CPUS=N` to cap, which also pins `POLARS_MAX_THREADS=N` so threads don't over/under-subscribe. Memory cap is `KINDLING_SANDBOX_MEM` (default 110g). There is no AST/blocklist filtering. A warm pool keeps containers ready; each chat turn checks one out, then it's killed and a fresh one spawned in the background. Worker containers are named `kindling-worker-<instance>-<rand>` (per-pool instance id). On startup the pool reaps leftover `kindling-worker-*` containers **only** when `KINDLING_REAP_ORPHANS=all` — off by default so a dev server, pytest, or `bench/` run can't kill a concurrently-running instance's warm workers (one shared host runtime); the launch tooling (run.sh/Makefile/compose/Quadlet) sets `=all` since those are single-instance-per-host. The namespace (including `result`) persists across `run_query` calls within a turn. **A container runtime is required** — startup fails fast without one. The runtime is auto-detected (podman preferred, then docker) and overridable via `KINDLING_CONTAINER_RUNTIME`. Rootless Podman is recommended; Docker works but prefer rootless/userns-remap (rootful Docker maps container-root→host-root, a weaker boundary — the app logs a warning). Build the image with `podman build -t kindling-worker:latest -f Containerfile .` (or `docker build`).
- **LLM guards (defense-in-depth, not the boundary)**: a flash-lite **prompt-guard** screens user messages for injection/abuse at the chat endpoint, and a flash-lite **code-judge** reviews generated code before execution. Both block on a clear-malicious verdict and **fail open** on judge error (the container contains the code regardless). Known limitation: the prompt-guard sees only the current message — client-supplied history (and images) reach the model unscreened, which is accepted at one-VM-per-user.
- **Plot serving**: worker plots are materialized to `plots/` and served by a session-gated route (`routes/plots.py`, 401 without a keystore token, strict filename allowlist) — deliberately **not** a public static mount. `plots/` is wiped on startup and capped at 500 files (oldest pruned on each write, `sandbox/pool.py`). Worker replies are capped at 6 MB and degrade gracefully (tabular rows halved with a truncation note, then an actionable error).
- **SSE streaming**: Chat endpoint streams events to the frontend: `status` (thinking/running_query), `rejected` (failed/blocked queries with error reason), `done` (final response + plots), `error`.

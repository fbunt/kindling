# Progress / session handoff

`kindling` is a natural-language query tool over 39 years of MTBS fire data: a FastAPI chat app
where Gemini writes Polars/Python, which runs in ephemeral Podman/Docker worker containers.
Architecture, invariants, and run/deploy wiring live in `CLAUDE.md`; user-facing setup in
`README.md`; the *what changed* history in `git log` (commit messages are detailed). This file
holds only what those don't: in-flight work, the *why* behind non-obvious decisions, and blockers.

_Updated: 2026-09-22._

> **Maintaining this file.** New work is appended to _Recently done_ with a date. When that
> section passes ~10 items, compact it: fold anything that would stop someone redoing or
> re-deciding something into _Milestones_ (thematic, undated), and **delete the rest** — if
> `git log` already answers it, it does not belong here. _Milestones_ is not append-only either:
> merge related facts into one theme, replace a superseded decision in place, and delete anything
> `CLAUDE.md`, the docs, or `docs/decisions/` now answers. A milestone that deserves a decision
> record but has none yet gets one first (the `decision-log` skill writes it and removes the
> bullet). TODO items carry `**blocked:**` and a reason when they are waiting on someone.

## TODO

- **Run the paper benchmark in full.** Only a smoke run exists (`.bench-runs/smoke`: one question,
  one trial, against the eval *sample* parquet) and the ground-truth cache holds two entries keyed to
  that sample, not the full dataset. Steps: `python -m bench gt` on the full parquet (est. 30–90 min,
  cached by parquet identity), then `python -m bench run` (25 questions × 3 trials, est. 5–9 h and
  ~$20–60 of API), then hand-annotate `failure_mode` in the run's `triage.json` and re-run
  `report`. Re-running `run` with the same `--run-dir` resumes. **blocked:** user's call on when to
  spend the time/API budget.
- **ICFFR short-paper revisions** (reviewer reports in the untracked `reviews.txt`): state up front
  that results are proof-of-concept demonstrations, not validated performance; state the
  sampling-fallback behaviour as a limitation (the model-side disclosure landed 2026-06-13, the paper
  text still needs it); rename one of the duplicated "Usage Example: Query Adaptation" headings.
- **gVisor (`runsc`) as the worker runtime** — recommended defense-in-depth now that there is no
  AST/blocklist layer (kernel-CVE isolation). **blocked:** not installed on current hosts.
- **Image-borne prompt injection** — the prompt-guard screens text only; uploaded images (and
  client-supplied history) reach the model unscreened. Accepted for now at one-VM-per-user; revisit
  if the deployment model changes.
- Prune stale local branches (`sandbox-container-pool`, `containerize`, `ci-ghcr`, `model-evals`,
  etc.) — all merged or superseded on `main`.

## Recently done

- **2026-06-15 — Worker orphan-reaping made opt-in (`KINDLING_REAP_ORPHANS=all`).** Default-on
  reaping of every `kindling-worker-*` container on startup killed the warm workers of any *other*
  instance sharing the host runtime (dev server vs pytest vs `bench/`). Launch tooling
  (run.sh/Makefile/compose/Quadlet) sets `=all` because those are single-instance-per-host.
- **2026-06-13 — Model told to disclose sampling fallbacks.** Direct response to ICFFR reviewer 2:
  when a query falls back to sampling after a failed full-data computation, the user must be told
  the result is on partial data.
- **2026-06-12 — Plots served through a session-gated route, not a public static mount; 500-file
  cap; wipe on startup.** Plots are per-user output and the app is internet-facing on a VM, so an
  unauthenticated `/plots/` listing was a leak. The cap bounds disk on long sessions.
- **2026-06-12 — `KINDLING_SESSION_SECRET` for a stable session secret.** Without it the secret
  is random per process, so every restart/redeploy logs everyone out. Optional; random remains the
  default for dev.
- **2026-06-12 — `bench/` benchmark harness committed.** 25 questions in four categories
  (lookup/aggregation/trend/multistep), 3 trials each; scores executability, accuracy vs
  reference-Polars ground truth, and per-category median latency. Memory instrumentation was
  deliberately dropped as not worth the plumbing for a short paper. Not yet run in full (see TODO).
- **2026-06-08 — `KINDLING_USE_VERTEX` toggle.** Vertex AI express mode uses `AQ.…` keys and a
  different endpoint, and does not support `models.list()` with API keys — hence key validation
  via `generate_content` and the fixed `MODEL` constant in `app.js` instead of a dropdown. Must be
  forwarded into the app *container* (it reads `.env` only via `uv run`).
- **2026-06-04/05 — App containerized (Option A) and CI added.** The app container spawns workers
  as *siblings* via the mounted host runtime socket rather than nesting a runtime, so the host
  kernel enforces cgroup limits and the worker container stays the sole boundary. CI runs
  ruff + pytest (unit tests use a synthetic-schema fixture; container tests and evals auto-skip) and
  pushes both images to GHCR on `main` and `v*` tags.
- **2026-06-03 — Query execution made container-only; AST/blocklist filtering removed.** The
  container (no network, read-only, cap-drop ALL, non-root, mem/pid limits) is the security
  boundary, so the worker runs full Python builtins and any installed library. The two flash-lite
  guards (prompt-guard, code-judge) are defense-in-depth and **fail open** by design.

## Milestones

- Durable decisions have graduated to `docs/decisions/` (deployment model, container as sole
  boundary, sibling workers); operational facts (env knobs, tests vs evals, bench cache
  keying) now live in `CLAUDE.md`.

## Commands

Dev, container, and deploy commands are in `CLAUDE.md`. Beyond those:

- Tests: `uv run pytest -q` (unit); `uv run pytest --run-evals tests/evals` (model evals, costs API).
- Lint: `uv run ruff check app/ tests/` (CI gate; `.pre-commit-config.yaml` also runs ruff).
- Benchmark: `uv run python -m bench {gt|run|grade|report}` — usage in `bench/__main__.py`'s
  docstring; output under `.bench-runs/<run-dir>/`.

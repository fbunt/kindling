# Progress / session handoff

`kindling` is a natural-language query tool over 39 years of MTBS fire data: a FastAPI chat app
where Gemini writes Polars/Python, which runs in ephemeral Podman/Docker worker containers.
Architecture, invariants, and run/deploy wiring live in `CLAUDE.md`; user-facing setup in
`README.md`; the *what changed* history in `git log` (commit messages are detailed). This file
holds only what those don't: in-flight work, the *why* behind non-obvious decisions, and blockers.

_Updated: 2026-09-23._

> **Maintaining this file.** New work is appended to _Recently done_ with a date. When that
> section passes ~10 items, compact it: fold anything that would stop someone redoing or
> re-deciding something into _Milestones_ (thematic, undated), and **delete the rest** — if
> `git log` already answers it, it does not belong here. _Milestones_ is not append-only either:
> merge related facts into one theme, replace a superseded decision in place, and delete anything
> `CLAUDE.md`, the docs, or `docs/decisions/` now answers. A milestone that deserves a decision
> record but has none yet gets one first (the `decision-log` skill writes it and removes the
> bullet). TODO items carry `**blocked:**` and a reason when they are waiting on someone.

## TODO

- **Benchmark 3.8 Flash against 3.1 Pro on the bench harness.** Google's Pro line has stalled at
  the 3.1 preview (Feb 2026; 3.5 Pro announced May 2026, repeatedly delayed) while Flash shipped
  3.5–3.8; public coding/agentic benchmarks put 3.8 Flash at or above 3.1 Pro at ~1/3 the price.
  Steps: `python -m bench gt` on the full parquet (est. 30–90 min, cached by parquet identity;
  of the two cached entries one is sample-keyed and one is the June full parquet, whose
  mtime-based identity has likely changed since the 2026-09-22 symlink move — expect a recompute), then
  `python -m bench run --model gemini-3.8-flash --run-dir ...` and the same with the default
  `gemini-3.1-pro-preview`, then `grade`/`report` each. Re-running `run` with the same
  `--run-dir` resumes. Switch `DEFAULT_CHAT_MODEL` in `app/config.py` if Flash wins.
- **gVisor (`runsc`) as the worker runtime** — recommended defense-in-depth now that there is no
  AST/blocklist layer (kernel-CVE isolation). **blocked:** not installed on current hosts.
- **Image-borne prompt injection** — the prompt-guard screens text only; uploaded images (and
  client-supplied history) reach the model unscreened. Accepted for now at one-VM-per-user; revisit
  if the deployment model changes.
- Prune stale local branches (`sandbox-container-pool`, `containerize`, `ci-ghcr`, `model-evals`,
  etc.) — all merged or superseded on `main`.

## Recently done

- **2026-09-23 - Model selector (3.1 Pro Preview / 3.8 Flash) and one model constant.**
  `app/config.py` holds `CHAT_MODELS`, `DEFAULT_CHAT_MODEL`, `LITE_MODEL`, `MAX_TOOL_ROUNDS`;
  chat/auth/guards/bench/evals import from it. `GET /api/config` serves the list; the header badge
  is now a `<select>` persisted in `localStorage`; `POST /api/chat` 400s on any id outside the
  list before opening the SSE stream. The choice is deliberately **not** locked once a chat
  starts: history is plain text+images (no function-call parts or thought signatures) and the
  server rebuilds `contents` each turn, so switching mid-conversation is safe and useful (explore
  on Flash, escalate to Pro). Each assistant bubble/history entry is stamped with the model the
  server echoes in the `done` event, so earlier turns aren't misattributed after a switch.
- **2026-09-22 — ICFFR short paper accepted; revisions done.** The paper is not stored in this
  repo and the reviewer notes have been discarded. The `bench/` harness was built for its
  evaluation section but was never run in full; it now serves model comparison (see TODO).
- **2026-09-22 — Switched from Vertex AI express mode back to the Gemini Developer API (AI
  Studio).** Vertex express has no prepaid-credit option. `KINDLING_USE_VERTEX=false` in `.env`
  and the Quadlet. Note: AI Studio keys now also start with `AQ.`, so the key prefix no longer
  tells the backends apart. A $0.07 Vertex charge appeared on the old key with nothing running
  since June — check the usage date; if recent, treat the key as leaked and delete it.
- **2026-09-22 — All flash-lite pins (guards, auth validation, eval judge) moved to
  `gemini-3.5-flash-lite`; pins stay explicit.** Full model-eval suite passed with the new judge. Google
  retired `gemini-2.5-flash-lite` for new accounts (404), prompting the bump. Decided against
  `gemini-flash-lite-latest`: the guards are classifiers whose false-positive rate was tuned and
  eval'd against a specific model, so a silent swap under an alias would change block rates
  without a commit. Retirements fail loudly and the guards fail open, so deliberate bumps + a
  guard-eval rerun is the cheaper discipline.
- **2026-09-22 — Dataset symlink repointed.** The parquet moved to `mtbs/data/results/` on the
  data mount; `data/mtbs_pix_data.parquet` (tracked) now follows it.

## Milestones

- Durable decisions have graduated to `docs/decisions/` (deployment model, container as sole
  boundary, sibling workers); operational facts (env knobs, tests vs evals, bench cache
  keying, plot serving, Vertex toggle, orphan reaping) live in `CLAUDE.md`. June 2026 entries
  were compacted out on 2026-09-22; `git log` has the detail.

## Commands

Dev, container, and deploy commands are in `CLAUDE.md`. Beyond those:

- Tests: `uv run pytest -q` (unit); `uv run pytest --run-evals tests/evals` (model evals, costs API).
- Lint: `uv run ruff check app/ tests/` (CI gate; `.pre-commit-config.yaml` also runs ruff).
- Benchmark: `uv run python -m bench {gt|run|grade|report}` — usage in `bench/__main__.py`'s
  docstring; output under `.bench-runs/<run-dir>/`.

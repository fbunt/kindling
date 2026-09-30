# Progress / session handoff

`kindling` is a natural-language query tool over 39 years of MTBS fire data: a FastAPI chat app
where Gemini writes Polars/Python, which runs in ephemeral Podman/Docker worker containers.
Architecture, invariants, and run/deploy wiring live in `CLAUDE.md`; user-facing setup in
`README.md`; the *what changed* history in `git log` (commit messages are detailed). This file
holds only what those don't: in-flight work, the *why* behind non-obvious decisions, and blockers.

_Updated: 2026-09-29._

> **Maintaining this file.** New work is appended to _Recently done_ with a date. When that
> section passes ~10 items, compact it: fold anything that would stop someone redoing or
> re-deciding something into _Milestones_ (thematic, undated), and **delete the rest** — if
> `git log` already answers it, it does not belong here. _Milestones_ is not append-only either:
> merge related facts into one theme, replace a superseded decision in place, and delete anything
> `CLAUDE.md`, the docs, or `docs/decisions/` now answers. A milestone that deserves a decision
> record but has none yet gets one first (the `decision-log` skill writes it and removes the
> bullet). TODO items carry `**blocked:**` and a reason when they are waiting on someone.

## TODO

- **Check the new grader on real traces (pending user runs).** The blind extractor's temperature-0
  stability on Gemini 3.x is unmeasured: grade one real run twice with `--regrade` and diff the
  `extraction` fields (if they drift, the review's fallback is voting plus a vote-split rate). Then
  `python -m bench audit --run-dir DIR` and fill `human_verdict` for the 10% sample so the report
  shows judge-human agreement, and adjudicate the trials pending review in
  `DIR/adjudication.json`.
- **Cost-sizing bench run.** `uv run python -m bench run --run-dir .bench-runs/cost-pro` (25 questions x
  3 trials, 3.1 Pro), then `grade` and `report`. Settles whether the model now uses
  `engine="streaming"` instead of sampling (the other open item from the 2026-09-28 prompt change;
  the loop-exhaustion final call was confirmed accepted by the live eval on 2026-09-28). The report
  prints a dollar estimate from `bench/prices.py` (Developer API list prices as of 2026-09-24; 3.8
  Flash doubles on 2027-01-01).
- **Benchmark 3.8 Flash against 3.1 Pro.** `python -m bench run --models
  gemini-3.1-pro-preview,gemini-3.8-flash --run-dir .bench-runs/pro-vs-flash` (interleaved, one dir per
  model), then `grade`/`report`/`compare` on the parent dir. Ground truth takes ~1.5 min and is cached.
  Best after the grader check above; the review suggests >=5 trials to resolve a 5-15 pp gap. Google's
  Pro line has stalled at the 3.1 preview while Flash reached 3.8; switch `DEFAULT_CHAT_MODEL` in
  `app/config.py` if Flash wins.
- **Bench follow-ups from the 2026-09-29 fix review (non-blocking).** Prompt-guard preflight caches a
  fail-open "guard unavailable" verdict and never retries it on resume; `git_diff_sha` is not a drift key
  and untracked files don't count as dirty; SDK retries x 300 s request timeout can exhaust the 1800 s
  turn timeout and bill an API hang to the model; `report.py` still indexes `trial`/`category` directly;
  `grade` checks `reference_sha` but not `question_sha`.
- **gVisor (`runsc`) as the worker runtime** — recommended defense-in-depth now that there is no
  AST/blocklist layer (kernel-CVE isolation). **blocked:** not installed on current hosts.
- **Image-borne prompt injection** — the prompt-guard screens text only; uploaded images (and
  client-supplied history) reach the model unscreened. Accepted for now at one-VM-per-user; revisit
  if the deployment model changes.
- Prune stale local branches (`sandbox-container-pool`, `containerize`, `ci-ghcr`, `model-evals`,
  etc.) — all merged or superseded on `main`.

## Recently done

- **2026-09-29 — Bench grading redesign (review items 5, 6, 7, 10).** The 3-vote flash-lite yes/no judge is gone: one blind extractor call (never sees the expected value) returns status/value/unit/multiple_candidates/sampled_disclosed as JSON and Python decides accuracy, so a verdict is reproducible from the stored extraction. Grading rules live in `bench/answers.py`, not on `Question`, so existing traces keep their hashes and stay gradable. Strict accuracy fails clarifications (new executability reason `clarified`) and partial-data answers. Deviation from the review's literal partial-data rule: a static-analysis heuristic follows `result` back through the turn's queries and counts only cuts that feed it, exempting ranked top-N rows only while they are the answer. It has known gaps (it is not a sandbox-level row count), so anything it is unsure of, plus unit-only fails, unsure name readings and regex disagreements, goes to `adjudication.json` for review rather than a silent verdict. `audit` samples 10% for hand verdicts; `compare` gives question-clustered bootstrap CIs and a sign-flip test and refuses mismatched grader/ground-truth/dataset hashes. Built over three review rounds; unverified: extractor stability at temperature 0 and judge-human agreement (see TODO).
- **2026-09-29 — Benchmark fixed before its first run.** A 4-lens review (all 25 references correct) found the questions ambiguous on incident-type scope, 4 lookups answerable from the prompt's stats table, and a degenerate M05; rewritten with full-parquet ground truth matching every value the review claimed. The system prompt's examples no longer teach the non-deduplicated pixel-sum error two questions grade. The harness now buckets failures (model-caused failures stay in denominators), fingerprints runs and refuses drifted resumes, captures per-call tokens, interleaves models, and holds a host-wide lock (two 80-90 GB queries would exceed the 125 GB host). Worker polars/pyarrow now match uv.lock, so ground truth and model queries run the same polars.
- **2026-09-28 — SSE keepalive + sanitizer fetch allowlist (audit fix-first #8).** The chat stream sends `: keepalive` after 15 s of silence (prompt-guard, long queries) so a future TLS proxy doesn't cut long turns; the step runs as a task waited on with a timeout, never `wait_for`, which would cancel `run_chat_turn` mid-step. Rendered assistant markdown keeps `<img src>` only for `/plots/` or `data:image/`, strips src/href on other elements and forbids `svg`/`math`/`style`/srcset, closing web_search-injected beacon URLs. CSP is still open and is the backstop for vectors outside that list.
- **2026-09-28 — System prompt corrected (audit fix-first #5).** The Performance section said full scans time out, which pushed the model to sample on the exact questions the paper bench grades; it now requires `.collect(engine="streaming")` and says full scans are fine. The 100-row result cap is stated, and pandas/Series results now carry `total_rows`/`truncated`/`note` like polars frames. The prompt no longer claims eco2/eco3 name mappings exist. The loop-exhaustion final call runs with tools off so 20 rounds of work don't end in "ran out of tool-use rounds".
- **2026-09-28 — Worker protocol hardening + pool fail-fast (audit fix-first #4, #6).** Every JSONL frame carries a request id and the host drops mismatches; a host timeout marks the worker dead and kills it (a late reply could previously answer the next `run_query`); the worker `dup2`s stderr onto fd 1 so query code can't write to the pipe. `SandboxPool.start()` raises on 0 workers so uvicorn exits non-zero; refills retry 3x. The Quadlet gained `Requires=podman.socket` plus `StartLimitIntervalSec=0`/`RestartSec=5`: a review skeptic measured that the ~1 s fail-fast would otherwise trip systemd's 5-starts/10 s limit and leave the unit `failed`.
- **2026-09-28 — Loopback publish + explicit env-key login (audit fix-first #3).** All four launch paths publish on `127.0.0.1` (`KINDLING_BIND`/`BIND=0.0.0.0` opts out; SSH tunnel is the access path). `GET /api/auth/status` no longer mints a session on `GEMINI_API_KEY`; the login view offers a 'Use server API key' button that POSTs `/api/auth/env`. Rationale: an anonymous GET on a LAN-reachable bind spent the operator's key; the bind is the real mitigation, the read-only status is hygiene that also makes logout work.
- **2026-09-23 - History carries plot refs, not base64 (audit finding #4).** The `done` payload
  no longer embeds plot PNGs; assistant history entries hold `{name, epoch}` and the server
  re-reads `plots/<name>.png` itself. Why each piece: the **epoch token** (`PLOT_EPOCH`, random
  per process) exists because `showLogin()` keeps `history` across a server restart while the
  plot counter restarts at `plot-000`, so a name alone would silently show the model the wrong
  plot; refs from a dead process become "no longer available" stubs instead. **Window = 2 turns**
  for both plots and uploads: follow-ups almost always target the previous turn, one rule is
  explainable, and older plots keep their text label (the model can regenerate them). **Structural-
  only pydantic validation**: only wrong types/roles/missing `content` 422; a bad plot name, mime,
  or oversized/invalid `image.data` degrades to a stub, because the user cannot edit history and a
  422 would strand them until Clear (`ImageRef.data` has no `max_length` for exactly this reason;
  the body is bounded by the 48 MiB Content-Length precheck and 32 MiB part cap first). Over
  `_MAX_HISTORY_MSGS=400` entries the oldest whole turns are **truncated, not 422'd**, so a long
  session keeps working. No per-entry `content` length cap (the part cap suffices). The client
  now treats any non-SSE response as an error and shows `detail`, which ends the "Thinking..."
  hang; oversized history is a friendly 400. Legacy `plot_images` entries from stale tabs are
  accepted and stubbed. Still open: the audit UX finding "Transcript diverges from history on
  error or Stop", and the model still never sees a plot within the turn that generated it.
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
- Benchmark: `uv run python -m bench {gt|run|grade|report|audit|compare}` — usage in `bench/__main__.py`'s
  docstring; output under `.bench-runs/<run-dir>/`.

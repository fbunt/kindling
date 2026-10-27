# Progress / session handoff

`kindling` is a natural-language query tool over 39 years of MTBS fire data: a FastAPI chat app
where Gemini writes Polars/Python, which runs in ephemeral Podman/Docker worker containers.
Architecture, invariants, and run/deploy wiring live in `CLAUDE.md`; user-facing setup in
`README.md`; the *what changed* history in `git log` (commit messages are detailed). This file
holds only what those don't: in-flight work, the *why* behind non-obvious decisions, and blockers.

_Updated: 2026-10-26._

> **Maintaining this file.** New work is appended to _Recently done_ with a date. When that
> section passes ~10 items, compact it: fold anything that would stop someone redoing or
> re-deciding something into _Milestones_ (thematic, undated), and **delete the rest** — if
> `git log` already answers it, it does not belong here. _Milestones_ is not append-only either:
> merge related facts into one theme, replace a superseded decision in place, and delete anything
> `CLAUDE.md`, the docs, or `docs/decisions/` now answers. A milestone that deserves a decision
> record but has none yet gets one first (the `decision-log` skill writes it and removes the
> bullet). TODO items carry `**blocked:**` and a reason when they are waiting on someone.

## TODO

- **A bench set that separates the models.** Core, extended and hard are all at or near ceiling
  (see *Recently done*). Fully specified questions are easy for every chat model; the next set needs
  questions where the model must choose the method, graded like E07 (it asks, or states its
  reading and gets that reading's answer), or reproductions from the paper survey. The hard
  tier's plan and reviews are in `.bench-runs/hard-tier-*.md`, the survey in
  `.bench-runs/fire-paper-survey.md` (gitignored).
- **Bench on the 2026-10-26 dataframe: not yet run.** The app runs on it; the bench needs work
  first. E07 and E13 references group by `eco2s`/`eco3s`, which no longer exist, and E13's text
  asks for "its eco2s code": rewrite both on `eco2`/`eco3` (E13's accepted answers are the dotted
  `8.5` form; `eco_names` now hands models the official dotted codes and names). H01's trap
  (state only in the `Event_ID` prefix) is gone now that a `state` column exists: the question says
  "the state MTBS records for it", but a model may reasonably use `state` instead (count unchecked).
  L05 (earliest Wildland Fire Use event) moves from 1988-03-26 to a 1986-08-09 event. Fix
  `grading_ext._region_code_matches` in the same pass: it joins dotted parts unpadded, so `6.2.5`
  reads as `625` and the dataset's own integer code `6205` (or `10105`) fails; add a test.
  Then `gt` (new parquet identity) and treat runs as a new series; 1986 alone moves most counts.
  The land-cover questions held for pre-fire NLCD (Parks 2025 / Haugo 2019 fire rotation,
  Parks-Coop-Davis 2025 high-severity fold increase, Buma 2020 reburn share; drafts in the paper
  survey) can now use `nlcd_pre`, which covers fires from 1986 on.
- **Polars: on 1.41.2, capped `<2` in `pyproject.toml`; 1.44.2 is the latest 1.x and 2.0.0 shipped
  2026-10-06.** Upgrade between bench runs only, bumping `uv.lock` and the Containerfile pin together
  and recomputing ground truth (`bench gt --force`).
- **Bench follow-ups from the 2026-10-01 trace audit (non-blocking; report in
  `.bench-runs/trace-audit-2026-10-01.md`).** A06 `grounded` requires the optional ecoregion name to
  appear in a result (9 of 35 grounded=False); the number cross-check can't read "Four"; report.md
  prints the run's git sha, not the grader's (grader sha is there separately). Unverified: in Flash
  L04 trial 4 a streaming `group_by().agg(unique)` + `list.contains` gave 8124 events vs 8129 in every
  other query; repro on one parquet part before trusting streaming set ops.
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

- **2026-10-26 — App switched to the 2026-10-26 dataframe.** 748,822,169 rows: 1986 restored
  (129 to 398 events), 8,239 border pixels dropped, new `state` (the pixel's state, which differs
  from the `Event_ID` prefix on 1.8% of rows), `area_acres`, `nlcd_pre`/`nlcd_post` (prior- and
  next-year maps), `slope`, `aspect` (-1 = flat); `eco1s`/`eco2s`/`eco3s` and the dask index gone.
  `nlcd` is null for 1984 and `nlcd_pre` for 1984-85 (the annual maps start in 1985); with
  `nlcd_pre`, high-severity forest rows showing non-forest drop from 41% to 5% (2001-2022 sample).
  Ecoregion names come from `app/sandbox/ecoregions.csv` (EPA/CEC Level III shapefile attributes,
  81 rows), preloaded in the worker as `eco_names` and joined on the integer codes, instead of
  names typed into the prompt or web_search. `.eval-runs/sample.parquet` was rebuilt by hand: the
  test and eval fixtures never invalidate it.
- **2026-10-26 — Hard tier piloted: at ceiling, no 5-trial run.** Built and filled from full-data
  ground truth (`9dc05e5`..`9ffbdc3`; decision 0006). Three trials per model: argon 27/27, Flash
  26/27, Pro 25/27, $3.79 in all; no review entries; hand audit (every H01/H04/H05 trial plus a
  sample) agreed with the grader 33/33. The only failures: Pro H07 twice (`pl.len()` is UInt32, so
  numpy pair differences wrapped to ~2.1e9 and the median slope came out 33.00 instead of 4.40;
  Pro's third trial caught it) and Flash H03 once (its query OOM-killed the worker and it spent
  the remaining rounds on the dead worker). Every trap was otherwise avoided: the review pass made
  each question spell out its method, which turned the tier into a test of careful execution.
  5 trials skipped: at ceiling they would only narrow intervals around 100%. Run in
  `.bench-runs/hard-pilot`.
- **2026-10-20 — Three-model bench: core and extended at ceiling; gemini-4-argon added.**
  `gemini-4-argon` joined `CHAT_MODELS` (`d16ddae`; $2/$10 per 1M through 2026-11-30, doubling
  2026-12-01). Core, 5 trials interleaved: Pro, Flash and argon all 125/125 (the A06 rewording
  removed Flash's only misses). Extended, 5 trials: argon 70/70, Pro 66/70, Flash 65/70; the
  only failures are E04 (quoting NIFC's $2.27B suppression cost from recall after a blocked
  web_search: Pro 3/5, Flash 5/5, argon 0/5) and one Pro E10 query bug (`explode` of empty lists
  counted as rows). The extractor misses "0 events, coverage ends 2022" as an abstention every
  time (E05); review catches it. Runs in `.bench-runs/core-pro-flash-argon` and
  `.bench-runs/ext-pilot` (argon not interleaved there).
- **2026-10-20 — Extended-tier ground truth on the full parquet.** Every reference but E07 and E10
  peaks under 7 GB. E07 was OOM-killed at the 100 GB cap: Polars 1.41.2 does not stream `is_in`
  inside a streaming `group_by().agg()`, so memory grows with rows (15 GB on 60 of 623 parts);
  `is_between` streams (19 s, 2.8 GB). A model query written the same way can hit the 110 GB
  sandbox cap. E12 became a row *ceiling* (fewer than 100M rows): eco1 6 has both the top share
  (21.0%) and the most rows, so no floor could change the answer; the ceiling moves it to eco1 7
  (18.2%, 3.9 pp ahead of 11). E08's 2020 top 5 is filled in (answer NORTH COMPLEX, 46.8% vs
  27.4%); E13 accepts `8.5`. Values are pinned in `FULL_EXT` (`tests/test_bench_grading_ext.py`).
- **2026-10-19 — L07, A02 and A06 reworded: "burned-pixel rows" is now "every row whatever its bs value, null included".** Flash's A06 hedge read "burned" as bs 2-4 (eco1 10 instead of 6); L07 and A02 had the same wording. "Any burn-severity class" was rejected because `bs` has nulls (L08) that the references count. References are unchanged; the three `question_sha`s move, and `compare` does not check them, so comparing a new core run with pro-vs-flash would mix wordings on those three. The pro-vs-flash verdicts stand (decision 0004).
- **2026-10-19 — Hand audit of pro-vs-flash: judge-human agreement 26/26** (13 per model, filled blind in `bench audit-web`; Clopper-Pearson 95% lower bound 86.8%). Every sampled trial was a grader pass, so this bounds false passes only; all three grader fails were hand-adjudicated separately. Extractor stability was measured earlier (3 grades, 0 verdict flips), so the core-tier numbers (Pro 125/125, Flash 122/125) are final.
- **2026-10-01 — First real bench: 3.1 Pro 125/125, 3.8 Flash 122/125 (5 trials, $5.53 of chat calls).** The core tier is at ceiling for Pro; the paired gap is +2.4 pp, 95% CI [0, +6.4], p = 0.51, so it does not separate the models (disagreements: Flash A06 hedges between eco1 6 and 10 on 2 trials, T02 one empty response). `DEFAULT_CHAT_MODEL` stays Pro; the extended tier (plan + review in `.bench-runs/extended-tier-plan.md`) is what can separate them. The first grade flagged 23 correct answers as partial data and no real sample, so the row-cut rule now exempts ranked, tie-pick and short (fewer rows than N) cuts, and judges the query that holds the answer. Hedge policy for adjudication: a reply that names a different answer under another reading fails; a hedge whose candidates all grade correct passes. A trace-consistency audit over all 275 traces (every field recomputed, every response read) found the two things the code reviews missed: an empty STOP response was graded executable, and the chat loop said "ran out of tool-use rounds" for any empty reply; both fixed, and decided adjudications now keep the flags they answered. Lesson: review real traces for cross-field consistency, not just code against spec with fakes.
- **2026-09-29 — Bench grading redesign (review items 5, 6, 7, 10).** The 3-vote flash-lite yes/no judge is gone: one blind extractor call (never sees the expected value) returns status/value/unit/multiple_candidates/sampled_disclosed as JSON and Python decides accuracy, so a verdict is reproducible from the stored extraction. Grading rules live in `bench/answers.py`, not on `Question`, so existing traces keep their hashes and stay gradable. Strict accuracy fails clarifications (new executability reason `clarified`) and partial-data answers. Deviation from the review's literal partial-data rule: a static-analysis heuristic follows `result` back through the turn's queries and counts only cuts that feed it, exempting ranked top-N rows only while they are the answer. It has known gaps (it is not a sandbox-level row count), so anything it is unsure of, plus unit-only fails, unsure name readings and regex disagreements, goes to `adjudication.json` for review rather than a silent verdict. `audit` samples 10% for hand verdicts; `compare` gives question-clustered bootstrap CIs and a sign-flip test and refuses mismatched grader/ground-truth/dataset hashes. Built over three review rounds; unverified: extractor stability at temperature 0 and judge-human agreement (see TODO).
- **2026-09-29 — Benchmark fixed before its first run.** A 4-lens review (all 25 references correct) found the questions ambiguous on incident-type scope, 4 lookups answerable from the prompt's stats table, and a degenerate M05; rewritten with full-parquet ground truth matching every value the review claimed. The system prompt's examples no longer teach the non-deduplicated pixel-sum error two questions grade. The harness now buckets failures (model-caused failures stay in denominators), fingerprints runs and refuses drifted resumes, captures per-call tokens, interleaves models, and holds a host-wide lock (two 80-90 GB queries would exceed the 125 GB host). Worker polars/pyarrow now match uv.lock, so ground truth and model queries run the same polars.
- **2026-09-28 — SSE keepalive + sanitizer fetch allowlist (audit fix-first #8).** The chat stream sends `: keepalive` after 15 s of silence (prompt-guard, long queries) so a future TLS proxy doesn't cut long turns; the step runs as a task waited on with a timeout, never `wait_for`, which would cancel `run_chat_turn` mid-step. Rendered assistant markdown keeps `<img src>` only for `/plots/` or `data:image/`, strips src/href on other elements and forbids `svg`/`math`/`style`/srcset, closing web_search-injected beacon URLs. CSP is still open and is the backstop for vectors outside that list.
- **2026-09-28 — System prompt corrected (audit fix-first #5).** The Performance section said full scans time out, which pushed the model to sample on the exact questions the paper bench grades; it now requires `.collect(engine="streaming")` and says full scans are fine. The 100-row result cap is stated, and pandas/Series results now carry `total_rows`/`truncated`/`note` like polars frames. The prompt no longer claims eco2/eco3 name mappings exist. The loop-exhaustion final call runs with tools off so 20 rounds of work don't end in "ran out of tool-use rounds".
- **2026-09-28 — Worker protocol hardening + pool fail-fast (audit fix-first #4, #6).** Every JSONL frame carries a request id and the host drops mismatches; a host timeout marks the worker dead and kills it (a late reply could previously answer the next `run_query`); the worker `dup2`s stderr onto fd 1 so query code can't write to the pipe. `SandboxPool.start()` raises on 0 workers so uvicorn exits non-zero; refills retry 3x. The Quadlet gained `Requires=podman.socket` plus `StartLimitIntervalSec=0`/`RestartSec=5`: a review skeptic measured that the ~1 s fail-fast would otherwise trip systemd's 5-starts/10 s limit and leave the unit `failed`.
- **2026-09-28 — Loopback publish + explicit env-key login (audit fix-first #3).** All four launch paths publish on `127.0.0.1` (`KINDLING_BIND`/`BIND=0.0.0.0` opts out; SSH tunnel is the access path). `GET /api/auth/status` no longer mints a session on `GEMINI_API_KEY`; the login view offers a 'Use server API key' button that POSTs `/api/auth/env`. Rationale: an anonymous GET on a LAN-reachable bind spent the operator's key; the bind is the real mitigation, the read-only status is hygiene that also makes logout work.

## Milestones

- Durable decisions have graduated to `docs/decisions/` (deployment model, container as sole
  boundary, sibling workers); operational facts (env knobs, tests vs evals, bench cache
  keying, plot serving, Vertex toggle, orphan reaping) live in `CLAUDE.md`. June 2026 entries
  were compacted out on 2026-09-22, the 2026-09-22/23 ones on 2026-10-01; `git log` has the detail.
- **ICFFR short paper accepted (2026-09-22).** Not stored in this repo; reviewer notes discarded.
  `bench/` was built for its evaluation section and now also drives model comparison.
- **On the Gemini Developer API (AI Studio) since 2026-09-22.** Vertex express has no prepaid-credit
  option. AI Studio keys also start with `AQ.`, so the prefix no longer tells the backends apart. A
  $0.07 Vertex charge appeared on the old key with nothing running since June: if its usage date is
  recent, treat that key as leaked and delete it.
- **Flash-lite pins stay explicit, never `-latest`.** The guards are classifiers whose false-positive
  rate was tuned against a specific model; an alias would change block rates without a commit.
  Retirements fail loudly (404) and the guards fail open, so a deliberate bump plus a guard-eval rerun
  is the cheaper discipline.
- **History plot refs: the non-obvious whys.** The per-process epoch token exists because the client
  keeps history across a server restart while plot names restart at `plot-000`. The structural-only
  validation and truncate-not-422 rules exist because the user cannot edit history, so a 422 would
  strand them until Clear. Still open: the transcript diverges from history on error or Stop, and the
  model never sees a plot within the turn that generated it.

## Commands

Dev, container, and deploy commands are in `CLAUDE.md`. Beyond those:

- Tests: `uv run pytest -q` (unit); `uv run pytest --run-evals tests/evals` (model evals, costs API).
- Lint: `uv run ruff check app/ tests/` (CI gate; `.pre-commit-config.yaml` also runs ruff).
- Benchmark: `uv run python -m bench {gt|run|grade|report|audit|compare}` — usage in `bench/__main__.py`'s
  docstring; output under `.bench-runs/<run-dir>/`.

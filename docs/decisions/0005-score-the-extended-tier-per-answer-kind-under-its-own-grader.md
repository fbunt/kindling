# 0005. Score the extended bench tier per answer kind, under its own grader version

_Status: Accepted. Recorded 2026-10-20. Decided 2026-10-19._

## Context

The core tier is at ceiling (Pro 125/125, Flash 122/125; [0004](0004-score-the-bench-strictly-with-rules-fixed-before-grading.md)),
so the extended tier (E01-E14) is what can separate models. Several of its questions have no right
number to check: the right reply to E04 (no cost column) and E05 (no 2023 data) is to say so, E06
asks for a list longer than a 100-row result, E07 is ambiguous on purpose, and E09 asks for a plot.
0004's strict rules would fail the correct behaviour on each (a refusal has no value, asking about
E07 is a clarification, listing the first 100 of 507 events is a row cut). The core run was already
graded, so any change to the shared grader would mark the archived run stale.

## Decision

- **Own grader version.** Extended traces are graded under `ext_grader_sha()` (`EXT_GRADER_VERSION`
  over the core sha, the extended extractor kinds and `EXT_ANSWERS`), so the core sha stays
  `ee57b57a09b8`. One combined sha was rejected because it would move the archived run's sha.
  One tier per run dir; the tiers are reported apart, never in one headline.
- **Four answer kinds with their own rules** (`bench/grading_ext.py`); scalar and text questions
  keep 0004's rules.
  - *abstain (E04, E05)*: pass iff the reply says the data is missing and presents no value,
    whatever its status; a refusal counts. The extractor fills these fields under any status,
    overriding the core null-unless-answer rule, which would read a refusal as no answer. E05 passes
    a stated 0 that comes with the coverage statement. Partial data never fails an abstention.
  - *disclosure (E06)*: the leading 10 names in order and either the right total (1%), a stated
    truncation, or every item as a markdown table row (a bulleted list goes to review). Only
    `.sample(`/`.fetch(`/`.gather_every(` count as partial data, since a cut list is the subject.
  - *clarify_or_assume (E07)*: asking with no committed reading passes, unlike 0004's
    clarification fail, because on an ambiguous question asking is correct. A committed reading must
    state its measure and level and give an answer in that reading's accepted set; each set holds the
    winners of every defensible variant (rows over any bs, bs 1-4, bs 2-4; summed event area; events
    per region touched or per dominant region), not one canonical reading (the A06 lesson). An
    unstated level counts as level 1 only for a level-I name. 0004's hedge rule does not apply here.
  - *plot (E09)*: right year and count plus a plot from a successful query. No pixel check; every
    E09 plot is checked by hand.
- **Executability is not applicable** for abstain and clarify_or_assume, which need no query;
  counting them as non-executable would penalise the right behaviour. Model and infra errors still
  count as non-executable and stay in every denominator.
- **Review, not auto-fail,** for cases a rule cannot judge alone: E08 right with no query, E06
  `wrong_items`, hedges in abstain/disclosure (which still fail), a measure outside E07's sets, a
  name-only E07 region.
- **E12 uses a row ceiling** (fewer than 100,000,000 rows), decided 2026-10-20: on the full data
  eco1 6 has both the top share and the most rows, so no floor could make the filter change the
  answer. Dropping E12 was the planned fallback; the ceiling keeps a multi-step question.
- **Undisclosed truncation** is a report-only line with its own definition version, not a verdict,
  so defining it moved no grader sha.

## Consequences

- These rules are fixed before the pilot is graded. A rule change after that bumps
  `EXT_GRADER_VERSION` (grading code is not hashed, so the bump is manual) and the final run goes in
  a new run dir; pilot traces never reach the paper under rules written after seeing them.
- Extended accuracy is not comparable with core accuracy, and its executability rate has a smaller
  denominator.
- E07's sets are only the variants we listed; a model that finds another defensible one is decided
  by hand.

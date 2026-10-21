# 0006. Add a hard bench tier, scored by answer parts and namesake disclosure

_Status: Accepted. Recorded 2026-10-20. Decided 2026-10-20._

## Context

Both earlier tiers are at ceiling. On the extended tier Pro scored 66/70, Flash 65/70 and
gemini-4-argon 70/70, and every question that one well-specified query answers was 100% for all
three. The one quiet model error in 210 extended trials was a computation bug, not a misread
question. A draft of harder questions was reviewed twice: a reader answering cold saw every
semantic trap coming from the system prompt, so most of the draft was dropped. What remained
was rebuilt around per-pixel histories, two-stage statistics, multi-turn carry-over and fire
names shared by several events, plus analyses taken from MTBS papers (Dennison 2014, Iglesias
2022, Parks 2014). Questions that filter on land cover are held: same-year `nlcd` is largely
post-fire for moderate and high severity, so forest-trend reproductions (Parks 2025, Buma 2020)
wait for a pre-fire land-cover column.

## Decision

- **A third tier, `hard` (H01-H09), with its own grader sha** (`HARD_GRADER_VERSION` over the
  core sha, `judge.hard_fingerprint()` and `HARD_ANSWERS`). Adding the questions to the extended
  tier was rejected: it would move the extended sha and make the finished extended runs
  incomparable with later ones. One tier per run dir; tiers are reported apart, and a hard
  comparison is descriptive.
- **Multi-part answers (`fields`, H01 and H05) pass only when every part is right.** The
  extractor is told the part names, never values. A state passes by its full name, a list of
  codes, or the expected state's capital code anywhere; other codes in prose are not read as
  states, because two-letter capitals are too often words. A guessed state without the exact
  count fails, so H01 tests finding the state in the `Event_ID` prefix rather than recalling it.
  A bare "Yolla Bolly Complex" goes to review because a second 2008 event shares that name.
- **Namesake disclosure (`disambiguate`, H04) passes iff the reply notes several events of that
  year share the name and does not present a sum over them as one fire.** Listing them in a
  clarifying question passes; a blind "which one?" fails, because noticing the namesakes needs a
  query. So, unlike E07, executability applies.
- **Tolerances are set to exclude named wrong readings, and every reference asserts that.** For
  example, H07's 0.01 band excludes `median_low`/`median_high` and scikit-learn's spatial-median
  Theil-Sen; H09's 0.1 pp band excludes a wildfire-only lookback. A misreading that lands inside
  the band is logged, not asserted (H03's bs 1-4 reading, 0.4 pp off).
- **Paper reproductions are graded against our own references**, never the published value.
  The MTBS release, severity measure and region definitions differ; the published number is a
  sanity check only.

## Consequences

- These rules are fixed before the hard pilot is graded. A later change bumps
  `HARD_GRADER_VERSION` and the final run goes in a new run dir.
- Review routes added: `fraction_for_percent`, `unqualified_name`, `year_disagreement`,
  `second_state`, `fewer_events_listed`; triage adds `conflated_namesakes`.
- A hedge written inside a state part ("FL (or possibly GA)") relies on the extractor's
  `multiple_candidates` flag; the hand audit runs `--full` on H01, H04 and H05.
- The land-cover questions return once the dataframe has pre-fire land cover, as a new run.

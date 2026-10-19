# 0004. Score the bench strictly, with rules fixed before grading and judged in Python

_Status: Accepted. Recorded 2026-10-19. Decided 2026-09-29._

## Context

The paper's accuracy numbers come from `bench/`: 25 questions, each answered by a chat model
in real sandboxes. A reply can be right but hedged, right but computed on sampled or cut rows,
or a clarifying question, and a grader can be an LLM verdict or code. The first real run
(pro-vs-flash, 2026-10-01: Pro 125/125, Flash 122/125) put these choices under pressure, and
one fail (Flash A06) arguably came from the question's wording rather than the model. That
run is archived in [`bench/results/pro-vs-flash-2026-10/`](../../bench/results/pro-vs-flash-2026-10/).

## Decision

- **Strict accuracy is the headline.** A clarification, more than one candidate answer, or
  an answer built on partial data fails, even when a number lands in the band. A user who
  gets two answers or a sample-based figure has not been answered. Exec-gated accuracy is
  reported, but only as secondary.
- **Accuracy is decided in Python, not by an LLM verdict.** A blind extractor (never sees
  the expected value) reports what the reply answered; `check_answer` compares it with
  ground truth. The 3-vote yes/no LLM judge it replaced lost because its verdicts could not
  be reproduced or audited from stored data.
- **Hedges:** a reply that names a different answer under another reading fails; a hedge
  whose candidates all grade correct passes. This is applied by hand in `adjudication.json`
  (decided 2026-10-01).
- **Partial data means rows were dropped.** Top-N selection after a sort, a one-row tie pick
  among filtered rows, and a cut that returned fewer rows than its limit are not partial
  data. The literal rule ("any head/limit") flagged 23 correct answers and no real sample on
  the first run.
- **Rules are fixed before a run is graded and never relaxed afterwards for that run.** A
  rule change bumps `GRADER_VERSION` and applies to every trace alike. A question found to be
  ambiguous keeps its verdicts in the graded run, gets a `question_ambiguity` triage label
  and a paper footnote, and is reworded only for the next run (precedent: Flash A06, where
  "burned-pixel rows" can mean all rows or burn severity 2-4). Relaxing a rule after seeing
  which model it hurts was rejected as post hoc.
- **Verification:** 5 trials per question; models compared with a question-clustered
  bootstrap and a sign-flip test; a seeded 10% hand audit filled blind (model and grader
  verdict hidden; `bench audit-web`); extractor stability checked by regrading.

## Consequences

- Reported accuracy is a lower bound on "useful answer" rates, and a careful model that
  flags ambiguity can score below a careless one; triage labels and footnotes carry that.
- A rule fix found after grading reaches the paper only by regrading every trace under the
  new `GRADER_VERSION`, never by editing single verdicts.
- Question wording is frozen while a run is live; rewording waits for the next run.

# Benchmark report - gemini-3.8-flash

- model: `gemini-3.8-flash` (interleaved with gemini-3.1-pro-preview, gemini-3.8-flash, seed 0)
- served model version(s): gemini-3.8-flash
- parquet: `data/mtbs_pix_data.parquet` (identity `29d168015b13`)
- prompt sha `910e543479e43198`, question set sha `31c1ed192052`, git `e2250596034027370697ba8a573688c751b1c01f`
- trials per question: 5
- trials graded: 125
- trials excluded as infra errors: 0
- error buckets: {'model': 1}

## Per category

| category | questions | accuracy (strict) | executability | exec & accurate | clarification | exec-gated accuracy | all-trials-correct | median turn latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lookup | 8 | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) | 0/40 (0%) | 40/40 (100%) | 8/8 | 5.6 |
| aggregation | 7 | 33/35 (94%) | 35/35 (100%) | 33/35 (94%) | 0/35 (0%) | 33/35 (94%) | 6/7 | 6.1 |
| trend | 5 | 24/25 (96%) | 24/25 (96%) | 24/25 (96%) | 0/25 (0%) | 24/24 (100%) | 4/5 | 23.3 |
| multistep | 5 | 25/25 (100%) | 25/25 (100%) | 25/25 (100%) | 0/25 (0%) | 25/25 (100%) | 5/5 | 22.8 |
| overall | 25 | 122/125 (98%) | 124/125 (99%) | 122/125 (98%) | 0/125 (0%) | 122/124 (98%) | 23/25 | 11.3 |

Accuracy is strict and not gated on executability: a clarification, a multiple-candidate answer or a partial-data answer fails even if a number lands in the band. Exec-gated accuracy (accurate among executable trials) is secondary.

Sensitivity, infra errors counted as wrong: accuracy 122/125 (98%).

Latency medians include failed trials and start after sandbox checkout; infra-error trials are excluded everywhere else. Model-caused failures (model_malformed) stay in every denominator.

## Per question

| id | category | exec | acc | pass@1 | all-correct | med turn (s) | med query (s) | fail reasons | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | aggregation | 5/5 | 5/5 | 1.00 | yes | 30.4 | 3.8 |  | needs_review, adjudicated |
| A02 | aggregation | 5/5 | 5/5 | 1.00 | yes | 4.0 | 0.4 |  |  |
| A03 | aggregation | 5/5 | 5/5 | 1.00 | yes | 5.6 | 0.1 |  | adjudicated |
| A04 | aggregation | 5/5 | 5/5 | 1.00 | yes | 3.6 | 0.1 |  |  |
| A05 | aggregation | 5/5 | 5/5 | 1.00 | yes | 13.1 | 5.7 |  |  |
| A06 | aggregation | 5/5 | 3/5 | 0.60 | no | 7.3 | 1.0 | adjudicated x2 | needs_review, adjudicated |
| A07 | aggregation | 5/5 | 5/5 | 1.00 | yes | 5.8 | 0.4 |  |  |
| L01 | lookup | 5/5 | 5/5 | 1.00 | yes | 6.0 | 1.1 |  |  |
| L02 | lookup | 5/5 | 5/5 | 1.00 | yes | 7.4 | 1.7 |  |  |
| L03 | lookup | 5/5 | 5/5 | 1.00 | yes | 5.6 | 2.7 |  |  |
| L04 | lookup | 5/5 | 5/5 | 1.00 | yes | 21.4 | 2.7 |  |  |
| L05 | lookup | 5/5 | 5/5 | 1.00 | yes | 4.6 | 0.2 |  |  |
| L06 | lookup | 5/5 | 5/5 | 1.00 | yes | 9.6 | 3.0 |  | adjudicated |
| L07 | lookup | 5/5 | 5/5 | 1.00 | yes | 2.9 | 0.0 |  |  |
| L08 | lookup | 5/5 | 5/5 | 1.00 | yes | 3.0 | 0.0 |  |  |
| M01 | multistep | 5/5 | 5/5 | 1.00 | yes | 26.5 | 5.7 |  | adjudicated |
| M02 | multistep | 5/5 | 5/5 | 1.00 | yes | 47.2 | 4.2 |  |  |
| M03 | multistep | 5/5 | 5/5 | 1.00 | yes | 11.1 | 0.1 |  | adjudicated |
| M04 | multistep | 5/5 | 5/5 | 1.00 | yes | 16.7 | 1.0 |  | adjudicated |
| M05 | multistep | 5/5 | 5/5 | 1.00 | yes | 22.8 | 5.7 |  | adjudicated |
| T01 | trend | 5/5 | 5/5 | 1.00 | yes | 21.2 | 0.0 |  |  |
| T02 | trend | 4/5 | 4/5 | 0.80 | no | 35.0 | 3.5 | no_answer x1 | model_error |
| T03 | trend | 5/5 | 5/5 | 1.00 | yes | 20.0 | 2.9 |  |  |
| T04 | trend | 5/5 | 5/5 | 1.00 | yes | 12.6 | 0.0 |  |  |
| T05 | trend | 5/5 | 5/5 | 1.00 | yes | 30.9 | 1.8 |  |  |

## Grading

- grader sha: `ee57b57a09b8`
- extractor served version(s): gemini-3.5-flash-lite
- grader errors (extractor failed twice; counted as ungraded): 0
- partial data: 0/125 (0%) of trials; disclosed 0/125 (0%), undisclosed 0/125 (0%) (strict accuracy fails all of them)
- grounded (answer appears in a query result): 111/124 (90%)
- adjudicated: 15 (1 overridden to pass, 1 to fail; 13 on trials no longer flagged); pending review: 0
- hand audit: 13 sampled, 13 filled; judge-human agreement 13/13 (100%)

## Tokens and cost

All generate_content calls the trials made: chat rounds plus code-judge verdicts, including attempts that were later retried. Grading (extractor) and prompt-guard preflight calls are not included. Thinking and temperature are the app defaults (not overridden).

| model | calls | prompt | cached | output | thoughts | total | est. USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.5-flash-lite | 289 | 110,347 | 0 | 9,679 | 0 | 120,026 | $0.06 |
| gemini-3.8-flash | 468 | 2,193,840 | 4,034 | 82,779 | 81,757 | 2,358,376 | $2.26 |

- trials with usage: 125; tokens per trial: mean 19,827, median 15,542, max 93,540
- tokens per correct answer: 20,314 (122 correct)
- estimated cost: $2.32 (prices as of 2026-09-24); $0.019 per correct answer

### Tokens per question

| id | trials | total tokens | mean per trial |
| --- | --- | --- | --- |
| A01 | 5 | 184,075 | 36,815 |
| A02 | 5 | 32,434 | 6,486 |
| A03 | 5 | 70,263 | 14,052 |
| A04 | 5 | 35,745 | 7,149 |
| A05 | 5 | 56,086 | 11,217 |
| A06 | 5 | 90,850 | 18,170 |
| A07 | 5 | 82,141 | 16,428 |
| L01 | 5 | 41,775 | 8,355 |
| L02 | 5 | 35,311 | 7,062 |
| L03 | 5 | 30,695 | 6,139 |
| L04 | 5 | 191,815 | 38,363 |
| L05 | 5 | 44,374 | 8,874 |
| L06 | 5 | 36,883 | 7,376 |
| L07 | 5 | 30,621 | 6,124 |
| L08 | 5 | 31,337 | 6,267 |
| M01 | 5 | 115,577 | 23,115 |
| M02 | 5 | 190,984 | 38,196 |
| M03 | 5 | 123,138 | 24,627 |
| M04 | 5 | 86,400 | 17,280 |
| M05 | 5 | 130,216 | 26,043 |
| T01 | 5 | 160,389 | 32,077 |
| T02 | 5 | 244,355 | 48,871 |
| T03 | 5 | 142,552 | 28,510 |
| T04 | 5 | 93,199 | 18,639 |
| T05 | 5 | 197,187 | 39,437 |

## Failure triage

| question | trial | executable | exec reason | accuracy reason | failure mode |
| --- | --- | --- | --- | --- | --- |
| A06 | 0 | yes | ok | adjudicated | question_ambiguity |
| A06 | 1 | yes | ok | adjudicated | question_ambiguity |
| T02 | 1 | no | model_malformed | no_answer | empty_response |

### Failure-mode distribution (annotated only)

| failure mode | count |
| --- | --- |
| empty_response | 1 |
| question_ambiguity | 2 |

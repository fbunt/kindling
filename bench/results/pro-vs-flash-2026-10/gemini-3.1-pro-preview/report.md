# Benchmark report - gemini-3.1-pro-preview

- model: `gemini-3.1-pro-preview` (interleaved with gemini-3.1-pro-preview, gemini-3.8-flash, seed 0)
- served model version(s): gemini-3.1-pro-preview
- parquet: `data/mtbs_pix_data.parquet` (identity `29d168015b13`)
- prompt sha `910e543479e43198`, question set sha `31c1ed192052`, git `e2250596034027370697ba8a573688c751b1c01f`
- trials per question: 5
- trials graded: 125
- trials excluded as infra errors: 0
- error buckets: none

## Per category

| category | questions | accuracy (strict) | executability | exec & accurate | clarification | exec-gated accuracy | all-trials-correct | median turn latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lookup | 8 | 40/40 (100%) | 40/40 (100%) | 40/40 (100%) | 0/40 (0%) | 40/40 (100%) | 8/8 | 9.2 |
| aggregation | 7 | 35/35 (100%) | 35/35 (100%) | 35/35 (100%) | 0/35 (0%) | 35/35 (100%) | 7/7 | 10.1 |
| trend | 5 | 25/25 (100%) | 25/25 (100%) | 25/25 (100%) | 0/25 (0%) | 25/25 (100%) | 5/5 | 16.7 |
| multistep | 5 | 25/25 (100%) | 25/25 (100%) | 25/25 (100%) | 0/25 (0%) | 25/25 (100%) | 5/5 | 19.4 |
| overall | 25 | 125/125 (100%) | 125/125 (100%) | 125/125 (100%) | 0/125 (0%) | 125/125 (100%) | 25/25 | 12.6 |

Accuracy is strict and not gated on executability: a clarification, a multiple-candidate answer or a partial-data answer fails even if a number lands in the band. Exec-gated accuracy (accurate among executable trials) is secondary.

Sensitivity, infra errors counted as wrong: accuracy 125/125 (100%).

Latency medians include failed trials and start after sandbox checkout; infra-error trials are excluded everywhere else. Model-caused failures (model_malformed) stay in every denominator.

## Per question

| id | category | exec | acc | pass@1 | all-correct | med turn (s) | med query (s) | fail reasons | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | aggregation | 5/5 | 5/5 | 1.00 | yes | 13.5 | 3.7 |  |  |
| A02 | aggregation | 5/5 | 5/5 | 1.00 | yes | 6.5 | 0.4 |  |  |
| A03 | aggregation | 5/5 | 5/5 | 1.00 | yes | 10.4 | 0.1 |  |  |
| A04 | aggregation | 5/5 | 5/5 | 1.00 | yes | 6.6 | 0.1 |  |  |
| A05 | aggregation | 5/5 | 5/5 | 1.00 | yes | 15.8 | 5.4 |  |  |
| A06 | aggregation | 5/5 | 5/5 | 1.00 | yes | 10.1 | 0.4 |  |  |
| A07 | aggregation | 5/5 | 5/5 | 1.00 | yes | 6.8 | 0.4 |  |  |
| L01 | lookup | 5/5 | 5/5 | 1.00 | yes | 9.6 | 1.3 |  |  |
| L02 | lookup | 5/5 | 5/5 | 1.00 | yes | 13.1 | 5.4 |  |  |
| L03 | lookup | 5/5 | 5/5 | 1.00 | yes | 9.2 | 2.7 |  |  |
| L04 | lookup | 5/5 | 5/5 | 1.00 | yes | 9.1 | 0.4 |  |  |
| L05 | lookup | 5/5 | 5/5 | 1.00 | yes | 7.8 | 0.1 |  |  |
| L06 | lookup | 5/5 | 5/5 | 1.00 | yes | 10.0 | 0.7 |  | adjudicated |
| L07 | lookup | 5/5 | 5/5 | 1.00 | yes | 6.6 | 0.0 |  |  |
| L08 | lookup | 5/5 | 5/5 | 1.00 | yes | 7.2 | 0.0 |  |  |
| M01 | multistep | 5/5 | 5/5 | 1.00 | yes | 24.9 | 6.3 |  | needs_review, adjudicated |
| M02 | multistep | 5/5 | 5/5 | 1.00 | yes | 14.7 | 4.3 |  |  |
| M03 | multistep | 5/5 | 5/5 | 1.00 | yes | 12.7 | 0.5 |  | adjudicated |
| M04 | multistep | 5/5 | 5/5 | 1.00 | yes | 20.7 | 7.0 |  | adjudicated |
| M05 | multistep | 5/5 | 5/5 | 1.00 | yes | 21.7 | 5.6 |  | adjudicated |
| T01 | trend | 5/5 | 5/5 | 1.00 | yes | 17.0 | 6.2 |  |  |
| T02 | trend | 5/5 | 5/5 | 1.00 | yes | 15.5 | 3.9 |  |  |
| T03 | trend | 5/5 | 5/5 | 1.00 | yes | 17.6 | 3.0 |  |  |
| T04 | trend | 5/5 | 5/5 | 1.00 | yes | 14.9 | 0.8 |  |  |
| T05 | trend | 5/5 | 5/5 | 1.00 | yes | 22.6 | 1.8 |  |  |

## Grading

- grader sha: `ee57b57a09b8`
- extractor served version(s): gemini-3.5-flash-lite
- grader errors (extractor failed twice; counted as ungraded): 0
- partial data: 0/125 (0%) of trials; disclosed 0/125 (0%), undisclosed 0/125 (0%) (strict accuracy fails all of them)
- grounded (answer appears in a query result): 107/125 (86%)
- adjudicated: 12 (0 overridden to pass, 0 to fail; 10 on trials no longer flagged); pending review: 0
- hand audit: 13 sampled, 13 filled; judge-human agreement 13/13 (100%)

## Tokens and cost

All generate_content calls the trials made: chat rounds plus code-judge verdicts, including attempts that were later retried. Grading (extractor) and prompt-guard preflight calls are not included. Thinking and temperature are the app defaults (not overridden).

| model | calls | prompt | cached | output | thoughts | total | est. USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 266 | 902,100 | 0 | 30,375 | 84,427 | 1,016,902 | $3.18 |
| gemini-3.5-flash-lite | 130 | 47,141 | 0 | 4,417 | 0 | 51,558 | $0.03 |

- trials with usage: 125; tokens per trial: mean 8,547, median 7,480, max 19,023
- tokens per correct answer: 8,547 (125 correct)
- estimated cost: $3.21 (prices as of 2026-09-24); $0.026 per correct answer

### Tokens per question

| id | trials | total tokens | mean per trial |
| --- | --- | --- | --- |
| A01 | 5 | 38,907 | 7,781 |
| A02 | 5 | 32,166 | 6,433 |
| A03 | 5 | 74,435 | 14,887 |
| A04 | 5 | 32,362 | 6,472 |
| A05 | 5 | 36,873 | 7,374 |
| A06 | 5 | 51,270 | 10,254 |
| A07 | 5 | 39,090 | 7,818 |
| L01 | 5 | 34,469 | 6,893 |
| L02 | 5 | 34,814 | 6,962 |
| L03 | 5 | 31,500 | 6,300 |
| L04 | 5 | 48,238 | 9,647 |
| L05 | 5 | 33,819 | 6,763 |
| L06 | 5 | 40,039 | 8,007 |
| L07 | 5 | 31,890 | 6,378 |
| L08 | 5 | 32,423 | 6,484 |
| M01 | 5 | 51,782 | 10,356 |
| M02 | 5 | 39,609 | 7,921 |
| M03 | 5 | 41,384 | 8,276 |
| M04 | 5 | 46,434 | 9,286 |
| M05 | 5 | 50,472 | 10,094 |
| T01 | 5 | 39,006 | 7,801 |
| T02 | 5 | 39,955 | 7,991 |
| T03 | 5 | 43,108 | 8,621 |
| T04 | 5 | 45,813 | 9,162 |
| T05 | 5 | 78,602 | 15,720 |

## Failure triage

No failed trials (or not graded yet).

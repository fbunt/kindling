# Compare - gemini-3.1-pro-preview vs gemini-3.8-flash

- A: `.bench-runs/pro-vs-flash/gemini-3.1-pro-preview` (125 graded trials)
- B: `.bench-runs/pro-vs-flash/gemini-3.8-flash` (125 graded trials)
- 25 questions; parquet identity `29d168015b13`; grader `ee57b57a09b8`
- question-clustered bootstrap: 10,000 resamples; sign-flip test: 10,000 Monte Carlo permutations; seed 0
- infra-error trials are excluded; model-caused failures count as wrong

## Review status

| model | pending review | unparseable adjudications | stale adjudications |
| --- | --- | --- | --- |
| gemini-3.1-pro-preview | 0 | 0 | 0 |
| gemini-3.8-flash | 0 | 0 | 0 |

Every flagged trial has an applied adjudication.

## Macro accuracy (equal weight per question)

| model | pass@1 | 95% CI | pass^5 |
| --- | --- | --- | --- |
| gemini-3.1-pro-preview | 100.0% | [100.0%, 100.0%] | 100.0% |
| gemini-3.8-flash | 97.6% | [93.6%, 100.0%] | 92.0% |

pass^5: probability that 5 trials of a question all pass (unbiased estimator C(c,5)/C(n,5), k = fewest graded trials of any question), averaged over questions.

## Paired difference (A - B)

- mean per-question difference: +2.4 pp, 95% CI [+0.0 pp, +6.4 pp]
- sign-flip permutation p = 0.5105 (two-sided)
- questions where the models differ: 2 of 25

### Questions where the models disagree

| id | category | A | B | A - B |
| --- | --- | --- | --- | --- |
| A06 | aggregation | 5/5 | 3/5 | +40.0 pp |
| T02 | trend | 5/5 | 4/5 | +20.0 pp |

## Per category (descriptive only; no interval at this n)

| category | n questions | A | B | A - B |
| --- | --- | --- | --- | --- |
| lookup | 8 | 100.0% | 100.0% | +0.0 pp |
| aggregation | 7 | 100.0% | 94.3% | +5.7 pp |
| trend | 5 | 100.0% | 96.0% | +4.0 pp |
| multistep | 5 | 100.0% | 100.0% | +0.0 pp |

## Tokens, cost and latency

| model | trials | tokens | est. USD | tokens per correct | med turn (s) | med model time (s) | med code-judge time (s) | med sandbox time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 125 | 1,068,460 | $3.21 | 8,547 | 12.6 | 9.2 | 0.7 | 1.3 |
| gemini-3.8-flash | 125 | 2,478,402 | $2.32 | 20,314 | 11.3 | 6.2 | 1.3 | 2.7 |

Tokens and cost cover every generate_content call of the graded trials (chat rounds and code-judge verdicts, retried attempts included; no grading calls). Latency medians are per trial: model and code-judge time sum the final attempt's call durations; sandbox time sums query latencies.

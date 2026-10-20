# Decision records

Design decisions for `kindling` whose rationale is not visible in the code or `CLAUDE.md`. One
file per decision that someone would reopen without knowing its rationale. Formatting,
tooling, and anything the code already shows do not get a record. New records copy
`0000-template.md` and take the next number. A changed decision gets a new record and the old
one's status is marked superseded, never rewritten.

- [0001](0001-one-vm-per-user-session.md) Deploy one VM per user session, not a multi-tenant service -- Accepted
- [0002](0002-container-is-the-sole-security-boundary.md) Make the worker container the sole security boundary; no AST/blocklist layer; LLM guards fail open -- Accepted
- [0003](0003-spawn-workers-as-siblings-via-host-runtime-socket.md) Spawn worker containers as siblings via the host runtime socket, not a nested runtime -- Accepted
- [0004](0004-score-the-bench-strictly-with-rules-fixed-before-grading.md) Score the bench strictly, with rules fixed before grading and judged in Python -- Accepted
- [0005](0005-score-the-extended-tier-per-answer-kind-under-its-own-grader.md) Score the extended bench tier per answer kind, under its own grader version -- Accepted

# 0002. Make the worker container the sole security boundary; no AST/blocklist layer; LLM guards fail open

_Status: Accepted. Recorded 2026-09-22. Decided 2026-06-03._

## Context

Until June 2026 model-generated code ran inside the web-server process behind an AST
blocklist (forbidden imports/names, restricted builtins, stripped imports). That layer was
brittle (each blocked construct broke a legitimate analysis) and unsound as a boundary
(Python sandboxing by source inspection is bypassable). An opt-in container pool was added
on 2026-06-02; the question was whether to keep both layers or commit to one.

## Decision

Query code runs only inside a locked-down, per-turn ephemeral container (`--network none`,
`--read-only`, `--cap-drop ALL`, non-root, cgroup memory/pid limits, parquet mounted
read-only). Because the container is the boundary, the worker exposes full Python builtins
and every installed library, and the AST/blocklist code was deleted rather than kept as
belt-and-braces. The in-process path was removed entirely so there is no mode in which code
escapes the container; startup fails fast without a container runtime.

Two flash-lite LLM checks (a prompt-guard on user messages, a code-judge on generated code)
remain as defense-in-depth against abuse and injection, and they **fail open** on judge
error: blocking a user because a secondary model call timed out was judged worse than letting
contained code run.

Rejected: keeping the AST filter alongside the container (false sense of a second boundary,
ongoing breakage of legitimate code); making the guards fail closed (availability cost with no
security gain, since the container contains the code regardless).

## Consequences

- Any hardening effort goes into the container (gVisor `runsc` is the recommended next step
  for kernel-CVE isolation), never into source filtering.
- Guard verdicts must never be relied on as a security control; they reduce abuse, they do
  not prevent it.
- Rootful Docker weakens the boundary (container root maps to host root); rootless Podman
  or userns-remap is expected, and the app warns otherwise.

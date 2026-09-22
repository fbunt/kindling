# 0001. Deploy one VM per user session, not a multi-tenant service

_Status: Accepted. Recorded 2026-09-22._

## Context

`kindling` runs model-generated Python against a large dataset. Where cross-user isolation
lives shapes everything downstream: how hard the sandbox must be, how many workers to keep
warm, whether per-turn container churn is affordable, and how much the LLM guards must
catch. The options were a shared multi-tenant web service with strong in-process isolation,
or one VM per user session with the VM as the isolation boundary.

## Decision

One VM per user session. Within a VM there is effectively a single user issuing turns
sequentially, so cross-user isolation is the VM boundary and nothing inside the app has to
provide it. A shared multi-tenant service lost because it would force hardening the sandbox
and the guards against user-to-user attacks, optimizing for concurrent throughput, and sizing
memory as a shared budget, none of which the intended use (a researcher and their dataset)
needs.

## Consequences

- Sandbox pools are sized small (`KINDLING_POOL_SIZE` 1–2) and per-turn kill-and-respawn of
  worker containers is affordable; nothing is optimized for cross-user contention.
- `KINDLING_SANDBOX_MEM` defaults to 110g because it is a per-VM budget, not a share; real
  queries push 80–90 GB.
- The prompt-guard's blind spot (client-supplied history and images reach the model
  unscreened) is an accepted limitation, not a bug.
- Every one of the above must be revisited if the deployment model changes.

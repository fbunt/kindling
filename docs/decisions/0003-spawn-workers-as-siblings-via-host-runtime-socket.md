# 0003. Spawn worker containers as siblings via the host runtime socket, not a nested runtime

_Status: Accepted. Recorded 2026-09-22. Decided 2026-06-04._

## Context

Once the FastAPI app itself was to run in a container (so the same images run on Silverblue
dev machines and GCP VMs), the workers it spawns had to run somewhere. The options were
nesting a container runtime inside the app container (podman-in-podman, privileged), or
mounting the host runtime socket into the app container and launching workers on the host
runtime as siblings.

## Decision

Siblings via the host socket ("Option A"). The app container mounts the host Podman/Docker
socket, sets `CONTAINER_HOST`, and uses `podman-remote` to spawn workers on the host. The host
kernel enforces the cgroup limits and the worker container remains the sole boundary
(see [0002](0002-container-is-the-sole-security-boundary.md)). Nesting lost because it
requires a privileged app container (which widens the attack surface of the very process that
handles untrusted model output), doubles the runtime layers that enforce limits, and was
harder to validate on rootless Podman.

## Consequences

- The app's view of the parquet differs from the host path workers bind-mount, hence the
  separate `KINDLING_WORKER_PARQUET_PATH`.
- The worker image must exist in the **host** image store; the app cannot build it.
- The host socket mount needs `--security-opt label=disable` under SELinux, and any process
  with access to that socket can control the host runtime, which is acceptable only because
  of [0001](0001-one-vm-per-user-session.md).
- All instances on a host share one runtime, which is why orphan reaping is opt-in.

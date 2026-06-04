"""SandboxPool concurrency/edge-case tests — no podman.

These fake the container-launch seam (`_launch_worker`) and the module-level
`_podman` helpers, so the pool's real concurrency logic (semaphore bounding,
checkout timeout, discard/refill, drain) is exercised without spawning anything.
"""

import asyncio

import pytest

import app.sandbox.pool as pool_mod
from app.sandbox.pool import (
    SandboxBusy,
    SandboxPool,
    Worker,
    WorkerDead,
    detect_runtime,
)

# --- runtime detection (no runtime installed needed) ---


def test_detect_runtime_honors_env(monkeypatch):
    monkeypatch.setenv("KINDLING_CONTAINER_RUNTIME", "docker")
    assert detect_runtime() == "docker"


def test_detect_runtime_prefers_podman(monkeypatch):
    monkeypatch.delenv("KINDLING_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(pool_mod.shutil, "which", lambda x: f"/usr/bin/{x}")
    assert detect_runtime() == "podman"


def test_detect_runtime_falls_back_to_docker(monkeypatch):
    monkeypatch.delenv("KINDLING_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        pool_mod.shutil, "which", lambda x: "/usr/bin/docker" if x == "docker" else None
    )
    assert detect_runtime() == "docker"


def test_detect_runtime_none_raises(monkeypatch):
    monkeypatch.delenv("KINDLING_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(pool_mod.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError):
        detect_runtime()


# --- worker parquet path decoupling (Option A: app in a container) ---


def test_worker_parquet_path_is_separate_and_unresolved():
    # The app reads its own (resolved) view; sibling workers mount the HOST path
    # verbatim (it need not exist in the app container, so it isn't resolved).
    pool = SandboxPool(
        "/tmp/fake.parquet",
        runtime="podman",
        worker_parquet_path="/host/real.parquet",
    )
    assert pool.parquet_path == "/tmp/fake.parquet"
    assert pool.worker_parquet_path == "/host/real.parquet"


def test_worker_parquet_path_defaults_to_app_path():
    # No override → app and workers share one filesystem (non-container case).
    pool = SandboxPool("/tmp/fake.parquet", runtime="podman")
    assert pool.worker_parquet_path == pool.parquet_path


def test_pool_defaults_to_all_cores():
    # No cap → workers use all host cores (no --cpus, polars auto-detects); the
    # pid cap is generous (all-cores workloads run many threads).
    pool = SandboxPool("/tmp/fake.parquet", runtime="podman")
    assert pool.cpus is None
    assert pool.max_threads is None
    assert pool.pids == 8192


# --- Worker.request: a dead/closed worker must surface WorkerDead, not crash ---


async def test_request_raises_workerdead_when_proc_exited():
    class FakeProc:
        returncode = 137  # OOM-killed
        stdin = None

    w = Worker(name="w", proc=FakeProc())
    with pytest.raises(WorkerDead):
        await w.request({"op": "ping"}, timeout=1)


async def test_request_maps_closed_transport_to_workerdead():
    class FakeStdin:
        def write(self, _data):
            # what uvloop raises writing to a dead worker's closed transport
            raise RuntimeError("unable to perform operation; the handler is closed")

        async def drain(self):
            pass

    class FakeProc:
        returncode = None  # not yet reaped, but the pipe is gone
        stdin = FakeStdin()

    w = Worker(name="w", proc=FakeProc())
    with pytest.raises(WorkerDead):
        await w.request({"op": "run_query", "code": "x"}, timeout=1)


class FakeWorker:
    def __init__(self, name: str):
        self.name = name
        self.stderr_task = None
        self.killed = False

    async def kill(self):
        self.killed = True


def make_pool(monkeypatch, **kw):
    """A pool whose container launch + runtime teardown probes are faked.

    runtime is pinned so detect_runtime() isn't invoked (these run without any
    container runtime installed)."""
    kw.setdefault("runtime", "podman")
    pool = SandboxPool("/tmp/fake.parquet", **kw)
    created: list[FakeWorker] = []
    counter = {"n": 0}

    async def fake_launch(*_a, **_k):
        counter["n"] += 1
        w = FakeWorker(f"w{counter['n']}")
        created.append(w)
        return w

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(pool, "_launch_worker", fake_launch)
    monkeypatch.setattr(pool, "_reap_orphans", noop)
    monkeypatch.setattr(pool, "_probe_limits", noop)
    return pool, created


async def test_checkout_timeout_raises_busy(monkeypatch):
    pool, _ = make_pool(monkeypatch, size=1, max_total=1, checkout_timeout=0.05)
    # No start() → _ready is empty → checkout times out.
    with pytest.raises(SandboxBusy):
        await pool.acquire_session()


async def test_start_fills_pool(monkeypatch):
    pool, created = make_pool(monkeypatch, size=2, max_total=4)
    await pool.start()
    assert pool._ready.qsize() == 2
    assert len(created) == 2


async def test_discard_and_refill_kills_and_refills(monkeypatch):
    pool, _ = make_pool(monkeypatch, size=1, max_total=2)
    await pool.start()
    session = await pool.acquire_session()
    assert pool._ready.qsize() == 0
    await pool._discard_and_refill(session.worker)
    assert session.worker.killed is True
    assert pool._ready.qsize() == 1  # refilled back to size


async def test_release_session_schedules_tracked_task(monkeypatch):
    pool, _ = make_pool(monkeypatch, size=1, max_total=2)
    await pool.start()
    session = await pool.acquire_session()
    pool.release_session(session)
    assert pool._inflight, "release_session should track a background task"
    await asyncio.gather(*list(pool._inflight))  # deterministic: await it
    assert session.worker.killed is True


async def test_max_total_bounds_concurrent(monkeypatch):
    # size=1, max_total=1: one worker; with it checked out and no release, a
    # second checkout finds _ready empty and the semaphore blocks any refill.
    pool, _ = make_pool(monkeypatch, size=1, max_total=1, checkout_timeout=0.05)
    await pool.start()
    await pool.acquire_session()  # holds the only permit
    with pytest.raises(SandboxBusy):
        await pool.acquire_session()


async def test_spawn_failure_releases_semaphore(monkeypatch):
    pool, _ = make_pool(monkeypatch, size=1, max_total=2)

    async def boom(*_a, **_k):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(pool, "_launch_worker", boom)
    with pytest.raises(RuntimeError):
        await pool._spawn()
    # Semaphore fully restored: both permits acquirable without blocking.
    for _ in range(2):
        await asyncio.wait_for(pool._sema.acquire(), timeout=0.2)


async def test_reap_orphans_removes_listed(monkeypatch):
    calls = []

    async def fake_capture(_runtime, *_a):
        return "id1 id2\n"

    async def fake_cli(_runtime, *args):
        calls.append(args)
        return 0

    monkeypatch.setattr(pool_mod, "_cli_capture", fake_capture)
    monkeypatch.setattr(pool_mod, "_cli", fake_cli)
    pool = SandboxPool("/tmp/fake.parquet", runtime="podman")
    await pool._reap_orphans()
    assert ("rm", "-f", "id1", "id2") in calls


async def test_drain_kills_ready_and_blocks_refill(monkeypatch):
    pool, created = make_pool(monkeypatch, size=2, max_total=3)
    await pool.start()
    session = await pool.acquire_session()  # hold one permit
    ready_worker = next(w for w in created if w is not session.worker)

    await pool.drain()
    assert pool._closing is True
    assert pool._ready.qsize() == 0
    assert ready_worker.killed is True  # the ready worker was retired

    # Releasing the still-held worker after drain must NOT refill.
    await pool._discard_and_refill(session.worker)
    assert session.worker.killed is True
    assert pool._ready.qsize() == 0

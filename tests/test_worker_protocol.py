"""Worker protocol tests — request ids, dead-marking, kill-on-timeout. No podman.

`Worker.proc` is faked with an asyncio.StreamReader for stdout (the test feeds
frames) and a recording stdin, so the id tagging / mismatch handling in
Worker.request and the kill path in SandboxSession.run_query run for real
without a container.
"""

import asyncio
import json
import logging

import pytest

import app.sandbox.pool as pool_mod
from app.sandbox.pool import SandboxPool, SandboxSession, Worker, WorkerDead


class FakeStdin:
    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        pass


class FakeProc:
    def __init__(self):
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stdin = FakeStdin()
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


def _feed(proc: FakeProc, *frames) -> None:
    for f in frames:
        line = f if isinstance(f, str) else json.dumps(f)
        proc.stdout.feed_data(line.encode() + b"\n")


async def test_request_tags_payload_with_id_and_strips_it_from_reply():
    proc = FakeProc()
    w = Worker(name="w", proc=proc)
    _feed(proc, {"id": 0, "op": "pong"})
    resp = await w.request({"op": "ping"}, timeout=1)
    assert resp == {"op": "pong"}  # id stripped
    sent = json.loads(proc.stdin.writes[0])
    assert sent == {"op": "ping", "id": 0}
    # ids are monotonic per worker
    _feed(proc, {"id": 1, "data": "x"})
    await w.request({"op": "run_query", "code": "x"}, timeout=1)
    assert json.loads(proc.stdin.writes[1])["id"] == 1


async def test_stale_or_untagged_frames_are_dropped(caplog):
    proc = FakeProc()
    w = Worker(name="w", proc=proc)
    _feed(
        proc,
        {"id": 999, "data": "stale"},  # late reply to some other request
        {"data": "forged"},  # no id at all (query code forging a frame)
        {"id": None, "error": "malformed request"},  # worker's null-id reply
        "[1, 2, 3]",  # valid JSON, not a dict
        {"id": 0, "data": "fresh"},
    )
    with caplog.at_level(logging.WARNING, logger="app.sandbox.pool"):
        resp = await w.request({"op": "run_query", "code": "x"}, timeout=1)
    assert resp == {"data": "fresh"}
    dropped = [r for r in caplog.records if "dropping frame with id" in r.message]
    assert len(dropped) == 4
    assert "expected 0" in dropped[0].message


async def test_timeout_marks_dead_and_next_request_fails_without_writing():
    proc = FakeProc()
    w = Worker(name="w", proc=proc)
    with pytest.raises(WorkerDead):
        await w.request({"op": "run_query", "code": "x"}, timeout=0.05)
    assert w.dead is True
    assert len(proc.stdin.writes) == 1
    with pytest.raises(WorkerDead, match="marked dead"):
        await w.request({"op": "run_query", "code": "y"}, timeout=1)
    assert len(proc.stdin.writes) == 1  # nothing written to the desynced pipe


async def test_eof_marks_dead():
    proc = FakeProc()
    proc.stdout.feed_eof()
    w = Worker(name="w", proc=proc)
    with pytest.raises(WorkerDead, match="EOF"):
        await w.request({"op": "ping"}, timeout=1)
    assert w.dead is True


async def test_kill_is_idempotent(monkeypatch):
    cli_calls = []

    async def fake_cli(_runtime, *args):
        cli_calls.append(args)
        return 0

    monkeypatch.setattr(pool_mod, "_cli", fake_cli)
    proc = FakeProc()
    w = Worker(name="w", proc=proc)
    await w.kill()
    await w.kill()
    assert proc.kill_calls == 1
    assert [c for c in cli_calls if c[0] == "kill"] == [("kill", "w")]
    assert w.dead is True


async def test_session_kills_worker_on_dead_and_pool_releases_permit_once(
    monkeypatch,
):
    cli_calls = []

    async def fake_cli(_runtime, *args):
        cli_calls.append(args)
        return 0

    async def no_refill(*_a, **_k):
        return None

    monkeypatch.setattr(pool_mod, "_cli", fake_cli)
    pool = SandboxPool("/tmp/fake.parquet", runtime="podman", size=1, max_total=2)
    pool.hard_timeout = 0.05
    monkeypatch.setattr(pool, "_spawn_into_ready", no_refill)
    # Simulate a spawned worker holding its permit (as _spawn would leave it).
    await pool._sema.acquire()
    proc = FakeProc()
    worker = Worker(name="w", proc=proc)
    session = SandboxSession(worker=worker, pool=pool)

    out = await session.run_query("while True: pass")  # no reply → host timeout
    assert out == {"error": pool_mod._WORKER_DEAD_MSG}
    assert worker.dead is True
    assert proc.kill_calls == 1  # killed right away, not left alive
    assert ("kill", "w") in cli_calls

    # A second run_query in the same turn fails fast, without writing or killing.
    out2 = await session.run_query("result = 1")
    assert out2 == {"error": pool_mod._WORKER_DEAD_MSG}
    assert len(proc.stdin.writes) == 1
    assert proc.kill_calls == 1

    # Turn end: release → _retire re-kills (no-op) and releases exactly one permit.
    pool.release_session(session)
    await asyncio.gather(*list(pool._inflight))
    assert proc.kill_calls == 1
    assert pool._sema._value == 2  # BoundedSemaphore would raise on a 2nd release


async def test_failed_launch_teardown_is_bounded_and_removes_container(
    monkeypatch, caplog
):
    """A failed launch must `rm -f` the container and never block on
    proc.wait(): asyncio only resolves wait() once every stdio pipe is
    disconnected, which a wrapper `podman` (host-exec shim) can hold open
    after the direct child is SIGKILLed."""
    cli_calls = []

    async def fake_cli(_runtime, *args):
        cli_calls.append(args)
        return 0

    class HangingProc(FakeProc):
        def __init__(self):
            super().__init__()
            self.stderr = asyncio.StreamReader()
            self.stderr.feed_data(b"Error: something went wrong\n")
            self.stderr.feed_eof()
            self.closed = False
            self._transport = self

        def close(self) -> None:
            self.closed = True

        async def wait(self) -> int:  # pipes never disconnect
            await asyncio.Event().wait()
            return 0  # pragma: no cover

    proc = HangingProc()

    async def fake_exec(*_argv, **_kw):
        return proc

    async def no_pong(self, *_a, **_k):
        raise WorkerDead("no pong")

    monkeypatch.setattr(pool_mod, "_cli", fake_cli)
    monkeypatch.setattr(pool_mod, "_LAUNCH_REAP_TIMEOUT", 0.05)
    monkeypatch.setattr(pool_mod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(Worker, "ping", no_pong)
    pool = SandboxPool("/tmp/fake.parquet", runtime="podman", size=1, max_total=2)

    with caplog.at_level(logging.WARNING, logger="app.sandbox.pool"):
        with pytest.raises(WorkerDead, match="no pong"):
            await asyncio.wait_for(pool._launch_worker(), 2.0)

    rm_calls = [c for c in cli_calls if c[:2] == ("rm", "-f")]
    assert len(rm_calls) == 1 and rm_calls[0][2].startswith(pool_mod._NAME_PREFIX)
    assert proc.kill_calls == 1
    assert proc.closed is True  # transport closed so wait() can't stay pending
    assert any("did not exit within" in r.getMessage() for r in caplog.records)

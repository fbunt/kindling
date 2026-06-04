"""Host-side warm pool of sandbox containers.

Each chat turn checks out a pre-started Podman worker container (no cold-start on
the critical path), uses it for the turn's run_query calls, and on turn end the
container is killed and a fresh one spawned in the background to refill the pool.
One container per turn gives pristine per-turn isolation; at one-VM-per-user with
sequential turns a size 1-2 pool absorbs the spawn churn easily.

See app/sandbox/worker.py for the in-container counterpart and the JSONL protocol.
"""

import asyncio
import base64
import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.query_engine import PLOTS_DIR

logger = logging.getLogger(__name__)

_STREAM_LIMIT = 8 * 1024 * 1024  # base64 PNG frames overflow the 64KB default
_NAME_PREFIX = "kindling-worker-"
_PING_TIMEOUT = 30.0
_WORKER_DEAD_MSG = (
    "The sandbox worker terminated unexpectedly (it may have run out of memory "
    "or timed out). Try a smaller or simpler query."
)


class WorkerDead(Exception):
    """The worker subprocess/container died or stopped responding."""


class SandboxBusy(Exception):
    """No worker became available within the checkout timeout."""


def podman_available() -> bool:
    return shutil.which("podman") is not None


def build_podman_argv(
    *,
    image: str,
    name: str,
    host_parquet_path: str,
    in_container_path: str,
    memory: str,
    cpus: str,
    pids: int,
    max_threads: int,
) -> list[str]:
    """Construct the hardened rootless `podman run` argv for one worker."""
    return [
        "podman",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=512m,noexec,nosuid",
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--cpus",
        cpus,
        "--pids-limit",
        str(pids),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--userns=keep-id",
        # :ro,z relabels for SELinux (Fedora host). If the mount source can't be
        # relabeled, swap for `--security-opt label=disable`.
        "-v",
        f"{host_parquet_path}:{in_container_path}:ro,z",
        "-e",
        f"KINDLING_PARQUET_PATH={in_container_path}",
        "-e",
        f"POLARS_MAX_THREADS={max_threads}",
        image,
    ]


async def _podman(*args: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        "podman",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode


async def _podman_capture(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "podman",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return out.decode()


def materialize_plots(b64_pngs: list[str], turn_id: str) -> list[str]:
    """Decode base64 PNGs from the worker to disk; return /plots URLs.

    Filenames are prefixed with the turn id (not a shared counter) so concurrent
    turns can't collide. Returns the worker's plot URLs for tools.py to name.
    """
    urls = []
    for i, b64 in enumerate(b64_pngs):
        name = f"plot-{turn_id}-{i:03d}.png"
        (PLOTS_DIR / name).write_bytes(base64.b64decode(b64))
        urls.append(f"/plots/{name}?t={int(time.time())}")
    return urls


@dataclass
class Worker:
    name: str
    proc: asyncio.subprocess.Process
    stderr_task: "asyncio.Task | None" = None

    async def _read_frame(self) -> dict:
        """Read one JSON frame, skipping any non-JSON noise on the pipe."""
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise WorkerDead("worker stdout closed (EOF)")
            text = line.strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "sandbox %s: dropping non-JSON line: %.200r", self.name, text
                )

    async def request(self, payload: dict, timeout: float) -> dict:
        try:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode())
            await self.proc.stdin.drain()
            return await asyncio.wait_for(self._read_frame(), timeout)
        except (TimeoutError, WorkerDead, BrokenPipeError, ConnectionResetError) as e:
            raise WorkerDead(str(e) or type(e).__name__) from e

    async def ping(self, timeout: float = _PING_TIMEOUT) -> None:
        resp = await self.request({"op": "ping"}, timeout)
        if resp.get("op") != "pong":
            raise WorkerDead(f"unexpected ping response: {resp!r}")

    async def run_query(self, code: str, timeout: float) -> dict:
        return await self.request({"op": "run_query", "code": code}, timeout)

    async def kill(self) -> None:
        if self.stderr_task is not None:
            self.stderr_task.cancel()
        try:
            await _podman("kill", self.name)
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        try:
            if self.proc.returncode is None:
                self.proc.kill()
            await self.proc.wait()
        except ProcessLookupError:
            pass


@dataclass
class SandboxSession:
    """Per-turn handle threaded through run_chat_turn in place of a namespace."""

    worker: Worker
    pool: "SandboxPool"
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    async def run_query(self, code: str) -> dict:
        try:
            result = await self.worker.run_query(code, self.pool.hard_timeout)
        except WorkerDead as e:
            logger.warning("sandbox worker %s died mid-turn: %s", self.worker.name, e)
            return {"error": _WORKER_DEAD_MSG}
        if result.get("plots"):
            result["plots"] = materialize_plots(result["plots"], self.turn_id)
        return result


class SandboxPool:
    def __init__(
        self,
        parquet_path: Path | str,
        *,
        size: int = 2,
        max_total: int = 3,
        image: str = "kindling-worker:latest",
        in_container_path: str = "/data/dataset.parquet",
        hard_timeout: float = 510.0,
        checkout_timeout: float = 120.0,
        memory: str = "110g",
        cpus: str = "1.0",
        pids: int = 128,
        max_threads: int = 4,
    ) -> None:
        self.parquet_path = str(Path(parquet_path).resolve())
        self.size = size
        self.image = image
        self.in_container_path = in_container_path
        self.hard_timeout = hard_timeout
        self.checkout_timeout = checkout_timeout
        self.memory = memory
        self.cpus = cpus
        self.pids = pids
        self.max_threads = max_threads
        self._ready: asyncio.Queue[Worker] = asyncio.Queue()
        self._sema = asyncio.BoundedSemaphore(max_total)
        self._closing = False

    async def start(self) -> None:
        await self._reap_orphans()
        await self._probe_limits()
        await asyncio.gather(*(self._spawn_into_ready() for _ in range(self.size)))
        logger.info(
            "sandbox pool ready: %d/%d warm workers", self._ready.qsize(), self.size
        )

    async def acquire_session(self) -> SandboxSession:
        worker = await self._checkout()
        return SandboxSession(worker=worker, pool=self)

    def release_session(self, session: SandboxSession) -> None:
        # Fire-and-forget so the turn's finally never blocks on a cold start.
        asyncio.create_task(self._discard_and_refill(session.worker))

    async def drain(self) -> None:
        self._closing = True
        workers = []
        while not self._ready.empty():
            workers.append(self._ready.get_nowait())
        await asyncio.gather(
            *(self._retire(w) for w in workers), return_exceptions=True
        )
        logger.info("sandbox pool drained")

    # --- internals ---

    async def _checkout(self) -> Worker:
        try:
            return await asyncio.wait_for(self._ready.get(), self.checkout_timeout)
        except TimeoutError as e:
            raise SandboxBusy("no sandbox worker available") from e

    async def _spawn(self) -> Worker:
        await self._sema.acquire()
        proc = None
        name = f"{_NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        try:
            argv = build_podman_argv(
                image=self.image,
                name=name,
                host_parquet_path=self.parquet_path,
                in_container_path=self.in_container_path,
                memory=self.memory,
                cpus=self.cpus,
                pids=self.pids,
                max_threads=self.max_threads,
            )
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_LIMIT,
            )
            worker = Worker(name=name, proc=proc)
            worker.stderr_task = asyncio.create_task(self._drain_stderr(worker))
            await worker.ping()
            return worker
        except BaseException:
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await _podman("rm", "-f", name)
            self._sema.release()
            raise

    async def _spawn_into_ready(self) -> None:
        try:
            worker = await self._spawn()
        except Exception:
            logger.exception("sandbox: failed to spawn worker")
            return
        if self._closing:
            await self._retire(worker)
            return
        await self._ready.put(worker)

    async def _retire(self, worker: Worker) -> None:
        await worker.kill()
        self._sema.release()

    async def _discard_and_refill(self, worker: Worker) -> None:
        await self._retire(worker)
        if not self._closing and self._ready.qsize() < self.size:
            await self._spawn_into_ready()

    async def _drain_stderr(self, worker: Worker) -> None:
        try:
            while True:
                line = await worker.proc.stderr.readline()
                if not line:
                    return
                logger.debug(
                    "sandbox %s: %s",
                    worker.name,
                    line.decode(errors="replace").rstrip(),
                )
        except Exception:  # noqa: BLE001
            return

    async def _reap_orphans(self) -> None:
        """Remove worker containers leaked by a prior hard-killed server."""
        try:
            out = await _podman_capture("ps", "-aq", "--filter", f"name={_NAME_PREFIX}")
            ids = out.split()
            if ids:
                logger.warning("sandbox: reaping %d orphaned container(s)", len(ids))
                await _podman("rm", "-f", *ids)
        except Exception:  # noqa: BLE001
            logger.debug("sandbox: orphan reap failed", exc_info=True)

    async def _probe_limits(self) -> None:
        """Warn (don't fail) if cgroup memory limits aren't enforced here."""
        try:
            out = await _podman_capture(
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "256m",
                self.image,
                "sh",
                "-c",
                "cat /sys/fs/cgroup/memory.max",
            )
            if out.strip() != str(256 * 1024 * 1024):
                logger.warning(
                    "sandbox: --memory limit not enforced on this host "
                    "(memory.max=%r); workers can exceed %s",
                    out.strip(),
                    self.memory,
                )
        except Exception:  # noqa: BLE001
            logger.debug("sandbox: limit probe failed", exc_info=True)

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
import os
import shutil
import time
import uuid
from collections import deque
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


_RUNTIMES = ("podman", "docker")


def detect_runtime() -> str:
    """Pick a container runtime: KINDLING_CONTAINER_RUNTIME if set, else the first
    of podman/docker on PATH (podman preferred — stronger rootless default)."""
    explicit = os.environ.get("KINDLING_CONTAINER_RUNTIME")
    if explicit:
        return explicit
    for rt in _RUNTIMES:
        if shutil.which(rt):
            return rt
    raise RuntimeError(
        "No container runtime found. Install podman or docker, or set "
        "KINDLING_CONTAINER_RUNTIME."
    )


def runtime_available() -> bool:
    """True if a usable container runtime is available."""
    explicit = os.environ.get("KINDLING_CONTAINER_RUNTIME")
    if explicit:
        return shutil.which(explicit) is not None
    return any(shutil.which(rt) for rt in _RUNTIMES)


def build_run_argv(
    runtime: str,
    *,
    image: str,
    name: str,
    host_parquet_path: str,
    in_container_path: str,
    memory: str,
    cpus: str | None,
    pids: int | None,
    max_threads: int | None,
) -> list[str]:
    """Construct the hardened `run` argv for one worker. Identical across podman
    and docker except `--userns=keep-id`, which is podman-only (rootless UID
    mapping; docker has no equivalent).

    cpus=None  -> no CPU cap; the worker uses all host cores.
    max_threads=None -> POLARS_MAX_THREADS unset; polars/BLAS auto-detect cores.
    """
    argv = [
        runtime,
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
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    if pids:
        argv += ["--pids-limit", str(pids)]
    if cpus:
        argv += ["--cpus", cpus]
    if runtime == "podman":
        argv.append("--userns=keep-id")
    argv += [
        # :ro,z relabels for SELinux (no-op off SELinux on both runtimes).
        "-v",
        f"{host_parquet_path}:{in_container_path}:ro,z",
        "-e",
        f"KINDLING_PARQUET_PATH={in_container_path}",
    ]
    if max_threads:
        argv += ["-e", f"POLARS_MAX_THREADS={max_threads}"]
    argv.append(image)
    return argv


async def _cli(runtime: str, *args: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        runtime,
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode


async def _cli_capture(runtime: str, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        runtime,
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
    runtime: str = "podman"
    stderr_task: "asyncio.Task | None" = None
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=20))

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
        # If the container already exited (e.g. OOM-killed by a prior heavy query
        # in this turn), writing to its stdin would raise; surface WorkerDead.
        if self.proc.returncode is not None:
            raise WorkerDead(f"worker already exited (rc={self.proc.returncode})")
        try:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode())
            await self.proc.stdin.drain()
            return await asyncio.wait_for(self._read_frame(), timeout)
        except (TimeoutError, WorkerDead) as e:
            raise WorkerDead(str(e) or type(e).__name__) from e
        except (BrokenPipeError, ConnectionResetError, RuntimeError, ValueError) as e:
            # Writing/draining a closed pipe of a dead worker. uvloop raises a
            # plain RuntimeError ("handler is closed"); asyncio raises
            # ConnectionResetError/ValueError. All mean the worker is gone.
            raise WorkerDead(f"worker pipe closed: {type(e).__name__}: {e}") from e

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
            await _cli(self.runtime, "kill", self.name)
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        try:
            if self.proc.returncode is None:
                self.proc.kill()
            await self.proc.wait()
        except ProcessLookupError:
            pass
        # Close the subprocess transport while the loop is alive (idempotent;
        # subsumes the stdin/stdout/stderr pipes). Otherwise its __del__ runs at
        # GC after the loop closes and emits "Event loop is closed".
        try:
            transport = getattr(self.proc, "_transport", None)
            if transport is not None:
                transport.close()
        except Exception:  # noqa: BLE001
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
            tail = "\n".join(self.worker.stderr_tail)
            logger.warning(
                "sandbox worker %s died mid-turn: %s%s",
                self.worker.name,
                e,
                f"\n--- worker stderr tail ---\n{tail}" if tail else "",
            )
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
        cpus: str | None = None,  # None → no CPU cap (worker uses all host cores)
        # All-cores workers run many threads (polars rayon + OpenBLAS + sklearn
        # OpenMP, each ~core-count), so the pid cap must be generous; it still
        # bounds a runaway fork bomb. None → no cap.
        pids: int | None = 8192,
        max_threads: int | None = None,  # None → polars/BLAS auto-detect cores
        runtime: str | None = None,
        worker_parquet_path: str | None = None,
    ) -> None:
        self.parquet_path = str(Path(parquet_path).resolve())
        # Host path bind-mounted into workers. When the app itself runs in a
        # container (Option A), workers are siblings on the HOST runtime, so their
        # mount source is a host path that differs from the app's own view — and
        # must NOT be resolved against the app container's filesystem. Defaults to
        # the app's path (correct when app and workers share one filesystem).
        self.worker_parquet_path = worker_parquet_path or self.parquet_path
        self.runtime = runtime or detect_runtime()
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
        self._inflight: set[asyncio.Task] = set()  # in-flight discard/refill tasks

    async def start(self) -> None:
        await self._log_runtime()
        await self._reap_orphans()
        await self._probe_limits()
        await asyncio.gather(*(self._spawn_into_ready() for _ in range(self.size)))
        logger.info(
            "sandbox pool ready: %d/%d warm workers", self._ready.qsize(), self.size
        )

    async def _log_runtime(self) -> None:
        """Log the runtime and warn if it looks rootful (weaker boundary).

        rootless (podman, or docker rootless): a container escape lands on an
        unprivileged user. Rootful docker maps container-root to host-root, so the
        same flags are a weaker wall. Best-effort — never blocks startup."""
        rootless = None
        try:
            if self.runtime == "podman":
                out = await _cli_capture(
                    self.runtime, "info", "--format", "{{.Host.Security.Rootless}}"
                )
                rootless = out.strip().lower() == "true"
            else:  # docker
                out = await _cli_capture(
                    self.runtime, "info", "--format", "{{.SecurityOptions}}"
                )
                rootless = "rootless" in out.lower()
        except Exception:  # noqa: BLE001 — diagnostics only
            logger.debug("could not determine runtime rootless mode", exc_info=True)
        logger.info("sandbox runtime: %s (rootless=%s)", self.runtime, rootless)
        if rootless is False:
            logger.warning(
                "sandbox runtime %s appears rootful: a container escape maps to "
                "host root, weakening the sandbox boundary. Prefer rootless "
                "podman, or rootless/userns-remap docker.",
                self.runtime,
            )

    async def acquire_session(self) -> SandboxSession:
        worker = await self._checkout()
        return SandboxSession(worker=worker, pool=self)

    def release_session(self, session: SandboxSession) -> None:
        # Fire-and-forget so the turn's finally never blocks on a cold start, but
        # track the task so drain() can wait for it (no leaked container on shutdown).
        task = asyncio.create_task(self._discard_and_refill(session.worker))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def drain(self) -> None:
        self._closing = True
        # Wait for in-flight discard/refill tasks so their workers are killed and
        # no refill spawns a container after we've "drained" (they see _closing).
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
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
        # Bound total live containers; release the permit only on failure (a live
        # worker holds its permit until _retire releases it). _launch_worker is a
        # seam tests fake to exercise this logic without real podman.
        await self._sema.acquire()
        try:
            return await self._launch_worker()
        except BaseException:
            self._sema.release()
            raise

    async def _launch_worker(self) -> Worker:
        proc = None
        name = f"{_NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        try:
            argv = build_run_argv(
                self.runtime,
                image=self.image,
                name=name,
                host_parquet_path=self.worker_parquet_path,
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
            worker = Worker(name=name, proc=proc, runtime=self.runtime)
            worker.stderr_task = asyncio.create_task(self._drain_stderr(worker))
            await worker.ping()
            return worker
        except BaseException:
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await _cli(self.runtime, "rm", "-f", name)
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
        # Worker stderr carries tracebacks + library chatter from user code. Keep
        # it at DEBUG (noisy on normal runs), but retain a tail so a WorkerDead
        # failure can surface the last lines at WARNING (see SandboxSession).
        try:
            while True:
                line = await worker.proc.stderr.readline()
                if not line:
                    return
                text = line.decode(errors="replace").rstrip()
                worker.stderr_tail.append(text)
                logger.debug("sandbox %s: %s", worker.name, text)
        except Exception:  # noqa: BLE001
            return

    async def _reap_orphans(self) -> None:
        """Remove worker containers leaked by a prior hard-killed server."""
        try:
            out = await _cli_capture(
                self.runtime, "ps", "-aq", "--filter", f"name={_NAME_PREFIX}"
            )
            ids = out.split()
            if ids:
                logger.warning("sandbox: reaping %d orphaned container(s)", len(ids))
                await _cli(self.runtime, "rm", "-f", *ids)
        except Exception:  # noqa: BLE001
            logger.debug("sandbox: orphan reap failed", exc_info=True)

    async def _probe_limits(self) -> None:
        """Warn (don't fail) if cgroup memory limits aren't enforced here."""
        try:
            out = await _cli_capture(
                self.runtime,
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "256m",
                # The worker image's ENTRYPOINT is the kernel; override it to run
                # a plain shell for the probe.
                "--entrypoint",
                "sh",
                self.image,
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

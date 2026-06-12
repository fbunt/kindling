import logging
import os
import secrets
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

logging.basicConfig(
    level=os.environ.get("KINDLING_LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("matplotlib.font_manager", "PIL", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from app.query_engine import PLOTS_DIR, configure  # noqa: E402
from app.routes import auth, chat, plots  # noqa: E402

load_dotenv()

# Auto-configure with default dataset when started via uvicorn directly
configure()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/drain the sandbox container pool. Query execution is container-only,
    so a container runtime (podman or docker) is required — fails fast without."""
    from app.sandbox.pool import SandboxPool, runtime_available

    if not runtime_available():
        raise RuntimeError(
            "Query execution runs in containers, but no container runtime was "
            "found. Install podman or docker (or set KINDLING_CONTAINER_RUNTIME) "
            "and build the kindling-worker image."
        )
    # Fresh slate: plots from a prior process are unreachable (frontend state
    # doesn't survive reload; history re-embeds plots as base64), so clear them.
    stale = list(PLOTS_DIR.glob("*.png"))
    for f in stale:
        f.unlink(missing_ok=True)
    if stale:
        logger.info("Cleared %d stale plot(s) from %s", len(stale), PLOTS_DIR)
    # Re-derive the parquet path from the env var configure() exported, not from
    # a module global that an import-time configure() may have set to the default.
    parquet = os.environ.get("KINDLING_PARQUET_PATH_HOST")
    if not parquet:
        raise RuntimeError("Parquet path unknown; was configure() called?")
    # Default: no CPU cap → workers use all host cores (polars/BLAS auto-detect).
    # Set KINDLING_SANDBOX_CPUS=N to cap; the polars thread count is coupled to N
    # so it doesn't over- or under-subscribe the allotted cores.
    cpus = os.environ.get("KINDLING_SANDBOX_CPUS") or None
    max_threads = max(1, int(float(cpus))) if cpus else None
    # pid cap: generous default (all-cores workers run many threads); "0"/"none"
    # disables it. Bounds a runaway fork bomb without starving legit ML threads.
    pids_env = os.environ.get("KINDLING_SANDBOX_PIDS")
    if pids_env is None:
        pids = 8192
    elif pids_env.strip().lower() in ("0", "none", "unlimited"):
        pids = None
    else:
        pids = int(pids_env)
    pool = SandboxPool(
        parquet,
        size=int(os.environ.get("KINDLING_POOL_SIZE", "2")),
        max_total=int(os.environ.get("KINDLING_SANDBOX_MAX_TOTAL", "3")),
        image=os.environ.get("KINDLING_SANDBOX_IMAGE", "kindling-worker:latest"),
        memory=os.environ.get("KINDLING_SANDBOX_MEM", "110g"),
        cpus=cpus,
        pids=pids,
        max_threads=max_threads,
        # When the app runs in a container, workers are siblings on the host
        # runtime; this is the HOST path to bind-mount into them (differs from the
        # app's own `parquet` view). Unset → same filesystem, use `parquet`.
        worker_parquet_path=os.environ.get("KINDLING_WORKER_PARQUET_PATH"),
    )
    await pool.start()
    logger.info("Sandbox: container pool active (parquet=%s)", parquet)
    app.state.sandbox_pool = pool
    try:
        yield
    finally:
        await pool.drain()


app = FastAPI(lifespan=lifespan)

# Random default: restarts invalidate session cookies and multi-worker uvicorn
# won't share sessions; set KINDLING_SESSION_SECRET to fix that. Surviving a
# restart buys little today anyway — the keystore (app/keystore.py) is
# in-memory, so an old cookie's token points at a dropped entry regardless.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("KINDLING_SESSION_SECRET") or secrets.token_hex(32),
)

app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
# Plots are session-gated (see routes/plots.py), not a public static mount.
app.include_router(plots.router)

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

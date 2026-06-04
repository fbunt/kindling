import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

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

from app.query_engine import configure  # noqa: E402
from app.routes import auth, chat  # noqa: E402

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
    # Re-derive the parquet path from the env var configure() exported, not from
    # a module global that an import-time configure() may have set to the default.
    parquet = os.environ.get("KINDLING_PARQUET_PATH_HOST")
    if not parquet:
        raise RuntimeError("Parquet path unknown; was configure() called?")
    pool = SandboxPool(
        parquet,
        size=int(os.environ.get("KINDLING_POOL_SIZE", "2")),
        max_total=int(os.environ.get("KINDLING_SANDBOX_MAX_TOTAL", "3")),
        image=os.environ.get("KINDLING_SANDBOX_IMAGE", "kindling-worker:latest"),
        memory=os.environ.get("KINDLING_SANDBOX_MEM", "110g"),
    )
    await pool.start()
    logger.info("Sandbox: container pool active (parquet=%s)", parquet)
    app.state.sandbox_pool = pool
    try:
        yield
    finally:
        await pool.drain()


app = FastAPI(lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(plots_dir)), name="plots")
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

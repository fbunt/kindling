import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.query_engine import configure
from app.routes import auth, chat

load_dotenv()

# Auto-configure with default dataset when started via uvicorn directly
configure()

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(plots_dir)), name="plots")
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

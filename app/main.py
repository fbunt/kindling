import secrets

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.routes import auth, chat

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

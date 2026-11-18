"""KrishiMitra FastAPI application.

Run locally:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000 (web app) and http://127.0.0.1:8000/api/docs (Swagger API docs).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.db import Database
from api.limiter import limiter
from api.routes import assistant, predict, system
from src.inference.predictor import load_predictor
from src.llm.advisor import Advisor
from src.llm.client import LLMClient
from src.llm.knowledge import KnowledgeBase
from src.llm.rag import ChatAssistant, Retriever
from src.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger("krishimitra.api")


def create_app() -> FastAPI:
    cfg = load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load everything ONCE at startup (loading a model per request would be very slow)."""
        icfg = cfg["inference"]
        app.state.cfg = cfg
        app.state.kb = KnowledgeBase(resolve_path(cfg["knowledge_base"]["path"]))
        app.state.predictor = load_predictor(
            resolve_path(os.getenv("KRISHIMITRA_ONNX_PATH", icfg["onnx_path"])),
            resolve_path(os.getenv("KRISHIMITRA_CLASSES_PATH", icfg["classes_path"])),
            icfg["top_k"], icfg["confidence_threshold"],
        )
        app.state.llm = LLMClient(cfg["llm"]["model"], cfg["llm"]["max_tokens"], cfg["llm"]["effort"])
        app.state.advisor = Advisor(app.state.kb, app.state.llm)
        app.state.chat = ChatAssistant(
            app.state.kb, Retriever(app.state.kb, resolve_path(cfg["knowledge_base"]["docs_dir"])), app.state.llm
        )
        app.state.db = Database(resolve_path(os.getenv("DB_PATH", cfg["api"]["db_path"])))
        log.info("KrishiMitra API ready (model loaded: %s, LLM: %s)",
                 app.state.predictor is not None, app.state.llm.enabled)
        yield

    app = FastAPI(
        title="KrishiMitra API",
        description="AI crop disease detection and farmer advisory. Click a leaf, save your crop.",
        version=cfg["project"]["version"],
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()] or cfg["api"]["allowed_origins"]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})

    for module in (predict, assistant, system):
        app.include_router(module.router, prefix="/api")

    # Serve the PWA (web/) from the same server → one URL for website, API and Android app.
    web_dir = PROJECT_ROOT / "web"
    if (web_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


app = create_app()

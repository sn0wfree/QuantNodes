from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings as app_settings
from .routers import wiki, backtest, factor, skill, dream, stats, strategy, settings as settings_router, prompts, code, agent as agent_router
from .services.nanobot_runtime import init_runtime, shutdown_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("/tmp/server.log"),
            logging.StreamHandler()
        ]
    )

    print(f"Starting QuantNodes API v{app_settings.VERSION}")

    # v3.0.0 Stage 5.3: Single-process nanobot runtime.
    # If nanobot-ai is installed, this starts AgentLoop + ChannelManager +
    # CronService as asyncio.create_task under uvicorn's event loop. The
    # WebUI SPA + WebSocket are served from cfg.gateway.port (default 18080)
    # inside this same Python process.
    # If nanobot-ai is NOT installed, init_runtime() still returns a runtime
    # object but its state stays "unavailable" — /api/agent/* endpoints will
    # return 503 with an install hint.
    runtime = init_runtime()
    await runtime.start()
    app.state.nanobot_runtime = runtime

    # Bridge uvicorn + fastapi stdlib logging into nanobot's loguru stream
    # so we get a single, unified log channel when nanobot is installed.
    try:
        from QuantNodes.agent import NANOBOT_AVAILABLE
        if NANOBOT_AVAILABLE:
            from nanobot.utils.logging_bridge import redirect_lib_logging
            redirect_lib_logging("uvicorn", level="INFO")
            redirect_lib_logging("uvicorn.access", level="INFO")
            redirect_lib_logging("fastapi", level="INFO")
    except Exception:  # pragma: no cover - logging best-effort
        logging.getLogger(__name__).debug("loguru bridge skipped", exc_info=True)

    try:
        yield
    finally:
        print("Shutting down QuantNodes API")
        await shutdown_runtime()


app = FastAPI(
    title="QuantNodes API",
    description="Quantitative Research Platform API (No LLM)",
    version=app_settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(wiki.router, prefix="/api/wiki", tags=["wiki"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(factor.router, prefix="/api/factor", tags=["factor"])
app.include_router(skill.router, prefix="/api/skills", tags=["skills"])
app.include_router(dream.router, prefix="/api/dreams", tags=["dreams"])
app.include_router(strategy.router, prefix="/api/strategy", tags=["strategy"])
app.include_router(settings_router.router, prefix="/api/settings", tags=["settings"])
app.include_router(prompts.router, prefix="/api", tags=["prompts"])
app.include_router(code.router, prefix="/api", tags=["code"])
app.include_router(agent_router.router, prefix="/api/agent", tags=["agent"])


@app.get("/")
async def root():
    return {"message": "QuantNodes API", "version": app_settings.VERSION}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "healthy"}

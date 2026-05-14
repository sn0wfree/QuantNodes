from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings as app_settings
from .routers import wiki, backtest, factor, skill, dream, stats, strategy, settings as settings_router, prompts, code


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
    yield
    print("Shutting down QuantNodes API")


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


@app.get("/")
async def root():
    return {"message": "QuantNodes API", "version": app_settings.VERSION}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "healthy"}

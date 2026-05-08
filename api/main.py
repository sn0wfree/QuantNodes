from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings as app_settings
from .routers import agent, wiki, backtest, factor, skill, dream, stats, strategy, settings as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting QuantNodes API v{app_settings.VERSION}")
    yield
    # Shutdown
    print("Shutting down QuantNodes API")


app = FastAPI(
    title="QuantNodes API",
    description="AI-Powered Quantitative Research Platform API",
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

app.include_router(stats.router, prefix="/api", tags=["stats"])
app.include_router(agent.router, prefix="/api", tags=["agent"])
app.include_router(wiki.router, prefix="/api/wiki", tags=["wiki"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(factor.router, prefix="/api/factor", tags=["factor"])
app.include_router(skill.router, prefix="/api/skills", tags=["skills"])
app.include_router(dream.router, prefix="/api/dreams", tags=["dreams"])
app.include_router(strategy.router, prefix="/api/strategy", tags=["strategy"])
app.include_router(settings_router.router, prefix="/api/settings", tags=["settings"])


@app.get("/")
async def root():
    return {"message": "QuantNodes API", "version": app_settings.VERSION}


@app.get("/health")
async def health():
    return {"status": "healthy"}

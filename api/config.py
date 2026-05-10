from pydantic_settings import BaseSettings
from typing import List, Optional
import os
import json
from pathlib import Path


def _load_agent_from_settings() -> dict:
    """Load agent config from .quant_agent/settings.json (single source of truth)"""
    settings_file = Path(".quant_agent/settings.json")
    if settings_file.exists():
        try:
            data = json.loads(settings_file.read_text(encoding="utf-8"))
            return data.get("agent", {})
        except Exception:
            pass
    return {}


class Settings(BaseSettings):
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # CORS - 支持环境变量配置，默认为允许所有
    CORS_ORIGINS: List[str] = ["*"]
    
    # 从环境变量读取，覆盖默认值
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cors_env = os.environ.get("CORS_ORIGINS", "")
        if cors_env:
            self.CORS_ORIGINS = [origin.strip() for origin in cors_env.split(",")]
        
        # Load agent config from settings.json (single source of truth)
        agent_cfg = _load_agent_from_settings()
        self.AGENT_PROVIDER = agent_cfg.get("provider", "openai")
        self.AGENT_MODEL = agent_cfg.get("model", "gpt-4")
    
    # Agent (defaults overridden by settings.json at init)
    AGENT_PROVIDER: str = "openai"
    AGENT_MODEL: str = "gpt-4"
    
    # Wiki
    WIKI_DATA_DIR: str = ".quant_agent"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

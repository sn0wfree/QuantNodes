# coding=utf-8
"""
api/config.py — **API 服务器配置** (Tier 2: API Server)

本模块提供 FastAPI 服务器专用的配置，**不是**库级配置的重复。

配置分层 (Configuration Tiers):
    ┌─────────────────────────────────────────────────────────┐
    │ Tier 3: YAML 策略配置                                      │
    │   → QuantNodes.agent.config (ConfigLoader/ConfigExecutor) │
    ├─────────────────────────────────────────────────────────┤
    │ Tier 2: API 服务器配置 ← 本文件                            │
    │   → api.config.Settings (CORS, VERSION, AGENT_PROVIDER)  │
    ├─────────────────────────────────────────────────────────┤
    │ Tier 1: 库级配置                                           │
    │   → core.config.Settings (MySQL/CH/DuckDB/LLM/缓存/回测)  │
    └─────────────────────────────────────────────────────────┘

字段说明:
    - VERSION: API 版本号 (用于 FastAPI 元数据 + 启动横幅)
    - DEBUG: 调试模式开关 (影响 api/deps.py 鉴权策略)
    - CORS_ORIGINS: 跨域白名单 (环境变量 CORS_ORIGINS 覆盖)
    - AGENT_PROVIDER / AGENT_MODEL: Agent LLM 提供商/模型
      (从 .agent/settings.json 加载, 与 core.config.llm 互补)
    - WIKI_DATA_DIR: Wiki 数据目录

与 core.config 的关系:
    - 本文件**不继承** core.config.Settings — 避免 API 服务依赖库全部字段
    - 库配置 (MySQL/CH/DuckDB/LLM) 走 core.config.Settings
    - 同步机制: api.services.settings_service.sync_core_config()
      把 .agent/settings.json 的 agent.api_key 等同步到 core 单例

环境变量:
    - DEBUG (默认 False, 不带 QUANTNODES_ 前缀)
    - CORS_ORIGINS (逗号分隔列表)
    - 字段全大写, 无 env_prefix
    - 从 .env + .agent/settings.json 双源加载
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
import json
from pathlib import Path

from QuantNodes.constants import DEFAULT_LLM_MODEL


def _load_agent_from_settings() -> dict:
    """Load agent config from .agent/settings.json (single source of truth)"""
    settings_file = Path(".agent/settings.json")
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
        self.AGENT_MODEL = agent_cfg.get("model", DEFAULT_LLM_MODEL)
    
    # Agent (defaults overridden by settings.json at init)
    AGENT_PROVIDER: str = "openai"
    AGENT_MODEL: str = DEFAULT_LLM_MODEL
    
    # Wiki
    WIKI_DATA_DIR: str = ".agent"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # QUANTNODES__LLM__* and NANOBOT_* are consumed by config_mapper.py via os.environ, not here


settings = Settings()

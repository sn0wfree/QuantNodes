# coding=utf-8

import json
from typing import Optional, Dict, Any
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "quant"
    charset: str = "utf8mb4"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


class ClickHouseConfig(BaseSettings):
    """ClickHouse 配置"""
    host: str = "localhost"
    port: int = 8123
    user: str = "default"
    password: str = ""
    database: str = "default"
    compression: bool = True
    secure: bool = False


class DuckDBConfig(BaseSettings):
    """DuckDB 配置"""
    path: str = ":memory:"
    read_only: bool = False


class LLMConfig(BaseSettings):
    """LLM 配置"""
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    timeout: int = 60
    max_retries: int = 3
    max_tokens: int = 102400
    providers: Dict[str, Any] = {}


class Settings(BaseSettings):
    """全局配置"""

    # 项目配置
    project_name: str = "QuantNodes"
    debug: bool = True
    log_level: str = "INFO"

    # 数据库
    mysql: DatabaseConfig = DatabaseConfig()
    clickhouse: ClickHouseConfig = ClickHouseConfig()
    duckdb: DuckDBConfig = DuckDBConfig()

    # LLM
    llm: LLMConfig = LLMConfig()

    # 回测默认参数
    default_commission: float = 0.001  # 千分之一手续费
    default_slippage: float = 0.001   # 千分之一滑点

    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1小时

    model_config = {
        "env_prefix": "QUANTNODES_",
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }

    def to_dict(self) -> Dict[str, Any]:
        """转为字典，隐藏敏感信息"""
        data = self.model_dump()

        # 隐藏密码
        if 'mysql' in data and 'password' in data['mysql']:
            data['mysql']['password'] = '***'
        if 'clickhouse' in data and 'password' in data['clickhouse']:
            data['clickhouse']['password'] = '***'
        if 'llm' in data and 'api_key' in data['llm']:
            data['llm']['api_key'] = '***' if data['llm']['api_key'] else None

        return data

    def load_from_settings(self) -> None:
        """Load LLM and backtest settings from .quant_agent/settings.json
        (single source of truth)
        """
        settings_file = Path(".quant_agent/settings.json")
        if not settings_file.exists():
            return
        try:
            data = json.loads(settings_file.read_text(encoding="utf-8"))
            agent = data.get("agent", {})
            if agent.get("api_key"):
                self.llm.api_key = agent["api_key"]
            if agent.get("api_base"):
                self.llm.base_url = agent["api_base"]
            if agent.get("model"):
                self.llm.model = agent["model"]
            if "llm_timeout" in agent:
                self.llm.timeout = agent["llm_timeout"]
            if "llm_max_retries" in agent:
                self.llm.max_retries = agent["llm_max_retries"]
            if "max_tokens" in agent:
                self.llm.max_tokens = agent["max_tokens"]
            if "providers" in agent:
                self.llm.providers = agent["providers"]
            backtest = data.get("backtest", {})
            if "default_commission" in backtest:
                self.default_commission = backtest["default_commission"]
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置（单例）"""
    s = Settings()
    s.load_from_settings()
    return s


settings = get_settings()

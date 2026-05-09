# coding=utf-8

from typing import Optional, Dict, Any
from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置（单例）"""
    return Settings()


settings = get_settings()

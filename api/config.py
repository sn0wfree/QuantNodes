from pydantic_settings import BaseSettings
from typing import List
import os


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
    
    # Agent
    AGENT_PROVIDER: str = "openai"
    AGENT_MODEL: str = "gpt-4"
    
    # Wiki
    WIKI_DATA_DIR: str = ".quant_agent"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

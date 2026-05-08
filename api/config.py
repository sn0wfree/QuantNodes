from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Agent
    AGENT_PROVIDER: str = "openai"
    AGENT_MODEL: str = "gpt-4"
    
    # Wiki
    WIKI_DATA_DIR: str = ".quant_agent"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

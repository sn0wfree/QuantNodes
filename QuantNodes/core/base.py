# coding=utf-8

from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, ConfigDict


class BaseModel(PydanticBaseModel):
    """基础模型类"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=False,
    )


class QuantNodesBase:
    """所有业务类的基类"""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class QuantNodesError(Exception):
    """基础异常类"""
    code = "QUANTNODES_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigError(QuantNodesError):
    """配置异常"""
    code = "CONFIG_ERROR"


class DatabaseError(QuantNodesError):
    """数据库异常"""
    code = "DATABASE_ERROR"


class FactorError(QuantNodesError):
    """因子异常"""
    code = "FACTOR_ERROR"


class BacktestError(QuantNodesError):
    """回测异常"""
    code = "BACKTEST_ERROR"


class ValidationError(QuantNodesError):
    """校验异常"""
    code = "VALIDATION_ERROR"

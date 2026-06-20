# coding=utf-8
"""
QuantNodes Agent Config - 配置文件驱动模块

基于 YAML 的策略配置加载和执行。

Modules:
    types: 类型定义
    loader: 配置加载器
    executor: 配置执行器

Usage:
    from QuantNodes.agent.config import ConfigLoader, ConfigExecutor

    loader = ConfigLoader()
    config = loader.load("strategy.yaml")

    executor = ConfigExecutor()
    result = executor.run(config, data)
"""

from .loader import ConfigLoader, load_config
from .executor import ConfigExecutor
from .types import StrategyConfig, FactorConfig, OperationConfig, BacktestConfig

__all__ = [
    "ConfigLoader",
    "ConfigExecutor",
    "load_config",
    "StrategyConfig",
    "FactorConfig",
    "OperationConfig",
    "BacktestConfig",
]

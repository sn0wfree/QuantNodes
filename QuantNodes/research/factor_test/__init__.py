# coding: utf-8
"""
单因子回测节点化模块 / Single-Factor Backtest Node Module

将单因子回测的 12 项能力拆分为独立 Node，通过 Pipeline 组合。

用法:
    from QuantNodes.research.factor_test import SingleFactorTestConfig, PipelineRunner

    config = SingleFactorTestConfig(...)
    runner = PipelineRunner(config)
    result = runner.run()
"""

from .config import (
    SingleFactorTestConfig, FactorSetting, PreprocessSetting,
    TradableSetting, AnalysisSetting, OutputSetting,
)
from .utils import DataLoader

__all__ = [
    'SingleFactorTestConfig', 'FactorSetting', 'PreprocessSetting',
    'TradableSetting', 'AnalysisSetting', 'OutputSetting',
    'DataLoader',
]

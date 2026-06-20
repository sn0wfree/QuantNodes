# coding=utf-8
"""
Factor Skills
"""

from .ic_analysis import ICAnalysisSkill
from .group_backtest import GroupBacktestSkill
from .correlation import CorrelationSkill

__all__ = [
    "ICAnalysisSkill",
    "GroupBacktestSkill",
    "CorrelationSkill",
]

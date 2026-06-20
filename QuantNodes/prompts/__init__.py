# coding=utf-8
"""
QuantNodes Prompts Package

Complete prompts for external agents with reference code.
"""

from .strategy.momentum import MOMENTUM_PROMPT
from .strategy.mean_reversion import MEAN_REVERSION_PROMPT
from .strategy.trend_following import TREND_FOLLOWING_PROMPT
from .strategy.pairs_trading import PAIRS_TRADING_PROMPT
from .strategy.market_neutral import MARKET_NEUTRAL_PROMPT

__all__ = [
    "MOMENTUM_PROMPT",
    "MEAN_REVERSION_PROMPT",
    "TREND_FOLLOWING_PROMPT",
    "PAIRS_TRADING_PROMPT",
    "MARKET_NEUTRAL_PROMPT",
]

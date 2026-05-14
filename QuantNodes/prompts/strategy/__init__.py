# coding=utf-8
"""
Strategy Prompts
"""

from .momentum import MOMENTUM_PROMPT
from .mean_reversion import MEAN_REVERSION_PROMPT
from .trend_following import TREND_FOLLOWING_PROMPT
from .pairs_trading import PAIRS_TRADING_PROMPT
from .market_neutral import MARKET_NEUTRAL_PROMPT

__all__ = [
    "MOMENTUM_PROMPT",
    "MEAN_REVERSION_PROMPT",
    "TREND_FOLLOWING_PROMPT",
    "PAIRS_TRADING_PROMPT",
    "MARKET_NEUTRAL_PROMPT",
]
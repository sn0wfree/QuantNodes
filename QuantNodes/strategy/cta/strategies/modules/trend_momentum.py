# coding=utf-8
"""趋势动量组 — 价格趋势/动量/突破类因子.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.trend import (
    ATRBreakoutFactor, EfficiencyFactor, MAFactor, MomentumFactor, RSIFactor,
)
from ...factors.trend_advanced import (
    AroonFactor, ChandelierExitFactor, LinearSlopeFactor, TRIXFactor, VortexFactor,
)
from ...factors.oscillators import MACDFactor
from ...factors.composite import (
    MomentumCrashFilterFactor, MultiHorizonMomentumFactor, SharpeMomentumFactor,
)
from ...factors.trend_quality import MomentumSlopeFactor, TrendQualityFactor
from ...factors.market_structure import NewHighLowFactor
from ...factors.momentum_xs import MomentumXSFactor
from ...factors.turn_point import FastTurnPointFactor, TurnPointBreakoutFactor
from ...factors.short_term_reversal import ShortTermReversalTrendFactor
from ...factors.microstructure import (
    IntradayReturnFactor, OvernightIntradayDivergenceFactor, OvernightReturnFactor,
)
from ...factors.candlestick import LongBodyFactor


class TrendMomentumModuleStrategy(_ModuleStrategy):
    """趋势动量组: 价格趋势/动量/突破类因子.

    去重 (docs/94 §10.4): donchian×ma 相关 0.798 (72 品种离散信号最高对)
    且 donchian 单因子 Sharpe 最弱 (-0.52), 删除.
    """

    FACTORS = [
        ("ma", MAFactor, "panel"),
        ("momentum", MomentumFactor, "panel"),
        ("rsi", RSIFactor, "panel"),
        ("efficiency", EfficiencyFactor, "panel"),
        ("aroon", AroonFactor, "panel"),
        ("linear_slope", LinearSlopeFactor, "panel"),
        ("trix", TRIXFactor, "panel"),
        ("macd", MACDFactor, "panel"),
        ("multi_horizon_mom", MultiHorizonMomentumFactor, "panel"),
        ("sharpe_mom", SharpeMomentumFactor, "panel"),
        ("mom_crash_filter", MomentumCrashFilterFactor, "panel"),
        ("mom_slope", MomentumSlopeFactor, "panel"),
        ("trend_quality", TrendQualityFactor, "panel"),
        ("new_high_low", NewHighLowFactor, "panel"),
        ("momentum_xs", MomentumXSFactor, "panel"),
        ("atr_breakout", ATRBreakoutFactor, "ohlc"),
        ("turn_point_breakout", TurnPointBreakoutFactor, "ohlc"),
        ("fast_turn_point", FastTurnPointFactor, "ohlc"),
        ("short_term_reversal_trend", ShortTermReversalTrendFactor, "panel"),
        ("overnight_ret", OvernightReturnFactor, "ohlc"),
        ("intraday_ret", IntradayReturnFactor, "ohlc"),
        ("overnight_intraday_div", OvernightIntradayDivergenceFactor, "ohlc"),
        ("long_body", LongBodyFactor, "ohlc"),
        ("vortex", VortexFactor, "ohlc"),
        ("chandelier", ChandelierExitFactor, "ohlc"),
    ]

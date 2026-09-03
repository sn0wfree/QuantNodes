# coding=utf-8
"""振荡器Alpha组 — 15因子振荡器+inv_vol加权+双向交易.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.trend import BIASFactor, BollingerFactor
from ...factors.mean_reversion import HalfLifeFactor, SqueezeFactor
from ...factors.price_location import OpenLocationFactor, PriceLocationFactor
from ...factors.cointegration import SpreadMeanReversionFactor
from ...factors.oscillators import CCIFactor, StochasticFactor, WilliamsRFactor
from ...factors.microstructure import CloseLocationFactor, GapFactor
from ...factors.candlestick import DojiFactor, EngulfingFactor, HammerFactor


class OscillatorAlphaModuleStrategy(_ModuleStrategy):
    """振荡器Alpha组: 15因子振荡器+inv_vol加权+双向交易.

    核心逻辑:
        1. 15个振荡器/反转因子独立计算信号
        2. inv_vol加权: 波动率低的因子权重更高
        3. 双向交易: 55%做多, 44%做空
        4. 始终在场: 仅1%时间空仓
        5. vol_targeting: 仓位按波动率调整

    收益来源:
        - 做多贡献: +11.0% 年化 (主要来源)
        - 做空贡献: +6.7% 年化 (次要来源)
        - 始终在场: 避免空仓机会成本

    适用场景:
        - 趋势市: 更赚钱 (33%年化)
        - 震荡市: 表现一般 (2%年化)
    """
    # 保留旧名称作为别名
    MeanReversionModuleStrategy = None  # 向后兼容

    FACTORS = [
        ("bollinger", BollingerFactor, "panel"),
        ("bias", BIASFactor, "panel"),
        ("half_life", HalfLifeFactor, "panel"),
        ("price_location", PriceLocationFactor, "panel"),
        ("spread_mr", SpreadMeanReversionFactor, "series"),
        ("stochastic", StochasticFactor, "ohlc"),
        ("cci", CCIFactor, "ohlc"),
        ("williams_r", WilliamsRFactor, "ohlc"),
        ("squeeze", SqueezeFactor, "ohlc"),
        ("close_location", CloseLocationFactor, "ohlc"),
        ("open_location", OpenLocationFactor, "ohlc"),
        ("doji", DojiFactor, "ohlc"),
        ("hammer", HammerFactor, "ohlc"),
        ("engulfing", EngulfingFactor, "ohlc"),
        ("gap", GapFactor, "ohlc"),
    ]


# 向后兼容
MeanReversionModuleStrategy = OscillatorAlphaModuleStrategy

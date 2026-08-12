# coding=utf-8
"""K线形态因子.

基于 OHLC 的经典蜡烛图形态:
    - LongBodyFactor: 长实体 (大阳/大阴)
    - DojiFactor: 十字星 (犹豫)
    - HammerFactor: 锤子线/上吊线 (反转)
    - EngulfingFactor: 吞没形态 (反转)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def _body(ohlc: pd.DataFrame) -> pd.Series:
    """实体大小 (close - open)."""
    return ohlc["close"] - ohlc["open"]


def _body_pct(ohlc: pd.DataFrame) -> pd.Series:
    """实体相对开盘价比例."""
    return _body(ohlc) / ohlc["open"].replace(0, np.nan)


def _range(ohlc: pd.DataFrame) -> pd.Series:
    """当日振幅 (high - low)."""
    return ohlc["high"] - ohlc["low"]


class LongBodyFactor(Factor):
    """长实体因子.

    大阳线 (实体 > N×振幅) → 强多头; 大阴线 → 强空头.
    研报: 经典 K 线形态策略.

    参数:
        window: 实体/振幅比阈值窗口 (用于自适应)
        mult: 实体占振幅倍数
    """

    name = "long_body"

    def __init__(self, window: int = 20, mult: float = 0.6):
        self.window = window
        self.mult = mult

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"open", "high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"LongBodyFactor requires columns {req}")
        body_ratio = _body(ohlc).abs() / _range(ohlc).replace(0, np.nan)
        # 自适应阈值: 窗口内平均实体占比
        threshold = body_ratio.rolling(self.window).mean() * self.mult
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["long_body"])
        sig.loc[(body_ratio > threshold) & (_body(ohlc) > 0), "long_body"] = 1.0
        sig.loc[(body_ratio > threshold) & (_body(ohlc) < 0), "long_body"] = -1.0
        return sig


class DojiFactor(Factor):
    """十字星因子.

    实体极小 (接近 0) → 多空犹豫 → 潜在反转.
    研报: 经典 K 线形态.

    参数:
        body_threshold: 实体/振幅比阈值
    """

    name = "doji"

    def __init__(self, body_threshold: float = 0.1):
        self.body_threshold = body_threshold

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"open", "high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"DojiFactor requires columns {req}")
        body_ratio = _body(ohlc).abs() / _range(ohlc).replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["doji"])
        sig.loc[body_ratio < self.body_threshold, "doji"] = 1.0
        return sig


class HammerFactor(Factor):
    """锤子线因子.

    下影线长 + 实体小 + 位于低位 → 看涨锤子线.
    上影线长 + 实体小 + 位于高位 → 看跌上吊线.
    研报: 经典反转形态.

    参数:
        shadow_mult: 影线/实体倍数
    """

    name = "hammer"

    def __init__(self, shadow_mult: float = 2.0):
        self.shadow_mult = shadow_mult

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"open", "high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"HammerFactor requires columns {req}")
        upper_shadow = ohlc["high"] - ohlc[["open", "close"]].max(axis=1)
        lower_shadow = ohlc[["open", "close"]].min(axis=1) - ohlc["low"]
        body = _body(ohlc).abs().replace(0, np.nan)

        # 锤子: 下影 > 2×实体, 上影小
        hammer = (lower_shadow > self.shadow_mult * body) & \
            (upper_shadow < body)
        # 上吊: 上影 > 2×实体
        hanging = (upper_shadow > self.shadow_mult * body) & \
            (lower_shadow < body)

        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["hammer"])
        sig.loc[hammer, "hammer"] = 1.0
        sig.loc[hanging, "hammer"] = -1.0
        return sig


class EngulfingFactor(Factor):
    """吞没形态因子.

    今日实体完全吞没昨日实体 → 强反转信号.
    看涨吞没: 昨阴今阳; 看跌吞没: 昨阳今阴.
    研报: 经典反转形态.

    参数:
        body_min: 实体最小占比 (过滤十字)
    """

    name = "engulfing"

    def __init__(self, body_min: float = 0.3):
        self.body_min = body_min

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"open", "high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"EngulfingFactor requires columns {req}")
        body = _body(ohlc)
        prev_body = body.shift(1)
        body_ratio = body.abs() / _range(ohlc).replace(0, np.nan)

        # 看涨吞没: 昨阴 (prev_body<0) + 今阳 (body>0) + 今实体吞昨实体
        bull = (prev_body < 0) & (body > 0) & \
            (ohlc["close"] >= ohlc["open"].shift(1)) & \
            (ohlc["open"] <= ohlc["close"].shift(1))
        # 看跌吞没: 昨阳 + 今阴
        bear = (prev_body > 0) & (body < 0) & \
            (ohlc["open"] >= ohlc["close"].shift(1)) & \
            (ohlc["close"] <= ohlc["open"].shift(1))
        # 实体足够大
        bull &= body_ratio > self.body_min
        bear &= body_ratio > self.body_min

        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["engulfing"])
        sig.loc[bull, "engulfing"] = 1.0
        sig.loc[bear, "engulfing"] = -1.0
        return sig

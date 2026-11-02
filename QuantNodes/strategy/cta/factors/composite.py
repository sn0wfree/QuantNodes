# coding=utf-8
"""复合增强因子.

研报主流复合方法, 基于已有因子/价格:
    - MultiHorizonMomentumFactor: 多周期动量聚合 (3/6/12月加权)
    - SharpeMomentumFactor: 波动率调整动量 (夏普动量)
    - MomentumCrashFilterFactor: 长短期动量差 (动量崩溃规避)
    - MultiPeriodResonanceFactor: 多周期 EMA 共振 (5/20/60 日多数决, 加权可选)
    - VolatilityRegimeResonanceFactor: 波动率体制感知 (仅高波动期启用)
    - CrossDimensionalResonanceFactor: 跨维度共振 (EMA + 动量 + 波动率压缩)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class MultiHorizonMomentumFactor(Factor):
    """多周期动量聚合因子.

    3/6/12月动量加权平均 (近端权重高).
    研报: AQR 多周期动量, 华泰金工.

    参数:
        horizons: 各周期天数 (默认 [63, 126, 252])
        weights: 权重 (默认 [0.5, 0.3, 0.2])
    """

    name = "multi_horizon_mom"

    def __init__(self, horizons: list[int] | None = None, weights: list[float] | None = None):
        self.horizons = horizons or [63, 126, 252]
        self.weights = weights or [0.5, 0.3, 0.2]
        if len(self.horizons) != len(self.weights):
            raise ValueError("horizons and weights must have equal length")

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        rets = {}
        for h, w in zip(self.horizons, self.weights):
            rets[h] = prices.pct_change(h, fill_method=None) * w
        combined = sum(rets.values())
        # 归一化: 除以权重和
        combined = combined / sum(self.weights)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[combined > 0] = 1.0
        sig[combined < 0] = -1.0
        return sig


class SharpeMomentumFactor(Factor):
    """夏普动量因子 (波动率调整动量).

    momentum / vol: 单位风险动量, 高 = 高质量趋势.
    研报: 招商证券 CTA 因子体系.

    参数:
        lookback: 动量窗口
        vol_window: 波动窗口
        threshold: 信号阈值
    """

    name = "sharpe_mom"

    def __init__(self, lookback: int = 63, vol_window: int = 20, threshold: float = 0.5):
        self.lookback = lookback
        self.vol_window = vol_window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        mom = prices.pct_change(self.lookback, fill_method=None)
        vol = ret.rolling(self.vol_window).std() * np.sqrt(252)
        sharpe_mom = mom / vol.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[sharpe_mom > self.threshold] = 1.0
        sig[sharpe_mom < -self.threshold] = -1.0
        return sig


class MomentumCrashFilterFactor(Factor):
    """动量崩溃过滤因子.

    短期动量转负且长期仍正 → 趋势崩溃 → 空头/规避多头.
    研报: 华泰金工动量崩溃研究.

    参数:
        short_lookback: 短期动量窗口
        long_lookback: 长期动量窗口
    """

    name = "momentum_crash"

    def __init__(self, short_lookback: int = 20, long_lookback: int = 252):
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        short_mom = prices.pct_change(self.short_lookback, fill_method=None)
        long_mom = prices.pct_change(self.long_lookback, fill_method=None)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # 短期转负 & 长期仍正 → 崩溃
        sig[(short_mom < 0) & (long_mom > 0)] = -1.0
        return sig


class MultiPeriodResonanceFactor(Factor):
    """多周期 EMA 共振因子.

    对每个 bar 在 N 个 EMA 周期上独立判定方向 (close > EMA → +1, close < EMA → -1),
    累计投票 ≥ min_agree → 输出 +1; ≤ -min_agree → 输出 -1; 否则 0.

    这是文章 §"开仓信号: 第三层 - 复合信号层" 多周期共振的工程实现:
        同一指标 (EMA) 在多个滚动窗口下计算结果 → 信号少数服从多数.
    文章引用 AQR 风格 "多组参数双均线系统" 与华泰前序报告
    《红利因子择时与2025Q1行业ETF投资建议》(2025-01-07) 同款思路.

    用于:
        - TREND_UP/DOWN: 作为现有自适应均线交叉的"二次确认"过滤
        - 单独使用: 作为复合信号层的中等频率择时指标

    参数:
        windows: EMA 周期列表 (默认 (5, 20, 60), 文章典型日频组合)
        weights: 各周期投票权重 (默认 None 等权). 提供时启用加权多数决:
                 累计加权和 ≥ threshold → +1; ≤ -threshold → -1.
        threshold: 加权多数决阈值 (默认 0.5, 即加权和超过半数权重和)
        min_agree: (兼容旧版) 至少多少个周期一致才输出 (默认 = len(windows)//2 + 1)
    """

    name = "multi_period_resonance"

    def __init__(
        self,
        windows: tuple[int, ...] = (5, 20, 60),
        weights: tuple[float, ...] | None = None,
        threshold: float = 0.5,
        min_agree: int | None = None,
    ):
        self.windows = tuple(windows)
        if weights is None:
            self.weights = None
        else:
            if len(weights) != len(windows):
                raise ValueError("weights must match windows length")
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            self.weights = tuple(weights)
        self.threshold = threshold
        if min_agree is None:
            min_agree = len(self.windows) // 2 + 1
        if min_agree < 1 or min_agree > len(self.windows):
            raise ValueError(
                f"min_agree must be in [1, {len(self.windows)}], got {min_agree}",
            )
        self.min_agree = min_agree

    def raw_value(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """连续值: 加权 (或等权) 投票数, 范围 [-sum_weights, +sum_weights]."""
        votes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for i, w in enumerate(self.windows):
            ema = prices.ewm(span=w, adjust=False).mean()
            sign = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            sign[prices > ema] = 1.0
            sign[prices < ema] = -1.0
            weight = self.weights[i] if self.weights is not None else 1.0
            votes += weight * sign
        return votes

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """离散 {-1, 0, +1}: 多数决结果 (加权或等权)."""
        votes = self.raw_value(prices)
        if self.weights is None:
            # 等权多数决 (旧行为)
            threshold = self.min_agree
            neg_threshold = -self.min_agree
        else:
            # 加权多数决: 超过半数权重和
            total_weight = sum(self.weights)
            threshold = self.threshold * total_weight
            neg_threshold = -self.threshold * total_weight
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[votes >= threshold] = 1.0
        sig[votes <= neg_threshold] = -1.0
        return sig


class VolatilityRegimeResonanceFactor(Factor):
    """波动率体制感知的多周期共振.

    仅在**波动率高于历史百分位阈值**时启用共振过滤.
    低波动率环境下 (震荡市) 共振频繁反向, 直接禁用以避免过度过滤.

    设计动机:
        实测 multi_period 变体 (5/20/60 等权多数决) 在 2020-2024 区间反而恶化业绩
        (Calmar +0.022 → -0.042). 根因之一: 低波动率震荡市中多周期方向频繁分歧,
        共振过滤大量本应入场的中信号.
    改进: 仅在高波动率环境下启用共振, 让低波动率环境保留单窗口信号.

    参数:
        windows: EMA 周期列表
        min_agree: 等权多数决阈值
        vol_window: 波动率估计窗口 (默认 20)
        percentile: 启用阈值百分位 (默认 0.75, 即波动率 > 历史 75 分位才启用)
        weights: 可选加权
    """

    name = "vol_regime_resonance"

    def __init__(
        self,
        windows: tuple[int, ...] = (5, 20, 60),
        min_agree: int = 2,
        vol_window: int = 20,
        percentile: float = 0.75,
        weights: tuple[float, ...] | None = None,
    ):
        self.windows = tuple(windows)
        self.min_agree = min_agree
        self.vol_window = vol_window
        self.percentile = percentile
        self.weights = weights
        # 复用 MultiPeriodResonanceFactor
        self._inner = MultiPeriodResonanceFactor(
            windows=windows, min_agree=min_agree, weights=weights,
        )

    def _vol_regime(self, prices: pd.DataFrame) -> pd.DataFrame:
        """逐时点判断当前波动率是否高于历史百分位."""
        ret = prices.pct_change(fill_method=None)
        vol = ret.rolling(self.vol_window).std()
        # 滚动百分位: 在过去 N 日的 vol 分布中, 当前 vol 排名
        # 使用 expanding 百分位避免 look-ahead
        threshold = vol.expanding(min_periods=self.vol_window * 4).quantile(self.percentile)
        return (vol >= threshold).fillna(False)

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raw = self._inner.compute(prices)
        high_vol = self._vol_regime(prices)
        # 低波动率环境 → 输出 0 (相当于关闭共振)
        return raw.where(high_vol, 0.0)


class CrossDimensionalResonanceFactor(Factor):
    """跨维度共振因子 (EMA 方向 + 动量符号 + 波动率压缩三维度投票).

    每个维度独立判定方向 (-1/0/+1), 三维投票多数决:
        - 维度 1: 多周期 EMA 方向 (近期 5/20 日投票)
        - 维度 2: 多窗口动量符号 (5/20 日收益率符号投票)
        - 维度 3: 波动率压缩方向 (近 5 日 vol < 远 60 日 vol → +1; 反之 -1)

    设计动机:
        文章 §"开仓信号: 第三层 - 复合信号层" 提到"多指标共振"也可作为复合层.
        跨维度比多周期能提供更丰富的信号特征, 缓解多周期共振的"快速反转早期
        被过滤"问题 (动量与波动率压缩对快速反转有反应, 而多周期 EMA 不响应).

    参数:
        ema_windows: EMA 维度周期 (默认 (5, 20))
        mom_windows: 动量维度窗口 (默认 (5, 20))
        vol_short / vol_long: 波动率压缩维度短期 / 长期窗口 (默认 5 / 60)
        min_agree_dim: 至少多少个维度一致 (默认 2 of 3)
    """

    name = "cross_dim_resonance"

    def __init__(
        self,
        ema_windows: tuple[int, ...] = (5, 20),
        mom_windows: tuple[int, ...] = (5, 20),
        vol_short: int = 5,
        vol_long: int = 60,
        min_agree_dim: int = 2,
    ):
        self.ema_windows = tuple(ema_windows)
        self.mom_windows = tuple(mom_windows)
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.min_agree_dim = min_agree_dim

    def _vote(self, prices: pd.DataFrame, indicator_fn) -> pd.DataFrame:
        """通用投票: 每个 indicator_fn(prices, window) 返回 {-1, 0, +1} DataFrame,
        多窗口投票得 {-3..+3} 区间."""
        votes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for w in (self.ema_windows if indicator_fn.__name__ == "_ema_signal" else self.mom_windows):
            sig = indicator_fn(prices, w)
            votes += sig
        return votes

    @staticmethod
    def _ema_signal(prices: pd.DataFrame, window: int) -> pd.DataFrame:
        ema = prices.ewm(span=window, adjust=False).mean()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[prices > ema] = 1.0
        sig[prices < ema] = -1.0
        return sig

    @staticmethod
    def _mom_signal(prices: pd.DataFrame, window: int) -> pd.DataFrame:
        ret = prices.pct_change(window, fill_method=None)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[ret > 0] = 1.0
        sig[ret < 0] = -1.0
        return sig

    def _vol_compress_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """波动率压缩: 短期 vol < 长期 vol → +1 (压缩中, 突破前兆); 反之 -1."""
        ret = prices.pct_change(fill_method=None)
        short_vol = ret.rolling(self.vol_short).std()
        long_vol = ret.rolling(self.vol_long).std()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[short_vol < long_vol] = 1.0
        sig[short_vol > long_vol] = -1.0
        return sig

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ema_votes = self._vote(prices, self._ema_signal)
        mom_votes = self._vote(prices, self._mom_signal)
        vol_signal = self._vol_compress_signal(prices)
        # 三维度各归一化到 {-1, 0, +1} (按各自多数决)
        ema_dir = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ema_dir[ema_votes > 0] = 1.0
        ema_dir[ema_votes < 0] = -1.0
        mom_dir = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        mom_dir[mom_votes > 0] = 1.0
        mom_dir[mom_votes < 0] = -1.0
        # 三维度投票
        total = ema_dir + mom_dir + vol_signal
        # 至少 min_agree_dim 个维度同向 → 输出信号
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[total >= self.min_agree_dim] = 1.0
        sig[total <= -self.min_agree_dim] = -1.0
        return sig

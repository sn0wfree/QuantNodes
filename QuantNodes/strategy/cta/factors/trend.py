# coding=utf-8
"""时序动量因子.

经典 CTA 趋势因子, 基于主力连续面板:
    - DonchianChannelFactor: 唐奇安通道突破 (海龟)
    - MAFactor: 双均线 (短/长)
    - MomentumFactor: N 期收益率
    - TSMAFactor: 时间序列动量 (AQR, N 期收益归一化)
    - RSIFactor: 相对强弱 (超买超卖)
    - BollingerFactor: 布林带突破
    - EfficiencyFactor: 考夫曼效率比趋势
    - ADXFilterFactor: ADX 趋势强度过滤
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor
from .indicators import adx, atr, bollinger_percent_b, efficiency_ratio, rsi


class DonchianChannelFactor(Factor):
    """唐奇安通道突破因子.

    通道上轨 = 过去 n 日最高, 下轨 = 过去 n 日最低.
    信号: 价格 > 上轨 → 多头; 价格 < 下轨 → 空头.

    参数:
        entry_window: 入场通道周期 (默认 20)
        exit_window:  出场通道周期 (默认 10)
    """

    name = "donchian"

    def __init__(self, entry_window: int = 20, exit_window: int = 10):
        self.entry_window = entry_window
        self.exit_window = exit_window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """返回每个品种 -1/0/+1 趋势状态 (向量化).

        Returns:
            pd.DataFrame (T, N), 每列: +1 价格>上轨, -1 价格<下轨, 0 中间.
        """
        # 入场: 突破过去 entry_window 日的最高/最低 (不含当日)
        up = prices.rolling(self.entry_window).max().shift(1)
        dn = prices.rolling(self.entry_window).min().shift(1)
        # 出场: 回落/回升至 exit_window 日内 (不含当日, 当日价必然 ≥ 当日最低)
        exit_dn = prices.rolling(self.exit_window).min().shift(1)
        exit_up = prices.rolling(self.exit_window).max().shift(1)

        # 状态机向量化: 逐日推进 (DataFrame 逐行, 但无 pandas 内建递归)
        state = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # copy=True: 新版 pandas 的 to_numpy() 可能返回零拷贝只读视图,
        # 原地写入 state_arr[i] 会抛 "assignment destination is read-only"
        state_arr = state.to_numpy(copy=True)
        p_arr = prices.to_numpy()
        up_arr = up.to_numpy()
        dn_arr = dn.to_numpy()
        e_dn = exit_dn.to_numpy()
        e_up = exit_up.to_numpy()

        n = len(prices)
        for i in range(1, n):
            prev = state_arr[i - 1]
            cur = prev.copy()
            # 多头: 空仓且突破上轨 → 多; 多头且跌破 exit_dn → 平
            long_enter = (prev == 0) & (p_arr[i] > up_arr[i])
            long_exit = (prev == 1) & (p_arr[i] < e_dn[i])
            short_enter = (prev == 0) & (p_arr[i] < dn_arr[i])
            short_exit = (prev == -1) & (p_arr[i] > e_up[i])
            cur[long_enter] = 1.0
            cur[long_exit] = 0.0
            cur[short_enter] = -1.0
            cur[short_exit] = 0.0
            state_arr[i] = cur

        return pd.DataFrame(state_arr, index=prices.index, columns=prices.columns)


class MAFactor(Factor):
    """双均线交叉因子.

    signal = +1 短均线>长均线, -1 短均线<长均线.
    """

    name = "ma"

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        short = prices.rolling(self.short_window).mean()
        long = prices.rolling(self.long_window).mean()
        sig = (short > long).astype(float)
        sig[short < long] = -1.0
        return sig


class MomentumFactor(Factor):
    """N 期收益率动量.

    参数:
        lookback: 回看窗口 (默认 20)
        threshold: 触发阈值 (默认 0, 收益>0 则多)
    """

    name = "momentum"

    def __init__(self, lookback: int = 20, threshold: float = 0.0):
        self.lookback = lookback
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(self.lookback, fill_method=None)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[ret > self.threshold] = 1.0
        sig[ret < -self.threshold] = -1.0
        return sig


class TSMAFactor(Factor):
    """时间序列动量 (AQR 风格).

    信号 = sign(N 期收益), 幅度可作仓位强度.
    参数:
        lookback: 回看窗口 (默认 12 月 ≈ 250 日)
    """

    name = "tsmom"

    def __init__(self, lookback: int = 250):
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sign(prices.pct_change(self.lookback, fill_method=None))


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """平均真实波幅 (ATR). 输入含 high/low/close 面板."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


class RSIFactor(Factor):
    """RSI 超买超卖因子.

    signal = +1 RSI<下界 (超卖反弹), -1 RSI>上界 (超买回落).
    参数:
        window: RSI 窗口 (默认 14)
        low_threshold: 超卖线 (默认 30)
        high_threshold: 超买线 (默认 70)
    """

    name = "rsi"

    def __init__(self, window: int = 14, low_threshold: float = 30, high_threshold: float = 70):
        self.window = window
        self.low = low_threshold
        self.high = high_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        r = prices.apply(lambda s: rsi(s, self.window))
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[r < self.low] = 1.0
        sig[r > self.high] = -1.0
        return sig


class BollingerFactor(Factor):
    """布林带突破因子.

    %B > 1 → 突破上轨 (强多头), %B < 0 → 跌破下轨 (强空头).
    signal = +1 %B>1, -1 %B<0, 否则 0.
    """

    name = "bollinger"

    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = window
        self.n_std = n_std

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pb = prices.apply(lambda s: bollinger_percent_b(s, self.window, self.n_std))
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[pb > 1.0] = 1.0
        sig[pb < 0.0] = -1.0
        return sig


class EfficiencyFactor(Factor):
    """考夫曼效率比因子.

    signal = +1 效率比>上阈 (强趋势), -1 效率比<下阈 (弱趋势/反转).
    """

    name = "efficiency"

    def __init__(self, window: int = 10, high_threshold: float = 0.5, low_threshold: float = 0.15):
        self.window = window
        self.high = high_threshold
        self.low = low_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        er = prices.apply(lambda s: efficiency_ratio(s, self.window))
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[er > self.high] = 1.0
        sig[er < self.low] = -1.0
        return sig


class ADXFilterFactor(Factor):
    """ADX 趋势强度过滤.

    输入需为 OHLC 面板 (含 high/low/close 列), 返回 ADX 值 (0-100).
    可用于过滤: 仅当 ADX > 阈值时启用趋势信号.
    """

    name = "adx"

    def __init__(self, window: int = 14):
        self.window = window

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        cols = [c for c in ("high", "low", "close") if c in ohlc.columns]
        if len(cols) < 3:
            raise ValueError("ADXFilterFactor requires high/low/close columns")
        return pd.DataFrame(
            {"adx": adx(ohlc, self.window)},
            index=ohlc.index,
        )


class ATRBreakoutFactor(Factor):
    """ATR 通道突破因子.

    价格突破 N×ATR 通道 (海龟风格, 波动率自适应):
    close > 前日收盘 + N×ATR → 多头; close < 前日收盘 - N×ATR → 空头.
    研报: 海龟法则, ATR 波动率通道.

    参数:
        atr_window: ATR 窗口
        mult: 通道倍数
    """

    name = "atr_breakout"

    def __init__(self, atr_window: int = 14, mult: float = 2.0):
        self.atr_window = atr_window
        self.mult = mult

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"ATRBreakoutFactor requires columns {req}")
        a = atr(ohlc, self.atr_window)
        prev_close = ohlc["close"].shift(1)
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["atr_breakout"])
        sig.loc[ohlc["close"] > prev_close + self.mult * a, "atr_breakout"] = 1.0
        sig.loc[ohlc["close"] < prev_close - self.mult * a, "atr_breakout"] = -1.0
        return sig


class BIASFactor(Factor):
    """乖离率因子.

    BIAS = (close - MA_N) / MA_N: 价格偏离均线程度.
    正乖离过大 → 超买; 负乖离过大 → 超卖.
    研报: 经典乖离率指标.

    参数:
        window: 均线窗口
        z_threshold: z-score 阈值
    """

    name = "bias"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ma = prices.rolling(self.window).mean()
        bias = prices / ma.replace(0, np.nan) - 1
        mean = bias.rolling(self.window).mean()
        std = bias.rolling(self.window).std()
        z = (bias - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = -1.0    # 正乖离过大 → 回落
        sig[z < -self.z_threshold] = 1.0    # 负乖离过大 → 反弹
        return sig

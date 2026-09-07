# coding=utf-8
"""波动率因子.

基于 continuous_main 面板:
    - HVFactor: 历史波动率 (年化)
    - ParkinsonFactor: Parkinson 区间波动率 (含 high/low)
    - VolRegimeFactor: 波动率状态 (低/高 regime)
    - VolSpreadFactor: 短长波动率差 (波动率期限结构)
    - GARCHFactor: GARCH(1,1) 条件波动率 (arch 库)
    - VolSkewnessFactor: 波动率偏度 (波动率分布的偏斜)
    - VolTargetPosition: 波动率目标仓位 (VT, 风险平价)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class HVFactor(Factor):
    """历史波动率 (年化).

    hv = std(ln_return, window) × sqrt(252)
    """

    name = "hv"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        return ret.rolling(self.window).std() * np.sqrt(252)


class ParkinsonFactor(Factor):
    """Parkinson 区间波动率 (年化).

    使用 high/low 区间, 比收盘价波动率更高效 (信息量更大).
    sigma = sqrt( (1/(4*ln2)) * mean(ln(H/L)^2) ) × sqrt(252)

    输入需含 high/low 列.
    """

    name = "parkinson"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "high" not in ohlc.columns or "low" not in ohlc.columns:
            raise ValueError("ParkinsonFactor requires high/low columns")
        log_range = np.log(ohlc["high"] / ohlc["low"])
        var = (log_range ** 2).rolling(self.window).mean() / (4 * np.log(2))
        return np.sqrt(var) * np.sqrt(252)


class VolRegimeFactor(Factor):
    """波动率状态因子.

    比较当前波动率与其长期均值的相对位置.
    signal = +1 高波动 regime, -1 低波动 regime (基于 HV z-score).

    参数:
        window: 短期波动窗口
        lookback: 长期基准窗口
        z_threshold: z-score 阈值
    """

    name = "vol_regime"

    def __init__(self, window: int = 20, lookback: int = 120, z_threshold: float = 1.0):
        self.window = window
        self.lookback = lookback
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        short_vol = ret.rolling(self.window).std() * np.sqrt(252)
        long_mean = short_vol.rolling(self.lookback).mean()
        long_std = short_vol.rolling(self.lookback).std()
        z = (short_vol - long_mean) / long_std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class VolSpreadFactor(Factor):
    """波动率期限结构因子.

    短周期波动 - 长周期波动: 正 = 短期波动放大 (风险上升).
    研报: AQR 波动率择时、兴业《波动率结构因子》.

    参数:
        short_window: 短期波动窗口 (默认 10)
        long_window: 长期波动窗口 (默认 60)
        z_threshold: z-score 阈值
    """

    name = "vol_spread"

    def __init__(self, short_window: int = 10, long_window: int = 60, z_threshold: float = 1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        short_vol = ret.rolling(self.short_window).std() * np.sqrt(252)
        long_vol = ret.rolling(self.long_window).std() * np.sqrt(252)
        spread = short_vol - long_vol
        mean = spread.rolling(self.long_window).mean()
        std = spread.rolling(self.long_window).std()
        z = (spread - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class GARCHFactor(Factor):
    """GARCH(1,1) 条件波动率因子.

    用 arch 库拟合 GARCH(1,1) 模型, 输出条件波动率.
    与 HV 相比, GARCH 捕捉波动率聚集与非对称性 (捕捉 ARCH 效应).
    研报: Duan (1995), 兴业证券波动率建模.

    参数:
        window: 拟合窗口 (滚动)
        p/q: GARCH 阶数 (默认 1,1)
    """

    name = "garch"

    def __init__(self, window: int = 252, p: int = 1, q: int = 1):
        self.window = window
        self.p = p
        self.q = q

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from arch import arch_model

        ret = prices.pct_change(fill_method=None).dropna()
        out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            s = ret[col].dropna() * 100  # 收益×100 提高数值稳定性
            if len(s) < self.window + 10:
                continue
            try:
                am = arch_model(s, vol="GARCH", p=self.p, q=self.q, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                cond_vol = res.conditional_volatility / 100
                # 对齐索引
                series = pd.Series(cond_vol.values, index=s.index)
                out.loc[series.index, col] = series * np.sqrt(252)
            except Exception:
                continue
        return out


class VolSkewnessFactor(Factor):
    """波动率偏度因子.

    滚动窗口内 |ret| (或波动率) 分布的偏度:
    正偏 → 波动率冲击上行 (风险上升), 负偏 → 波动率收敛.
    研报: 兴业证券《波动率结构因子》.

    参数:
        window: 偏度窗口
        threshold: 触发阈值
    """

    name = "vol_skewness"

    def __init__(self, window: int = 60, threshold: float = 0.3):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        # 绝对收益的偏度: 衡量波动率冲击的不对称
        vol_skew = ret.abs().rolling(self.window).skew()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[vol_skew > self.threshold] = 1.0    # 波动率冲击偏大 → 风险
        sig[vol_skew < -self.threshold] = -1.0  # 波动率收敛
        return sig


def vol_target_positions(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 3.0,
    daily_vol_target: pd.Series | None = None,
) -> pd.DataFrame:
    """波动率目标仓位.

    每品种: qty ∝ target_vol / rolling_vol, 限制杠杆.

    Args:
        signals: (T, N) 方向信号 (-1/0/+1)
        prices:  (T, N) 价格面板
        target_vol: 目标年化波动率 (0.15 = 15%)
        vol_lookback: 波动率回看窗口
        max_leverage: 最大杠杆
        daily_vol_target: (T,) 可选日频 vol_target 调节 (Zookeeper 状态变量接入用)
                       若提供则按日覆盖 target_vol.

    Returns:
        (T, N) 目标仓位手数 (含符号)
    """
    ret = np.log(prices).diff()
    vol = ret.rolling(vol_lookback).std() * np.sqrt(252)
    # 单位波动率敞口: 1/vol (vol 为 0 时用大数)
    weight_per_unit = target_vol / vol.replace(0, np.nan)
    if daily_vol_target is not None and len(daily_vol_target) > 0:
        dvt = daily_vol_target.reindex(weight_per_unit.index).ffill().fillna(target_vol)
        weight_per_unit = vol.replace(0, np.nan).div(dvt.values, axis=0)
    weight_per_unit = weight_per_unit.clip(upper=max_leverage)
    weight_per_unit = weight_per_unit.fillna(0)

    return signals * weight_per_unit

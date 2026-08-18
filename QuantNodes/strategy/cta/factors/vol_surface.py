"""波动率曲面套利因子.

利用IV曲面的定价异常生成期货交易信号:
- Skew极端 → 预测标的价格反转
- IV期限结构异常 → 预测波动率regime切换
- IV Z-Score → 波动率水平信号

数据来源: greeks_cache (option_greeks.parquet) + io_iv_skew_daily.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def extract_atm_iv_from_cache(greeks_df: pd.DataFrame) -> pd.DataFrame:
    """从greeks cache提取每日ATM IV面板 (date × product).

    ATM定义: |delta|最接近0.5的合约.
    """
    df = greeks_df[greeks_df["valid"]].copy()
    df["abs_delta"] = df["delta"].abs()
    # 每天每品种选|delta|最接近0.5的
    df["atm_rank"] = df.groupby(["trade_date", "product"])["abs_delta"].transform(
        lambda x: (x - 0.5).abs()
    )
    atm = df.loc[df.groupby(["trade_date", "product"])["atm_rank"].idxmin()]
    panel = atm.pivot_table(index="trade_date", columns="product", values="iv")
    return panel.sort_index()


def extract_skew_from_cache(greeks_df: pd.DataFrame) -> pd.DataFrame:
    """从greeks cache提取每日Skew面板 (date × product).

    Skew = IV(25-delta put) - IV(25-delta call)
    """
    df = greeks_df[greeks_df["valid"]].copy()
    df["abs_delta"] = df["delta"].abs()

    # 25-delta put: delta < 0 且 |delta| ≈ 0.25
    puts = df[(df["option_type"] == "P") & (df["abs_delta"].between(0.15, 0.35))]
    # 25-delta call: delta > 0 且 |delta| ≈ 0.25
    calls = df[(df["option_type"] == "C") & (df["abs_delta"].between(0.15, 0.35))]

    # 每天每品种选最接近0.25 delta的
    if puts.empty or calls.empty:
        return pd.DataFrame()

    puts["rank"] = puts.groupby(["trade_date", "product"])["abs_delta"].transform(
        lambda x: (x - 0.25).abs()
    )
    best_puts = puts.loc[puts.groupby(["trade_date", "product"])["rank"].idxmin()]
    put_iv = best_puts.pivot_table(index="trade_date", columns="product", values="iv")

    calls["rank"] = calls.groupby(["trade_date", "product"])["abs_delta"].transform(
        lambda x: (x - 0.25).abs()
    )
    best_calls = calls.loc[calls.groupby(["trade_date", "product"])["rank"].idxmin()]
    call_iv = best_calls.pivot_table(index="trade_date", columns="product", values="iv")

    skew = put_iv - call_iv
    return skew.sort_index()


class SkewReversalFactor(Factor):
    """Skew反转因子.

    当skew (25-delta put IV - 25-delta call IV) 处于极端水平时,
    预测标的价格反转:
    - Skew极高 (put skew很陡): 市场过度恐惧 → 标的可能反弹 → 多头
    - Skew极低 (call skew很陡): 市场过度贪婪 → 标的可能回落 → 空头

    参数:
        window: z-score计算窗口
        z_threshold: 极端阈值
    """

    name = "skew_reversal"

    def __init__(self, window: int = 60, z_threshold: float = 1.5):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, skew_df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """计算skew反转信号.

        Args:
            skew_df: DataFrame (date × product), 值为skew (put_iv - call_iv)

        Returns:
            DataFrame (date × product), 信号: +1 标的将涨, -1 标的将跌
        """
        if skew_df is None or skew_df.empty:
            return pd.DataFrame()

        # z-score
        mean = skew_df.rolling(self.window, min_periods=20).mean()
        std = skew_df.rolling(self.window, min_periods=20).std()
        z = (skew_df - mean) / std.replace(0, np.nan)

        # 信号: skew极高 → 市场恐惧 → 标的反弹 → +1
        sig = pd.DataFrame(0.0, index=z.index, columns=z.columns)
        sig[z > self.z_threshold] = 1.0    # skew极陡 → 恐惧过度 → 多
        sig[z < -self.z_threshold] = -1.0  # skew极平 → 贪婪过度 → 空
        return sig


class IVRegimeFactor(Factor):
    """IV Regime因子.

    利用IV的regime变化预测标的价格方向:
    - IV从极低位快速上升 → 波动率扩张 → 标的可能有趋势 → 跟随趋势
    - IV从极高位快速下降 → 波动率收缩 → 标的可能横盘 → 减仓

    返回: 仓位乘数 (0.5-1.5), 1.0=不变

    参数:
        iv_lookback: IV百分位回看窗口
        iv_momentum_window: IV动量窗口
    """

    name = "iv_regime"

    def __init__(self, iv_lookback: int = 252, iv_momentum_window: int = 20):
        self.iv_lookback = iv_lookback
        self.iv_momentum_window = iv_momentum_window

    def compute(self, iv_panel: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """计算IV regime信号.

        Args:
            iv_panel: DataFrame (date × product), ATM IV值

        Returns:
            DataFrame (date × product), 值为仓位乘数 (0.5-1.5)
        """
        if iv_panel is None or iv_panel.empty:
            return pd.DataFrame()

        # IV百分位
        iv_pct = iv_panel.rolling(self.iv_lookback, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # IV动量
        iv_mom = iv_panel.pct_change(self.iv_momentum_window)

        # 信号: IV低+上升 → 加仓, IV高+下降 → 减仓
        signal = pd.DataFrame(1.0, index=iv_panel.index, columns=iv_panel.columns)

        # IV低 (百分位<30%) 且上升 → 加仓
        low_rising = (iv_pct < 0.30) & (iv_mom > 0)
        signal[low_rising] = 1.3

        # IV高 (百分位>70%) 且下降 → 减仓
        high_falling = (iv_pct > 0.70) & (iv_mom < 0)
        signal[high_falling] = 0.7

        # IV极高 (百分位>90%) → 大幅减仓
        extreme_high = (iv_pct > 0.90)
        signal[extreme_high] = 0.5

        return signal


class IVZScoreFactor(Factor):
    """IV Z-Score因子.

    IV z-score极端时预测均值回归:
    - z > 2: IV极高 → 波动率溢价大 → 做空波动率 (卖跨式方向)
    - z < -2: IV极低 → 波动率被低估 → 做多波动率 (买跨式方向)

    对期货的影响:
    - IV极高时: 标的可能有大幅波动, 谨慎/减仓
    - IV极低时: 标的可能启动趋势, 跟随

    参数:
        window: z-score计算窗口
    """

    name = "iv_zscore"

    def __init__(self, window: int = 60):
        self.window = window

    def compute(self, iv_panel: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """计算IV z-score信号.

        Returns:
            DataFrame (date × product), 信号: +1/-1/0
        """
        if iv_panel is None or iv_panel.empty:
            return pd.DataFrame()

        mean = iv_panel.rolling(self.window, min_periods=20).mean()
        std = iv_panel.rolling(self.window, min_periods=20).std()
        z = (iv_panel - mean) / std.replace(0, np.nan)

        sig = pd.DataFrame(0.0, index=z.index, columns=z.columns)
        sig[z > 2.0] = -1.0   # IV极高 → 减仓/做空波动率
        sig[z < -2.0] = 1.0    # IV极低 → 跟随趋势
        return sig

"""波动率曲面套利因子.

利用IV曲面的定价异常生成期货交易信号:
- Skew极端 → 预测标的价格反转
- IV期限结构异常 → 预测波动率regime切换
- VRP极端 → 预测均值回归

数据来源: greeks_cache (option_greeks.parquet) + io_iv_skew_daily.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class SkewReversalFactor(Factor):
    """Skew反转因子.

    当skew (25-delta put IV - 25-delta call IV) 处于极端水平时,
    预测标的价格反转:
    - Skew极高 (put skew很陡): 市场过度恐惧 → 标的可能反弹 → 多头
    - Skew极低 (call skew很陡): 市场过度贪婪 → 标的可能回落 → 空头

    数据来源: io_iv_skew_daily.parquet (IO期权) 或 greeks_cache

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
            skew_df: DataFrame with columns [iv_skew, iv_atm], index=DatetimeIndex
                     或 None (使用kwargs中的数据)

        Returns:
            DataFrame (date × product), 信号: +1 标的将涨, -1 标的将跌
        """
        if skew_df is None:
            # 尝试从kwargs获取
            skew_df = kwargs.get("skew_data")
            if skew_df is None:
                return pd.DataFrame()

        if skew_df is None or skew_df.empty:
            return pd.DataFrame()

        # 确保iv_skew列存在
        if "iv_skew" not in skew_df.columns:
            return pd.DataFrame()

        skew = skew_df["iv_skew"].copy()
        skew.index = pd.to_datetime(skew.index)

        # z-score
        mean = skew.rolling(self.window, min_periods=20).mean()
        std = skew.rolling(self.window, min_periods=20).std()
        z = (skew - mean) / std.replace(0, np.nan)

        # 信号: skew极高 → 市场恐惧 → 标的反弹 → +1
        sig = pd.Series(0.0, index=z.index)
        sig[z > self.z_threshold] = 1.0    # skew极陡 → 恐惧过度 → 多
        sig[z < -self.z_threshold] = -1.0  # skew极平 → 贪婪过度 → 空

        # IO只有一个品种, 返回单列DataFrame
        return sig.to_frame("IO")


class IVRegimeFactor(Factor):
    """IV Regime因子.

    利用IV的regime变化预测标的价格方向:
    - IV从极低位快速上升 → 波动率扩张 → 标的可能有趋势 → 跟随趋势
    - IV从极高位快速下降 → 波动率收缩 → 标的可能横盘 → 减仓

    与现有IVTiming策略的区别:
    - IVTiming: IV低→买跨式 (期权策略)
    - IVRegime: IV变化→调整期货仓位 (CTA增强)

    参数:
        iv_lookback: IV百分位回看窗口
        iv_momentum_window: IV动量窗口
    """

    name = "iv_regime"

    def __init__(self, iv_lookback: int = 252, iv_momentum_window: int = 20):
        self.iv_lookback = iv_lookback
        self.iv_momentum_window = iv_momentum_window

    def compute(self, prices: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """计算IV regime信号.

        需要ATM IV数据 (从greeks_cache或iv_panel获取).
        如果没有IV数据, 返回空DataFrame.

        Returns:
            DataFrame (date × product), 值为仓位乘数 (0.5-1.5)
        """
        iv_panel = kwargs.get("iv_panel")
        if iv_panel is None or iv_panel.empty:
            return pd.DataFrame()

        # IV百分位
        iv_pct = iv_panel.rolling(self.iv_lookback, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # IV动量
        iv_mom = iv_panel.pct_change(self.iv_momentum_window)

        # 信号: IV低+上升 → 加仓 (乘数>1), IV高+下降 → 减仓 (乘数<1)
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

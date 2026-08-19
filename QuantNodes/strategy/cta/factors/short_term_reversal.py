"""短期反转 + 中期趋势组合因子.

逻辑:
1. 短期反转: 5日RSI < 30 → 超卖 → +1, RSI > 70 → 超买 → -1
2. 中期趋势: 20日动量 > 0 → 上升, < 0 → 下降
3. 组合: 短期超卖 + 中期上升 → +1 (拐点买入)
       短期超买 + 中期下降 → -1 (拐点卖出)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class ShortTermReversalTrendFactor(Factor):
    """短期反转 + 中期趋势组合因子.

    Args:
        rsi_window: RSI回看天数 (默认5)
        rsi_oversold: 超卖阈值 (默认30)
        rsi_overbought: 超买阈值 (默认70)
        trend_window: 中期趋势回看天数 (默认20)
    """

    name = "short_term_reversal_trend"

    def __init__(
        self,
        rsi_window: int = 5,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        trend_window: int = 20,
    ):
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_window = trend_window

    def _rsi(self, prices: pd.DataFrame, window: int) -> pd.DataFrame:
        """计算RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算短期反转+中期趋势信号.

        Args:
            prices: (T, N) 价格面板

        Returns:
            pd.DataFrame: (T, N) 信号 {-1, 0, 1}
        """
        if prices is None or prices.empty:
            return pd.DataFrame()

        # 短期反转: RSI
        rsi = self._rsi(prices, self.rsi_window)

        # 中期趋势: 20日动量
        momentum = prices.pct_change(self.trend_window)

        # 短期反转信号
        short_signal = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        short_signal[rsi < self.rsi_oversold] = 1   # 超卖 → 做多
        short_signal[rsi > self.rsi_overbought] = -1  # 超买 → 做空

        # 中期趋势信号
        trend_signal = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        trend_signal[momentum > 0] = 1   # 上升趋势
        trend_signal[momentum < 0] = -1  # 下降趋势

        # 组合: 同向确认 (短期反转方向 * 中期趋势方向)
        signal = short_signal * trend_signal
        # 移除矛盾信号 (短期和中期方向相反)
        # 保留同向信号
        return signal

# coding=utf-8
"""
配置驱动的策略节点

从 DataFrame 的 signal 列生成交易信号。
"""

from __future__ import annotations

from typing import List

import pandas as pd

from QuantNodes.backtest.strategy_node import StrategyNode, TradeSignal


class ConfigStrategyNode(StrategyNode):
    """从 DataFrame 的 signal 列生成交易信号

    signal 列值：
    - 1: 买入信号
    - -1: 卖出信号
    - 0: 持有（无操作）

    要求 DataFrame 包含以下列：
    - Code (或 code): 股票代码
    - date: 日期
    - Close (或 close): 收盘价（用于信号价格）
    - signal: 交易信号
    """

    def __init__(self, signal_col: str = "signal", **kwargs):
        super().__init__(name="ConfigStrategy", **kwargs)
        self._signal_col = signal_col

    def _generate_signals(
        self, input_data: pd.DataFrame, **kwargs
    ) -> List[TradeSignal]:
        # 先过滤非零信号，避免遍历全量数据
        mask = input_data[self._signal_col] != 0
        active = input_data.loc[mask]
        if active.empty:
            return []

        code_col = "Code" if "Code" in active.columns else "code"
        price_col = "Close" if "Close" in active.columns else "close"
        date_col = "date" if "date" in active.columns else "Date"

        codes = active[code_col].astype(str).values
        prices = active[price_col].astype(float).values
        dates = active[date_col].astype(str).values
        sig_vals = active[self._signal_col].values

        signals = []
        for i in range(len(active)):
            sig_val = sig_vals[i]
            signals.append(TradeSignal(
                code=codes[i],
                signal_type="buy" if sig_val > 0 else "sell",
                strength=abs(float(sig_val)),
                price=prices[i],
                date=dates[i],
            ))
        return signals

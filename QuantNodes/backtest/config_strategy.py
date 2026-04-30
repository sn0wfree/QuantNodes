# coding=utf-8
"""
配置驱动的策略节点

从 DataFrame 的 signal 列生成交易信号。
"""

from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from QuantNodes.backtest.strategy_node import StrategyNode, Signal


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
    ) -> List[Signal]:
        signals = []
        for _, row in input_data.iterrows():
            sig_val = row.get(self._signal_col, 0)
            if sig_val == 0:
                continue

            code = str(row.get("Code", row.get("code", "")))
            price = float(row.get("Close", row.get("close", 0)))
            date = str(row.get("date", ""))

            signals.append(Signal(
                code=code,
                signal_type="buy" if sig_val > 0 else "sell",
                strength=abs(float(sig_val)),
                price=price,
                date=date,
            ))
        return signals

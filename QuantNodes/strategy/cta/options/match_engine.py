# coding=utf-8
"""交易匹配引擎 — 从 期权工作平台/option_analysis_monitor.py 移植.

核心:
    - DequeMatcher: FIFO/LIFO 交易匹配 (开仓→平仓配对)
    - 用于计算已平仓期权的 P&L
"""
from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterable

import numpy as np
import pandas as pd


class DequeMatcher(deque):
    """基于 deque 的交易匹配器 (FIFO/LIFO).

    用法:
        matcher = DequeMatcher(buy_records.to_dict('records'))
        avg_price, matched = DequeMatcher.single_contract_recurs_match(
            transactions, method='FIFO'
        )
    """

    def __init__(
        self,
        *args,
        k=None,
        contract_col: str = "contract_code",
        share_col: str = "手数",
        price_col: str = "成交均价",
        trade_type: str = "trade_type",
        dt_col: str = "日期",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if k is not None:
            if isinstance(k, Iterable):
                self.extend(list(k))
        self._share_col = share_col
        self._price_col = price_col
        self._contract_col = contract_col
        self._dt_col = dt_col
        self._trade_type = trade_type

    @property
    def total_shares(self) -> float:
        return sum(s[self._share_col] for s in self)

    @property
    def avg_price(self) -> float | None:
        total = self.total_shares
        if total == 0:
            return None
        return sum(s[self._price_col] * s[self._share_col] for s in self) / total

    @classmethod
    def single_contract_recurs_match(
        cls,
        transactions_1contract: pd.DataFrame,
        contract_col: str = "contract_code",
        share_col: str = "手数",
        price_col: str = "成交均价",
        trade_type: str = "trade_type",
        dt_col: str = "日期",
        method: str = "FIFO",
    ) -> tuple[float | None, list]:
        """单合约递归匹配 (开仓→平仓).

        Args:
            transactions_1contract: 单合约的交易记录
            method: 'FIFO' 或 'LIFO'

        Returns:
            (开仓均价, matched_pairs)
        """
        buy_records = transactions_1contract[
            transactions_1contract[trade_type] == 1
        ]
        buy_deque = cls(
            k=buy_records.to_dict("records"),
            contract_col=contract_col,
            share_col=share_col,
            price_col=price_col,
            trade_type=trade_type,
            dt_col=dt_col,
        )

        sell_records = transactions_1contract[
            transactions_1contract[trade_type] == -1
        ]
        sell_deque = cls(
            k=sell_records.to_dict("records"),
            contract_col=contract_col,
            share_col=share_col,
            price_col=price_col,
            trade_type=trade_type,
            dt_col=dt_col,
        )

        h = []
        avg_price = cls._recur_match_core(
            buy_deque, sell_deque, h,
            contract_col=contract_col,
            share_col=share_col,
            price_col=price_col,
            trade_type=trade_type,
            dt_col=dt_col,
            method=method,
        )
        return avg_price, h

    @classmethod
    def _recur_match_core(
        cls,
        buy_deque,
        sell_deque,
        h: list,
        contract_col: str = "contract_code",
        share_col: str = "手数",
        price_col: str = "成交均价",
        trade_type: str = "trade_type",
        dt_col: str = "日期",
        method: str = "FIFO",
    ):
        if method == "FIFO":
            popfunc = lambda x: x.popleft()
            appendfunc = lambda dq, x: dq.appendleft(x)
        elif method == "LIFO":
            popfunc = lambda x: x.pop()
            appendfunc = lambda dq, x: dq.append(x)
        else:
            raise ValueError(f"Unknown method: {method}")

        if len(sell_deque) == 0:
            return buy_deque.avg_price, h

        if len(buy_deque) == 0:
            raise ValueError("No buy data left!")

        buy_total = buy_deque.total_shares
        sell_total = sell_deque.total_shares

        if buy_total < sell_total:
            raise ValueError("Buy side less than sell side!")

        buy1 = popfunc(buy_deque)
        sell1 = popfunc(sell_deque)
        buy_shares = buy1[share_col]
        sell_shares = sell1[share_col]

        if buy_shares == sell_shares:
            h.append((buy1, sell1))
        elif buy_shares > sell_shares:
            remaining = copy.deepcopy(buy1)
            matched = copy.deepcopy(buy1)
            remaining[share_col] = buy_shares - sell_shares
            matched[share_col] = sell_shares
            appendfunc(buy_deque, remaining)
            h.append((matched, sell1))
        else:
            remaining = copy.deepcopy(sell1)
            matched = copy.deepcopy(sell1)
            matched[share_col] = buy_shares
            remaining[share_col] = sell_shares - buy_shares
            appendfunc(sell_deque, remaining)
            h.append((buy1, matched))

        return cls._recur_match_core(
            buy_deque, sell_deque, h,
            contract_col=contract_col,
            share_col=share_col,
            price_col=price_col,
            trade_type=trade_type,
            dt_col=dt_col,
            method=method,
        )

    @classmethod
    def multi_contract_recurs_match(
        cls,
        transactions_multi: pd.DataFrame,
        contract_col: str = "contract_code",
        share_col: str = "手数",
        price_col: str = "成交均价",
        trade_type: str = "trade_type",
        dt_col: str = "报单日期",
        direction_col: str = "买卖开平",
        method: str = "FIFO",
    ):
        """多合约递归匹配.

        Args:
            transactions_multi: 全部交易记录
            direction_col: 方向列 (买开/卖平/卖开/买平)

        Yields:
            (contract, direction, (avg_price, matched_pairs))
        """
        for contract, sub in transactions_multi.sort_values(dt_col).groupby(
            contract_col
        ):
            # 多头: 买开 + 卖平
            long_mask = sub[direction_col].isin(["买开", "卖平"])
            if not sub[long_mask].empty:
                res = cls.single_contract_recurs_match(
                    sub[long_mask],
                    contract_col=contract_col,
                    share_col=share_col,
                    price_col=price_col,
                    trade_type=trade_type,
                    dt_col=dt_col,
                    method=method,
                )
                yield contract, "多头", res

            # 空头: 卖开 + 买平
            short_mask = sub[direction_col].isin(["卖开", "买平"])
            if not sub[short_mask].empty:
                res = cls.single_contract_recurs_match(
                    sub[short_mask],
                    contract_col=contract_col,
                    share_col=share_col,
                    price_col=price_col,
                    trade_type=trade_type,
                    dt_col=dt_col,
                    method=method,
                )
                yield contract, "空头", res

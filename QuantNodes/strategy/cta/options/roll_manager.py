# coding=utf-8
"""Roll Manager: 到期前自动展期 straddle 到下一月份.

Gamma Scalping 展期流程:
    1. 检测持仓到期天数 ≤ roll_days
    2. 平旧 straddle (call + put)
    3. 选新 ATM straddle (下一到期月)
    4. 开新 straddle (call + put)
    5. 记录展期成本

展期策略:
    - 到期前 roll_days 天触发
    - 平旧仓用当前市场价
    - 开新仓用 select_straddle 选 ATM
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .selector import OptionSelection, StraddleSelection, select_straddle


@dataclass(slots=True)
class RollAction:
    """单次展期操作记录."""

    date: str
    product: str
    close_call: str           # 平仓旧 call symbol
    close_put: str            # 平仓旧 put symbol
    open_call: OptionSelection | None  # 新 call 腿
    open_put: OptionSelection | None   # 新 put 腿
    cost: float               # 展期成本 (平旧 + 开新)
    reason: str               # "expiry_approaching" / "manual"


@dataclass
class RollManager:
    """展期管理器.

    Args:
        roll_days: 到期前多少天触发展期 (默认 7)
    """

    roll_days: int = 7
    _records: list[RollAction] = field(default_factory=list)

    def check_and_roll(
        self,
        date: str | pd.Timestamp,
        holdings: dict,
        prices: dict[str, float],
        loader=None,
        um=None,
        oe=None,
    ) -> list[RollAction]:
        """检查所有持仓, 到期前 roll.

        Args:
            date: 当前交易日
            holdings: {product: {"type": "straddle", "call_sym", "put_sym", "expiry", ...}}
            prices: {product: 当日收盘价}
            loader: CtaDataLoader (可选)
            um: UnderlyingMap (可选)
            oe: OptionExpiry (可选)

        Returns:
            list[RollAction] — 本次执行的展期操作
        """
        ts = pd.Timestamp(date)
        actions: list[RollAction] = []

        for product, h in list(holdings.items()):
            if h.get("type") != "straddle":
                continue

            expiry = pd.Timestamp(h["expiry"])
            days_left = (expiry - ts).days

            if days_left > self.roll_days:
                continue

            # 需要 roll
            spot = prices.get(product)
            if spot is None or spot <= 0:
                continue

            # 选新 straddle (下一月份)
            new_straddle = select_straddle(
                product, ts, spot,
                min_days=30, max_days=90,  # 宽窗口, 选最近到期
                loader=loader, um=um, oe=oe,
            )

            cost = 0.0
            if new_straddle is not None:
                # 新 straddle 成本
                cost = new_straddle.cost
                # 加上旧 straddle 平仓价值 (内在价值, 近似为 0 因为快到期)
                # 实际由 broker 平仓处理

            action = RollAction(
                date=str(ts.date()),
                product=product,
                close_call=h.get("call_sym", ""),
                close_put=h.get("put_sym", ""),
                open_call=new_straddle.call if new_straddle else None,
                open_put=new_straddle.put if new_straddle else None,
                cost=cost,
                reason="expiry_approaching",
            )
            actions.append(action)
            self._records.append(action)

        return actions

    def get_records(self) -> pd.DataFrame:
        """返回展期记录 DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": r.date,
                "product": r.product,
                "close_call": r.close_call,
                "close_put": r.close_put,
                "open_call": r.open_call.symbol if r.open_call else None,
                "open_put": r.open_put.symbol if r.open_put else None,
                "cost": r.cost,
                "reason": r.reason,
            }
            for r in self._records
        ])

    def get_total_roll_cost(self) -> float:
        """返回累计展期成本."""
        return sum(r.cost for r in self._records)

    def reset(self):
        """重置状态."""
        self._records.clear()

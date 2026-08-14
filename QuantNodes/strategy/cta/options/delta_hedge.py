# coding=utf-8
"""Delta Hedge Manager: 聚合 option delta → 计算 futures 对冲目标.

Gamma Scalping 核心组件:
    1. aggregate_delta: option deltas → {product: total_delta}
    2. compute_hedge_target: portfolio delta → {futures_symbol: target_qty}
    3. 记录 hedge 历史

设计:
    - Manager 只做聚合 + 目标计算, 不直接操作 broker
    - 策略层负责: 每日计算 option deltas → 调用 aggregate_delta → broker.place_order()
    - record_rebalance 跟踪调仓记录 + P&L
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.cta_broker import CtaBroker
from ..backtest.cta_contract import CtaContractRegistry


@dataclass(slots=True)
class HedgeRecord:
    """单次 delta hedge 调仓记录."""

    date: str
    product: str
    option_delta: float        # 当日期权总 delta × multiplier
    futures_symbol: str        # 对冲期货合约
    old_hedge_qty: float       # 调仓前对冲手数
    new_hedge_qty: float       # 调仓后对冲手数
    price: float               # 期货成交价
    cost: float                # 手续费
    hedge_pnl: float = 0.0     # 对冲浮盈 (dS × hedge_qty × mult)


@dataclass
class DeltaHedgeManager:
    """Delta 对冲管理器.

    Args:
        registry: 合约规格 (乘数/保证金)
        commission_rate: 手续费率
    """

    registry: CtaContractRegistry
    commission_rate: float = 0.0001
    _hedge_positions: dict[str, float] = field(default_factory=dict)
    _records: list[HedgeRecord] = field(default_factory=list)

    # ── 计算 ────────────────────────────────────────────────────────────

    def aggregate_delta(
        self,
        option_deltas: list[tuple[str, float, float]],
    ) -> dict[str, float]:
        """聚合 option delta 到品种级别.

        Args:
            option_deltas: [(product, delta × multiplier × qty, multiplier), ...]
                delta 含方向: 买方 delta 为正, 卖方 delta 为负

        Returns:
            {product: total_delta} (含乘数, 可直接用于对冲计算)
        """
        result: dict[str, float] = {}
        for product, delta, _mult in option_deltas:
            result[product] = result.get(product, 0.0) + delta
        return result

    def compute_hedge_target(
        self,
        portfolio_delta: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, float]:
        """portfolio delta → {futures_symbol: target_qty}.

        对冲公式: hedge_qty = -total_delta / (price × multiplier)
        total_delta 含乘数, 除以 price × multiplier 得到手数.

        Args:
            portfolio_delta: {product: total_delta}
            prices: {product: 当日收盘价}

        Returns:
            {futures_symbol: target_qty} (正=多, 负=空)
        """
        targets: dict[str, float] = {}
        for product, delta in portfolio_delta.items():
            if abs(delta) < 1e-9:
                continue
            price = prices.get(product)
            if price is None or price <= 0:
                continue
            spec = self.registry.get_or_default(product)
            mult = spec.contract_multiplier
            hedge_qty = -delta / (price * mult)
            hedge_qty = float(np.round(hedge_qty))
            if abs(hedge_qty) < 1e-9:
                continue
            targets[product] = hedge_qty
        return targets

    # ── 执行跟踪 ──────────────────────────────────────────────────────────

    def record_rebalance(
        self,
        date: str,
        product: str,
        option_delta: float,
        old_hedge_qty: float,
        new_hedge_qty: float,
        price: float,
        multiplier: float,
        prev_price: float | None = None,
    ) -> HedgeRecord:
        """记录一次 hedge 调仓, 计算 hedge P&L."""
        fee = abs(new_hedge_qty - old_hedge_qty) * price * multiplier * self.commission_rate
        hedge_pnl = 0.0
        if prev_price is not None and prev_price > 0:
            hedge_pnl = -new_hedge_qty * (price - prev_price) * multiplier

        rec = HedgeRecord(
            date=date,
            product=product,
            option_delta=option_delta,
            futures_symbol=product,
            old_hedge_qty=old_hedge_qty,
            new_hedge_qty=new_hedge_qty,
            price=price,
            cost=fee,
            hedge_pnl=hedge_pnl,
        )
        self._records.append(rec)
        return rec

    def get_records(self) -> pd.DataFrame:
        """返回 hedge 记录 DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": r.date,
                "product": r.product,
                "option_delta": r.option_delta,
                "old_hedge": r.old_hedge_qty,
                "new_hedge": r.new_hedge_qty,
                "price": r.price,
                "cost": r.cost,
                "hedge_pnl": r.hedge_pnl,
            }
            for r in self._records
        ])

    def get_hedge_pnl(self) -> pd.Series:
        """返回累计 hedge P&L 时间序列."""
        if not self._records:
            return pd.Series(dtype=float)
        df = self.get_records()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["hedge_pnl"].cumsum()

    def reset(self):
        """重置状态."""
        self._hedge_positions.clear()
        self._records.clear()

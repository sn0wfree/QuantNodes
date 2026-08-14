# coding=utf-8
"""Delta Hedge Manager: 聚合 option delta → 计算对冲目标.

Gamma Scalping 核心组件:
    1. aggregate_delta: option deltas → {product: total_delta}
    2. compute_hedge_target: portfolio delta → futures hedge qty
    3. compute_option_delta_adjustment: 用期权调整 delta (买方/卖方可配)
    4. 记录 hedge 历史

delta 单位: greek_delta × qty × multiplier (每 $1 变动的美元敞口).
    - 1 手期货: dollar_delta = 1 × multiplier
    - 1 手 ATM call: dollar_delta ≈ +0.5 × multiplier
    - 1 手 ATM put:  dollar_delta ≈ -0.5 × multiplier
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
    option_delta: float        # 当日期权总 delta (含乘数)
    old_hedge_qty: float       # 调仓前对冲手数
    new_hedge_qty: float       # 调仓后对冲手数
    price: float               # 成交价
    cost: float                # 手续费
    hedge_pnl: float = 0.0     # 对冲浮盈


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

    # ── 计算: option delta 聚合 ──────────────────────────────────────────

    def aggregate_delta(
        self,
        option_deltas: list[tuple[str, float, float]],
    ) -> dict[str, float]:
        """聚合 option delta 到品种级别.

        Args:
            option_deltas: [(product, dollar_delta, multiplier), ...]

        Returns:
            {product: total_dollar_delta}
        """
        result: dict[str, float] = {}
        for product, delta, _mult in option_deltas:
            result[product] = result.get(product, 0.0) + delta
        return result

    # ── 计算: futures 对冲 ───────────────────────────────────────────────

    def compute_hedge_target(
        self,
        portfolio_delta: dict[str, float],
    ) -> dict[str, float]:
        """portfolio delta → {product: futures_hedge_qty}.

        公式: hedge_qty = -dollar_delta / multiplier
        (1 手期货 dollar_delta = 1 × multiplier)

        Args:
            portfolio_delta: {product: dollar_delta}

        Returns:
            {product: target_qty} (正=多, 负=空)
        """
        targets: dict[str, float] = {}
        for product, delta in portfolio_delta.items():
            if abs(delta) < 1e-9:
                continue
            spec = self.registry.get_or_default(product)
            mult = spec.contract_multiplier
            hedge_qty = -delta / mult
            hedge_qty = float(np.round(hedge_qty))
            if abs(hedge_qty) < 1e-9:
                continue
            targets[product] = hedge_qty
        return targets

    # ── 计算: 期权调 delta ───────────────────────────────────────────────

    def compute_option_delta_adjustment(
        self,
        current_delta: float,
        target_delta: float = 0.0,
        product: str = "",
        side: str = "buy",
    ) -> tuple[str, float, float]:
        """用期权调整 delta → (option_type, qty, delta_per_lot).

        Args:
            current_delta: 当前组合 dollar_delta
            target_delta: 目标 dollar_delta (通常 0 = delta-neutral)
            product: 品种代码 (查乘数)
            side: "buy" = 买方调 delta, "sell" = 卖方调 delta

        Returns:
            (option_type, qty, delta_per_lot)
            option_type: "C" / "P" / "" (无需调整)
            qty: 手数 (正数)
            delta_per_lot: 每手该期权的 dollar_delta (含乘数)
        """
        gap = target_delta - current_delta
        if abs(gap) < 1e-9:
            return ("", 0.0, 0.0)

        spec = self.registry.get_or_default(product)
        mult = spec.contract_multiplier

        # ATM delta ≈ ±0.5
        if gap > 0:
            opt_type = "C"
            delta_per_lot = 0.5 * mult
        else:
            opt_type = "P"
            delta_per_lot = -0.5 * mult

        qty = abs(gap) / abs(delta_per_lot)
        return (opt_type, float(np.round(qty)), delta_per_lot)

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
        """记录一次 hedge 调仓."""
        fee = abs(new_hedge_qty - old_hedge_qty) * price * multiplier * self.commission_rate
        hedge_pnl = 0.0
        if prev_price is not None and prev_price > 0:
            hedge_pnl = -new_hedge_qty * (price - prev_price) * multiplier

        rec = HedgeRecord(
            date=date,
            product=product,
            option_delta=option_delta,
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

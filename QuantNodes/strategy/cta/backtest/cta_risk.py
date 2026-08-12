# coding=utf-8
"""CTA 风控.

策略 target_positions 输出目标持仓后, 经风控调整:
    - MarginRisk:       保证金占用上限 (按最新价 × 乘数 × 保证金比例)
    - PositionLimitRisk: 单品种手数上限 / 最大敞口
    - CompositeRisk:     链式组合
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .cta_contract import CtaContractRegistry


class CtaRisk(ABC):
    """风控基类."""

    @abstractmethod
    def apply(
        self,
        targets: dict[str, float],
        mark_prices: dict[str, float],
        index: int,
        dates: list[pd.Timestamp],
        prices: dict[str, list[float]],
    ) -> dict[str, float]:
        """调整目标持仓."""
        raise NotImplementedError


class MarginRisk(CtaRisk):
    """保证金占用上限风控.

    按比例缩放全部目标持仓, 使总保证金占用 ≤ 上限.
    """

    def __init__(
        self,
        registry: CtaContractRegistry,
        products: dict[str, str],
        equity: float,
        max_margin_ratio: float = 0.8,
    ):
        self._registry = registry
        self._products = products
        self._equity = equity
        self._max_margin_ratio = max_margin_ratio

    def apply(self, targets, mark_prices, index, dates, prices):
        total_margin = 0.0
        for symbol, qty in targets.items():
            if qty == 0:
                continue
            price = mark_prices.get(symbol)
            if price is None or price <= 0:
                continue
            spec = self._registry.get_or_default(self._products.get(symbol, symbol))
            total_margin += abs(qty) * price * spec.contract_multiplier * spec.margin_ratio

        if total_margin <= 0:
            return targets

        max_margin = self._equity * self._max_margin_ratio
        if total_margin <= max_margin:
            return targets

        scale = max_margin / total_margin
        return {s: qty * scale for s, qty in targets.items()}


class PositionLimitRisk(CtaRisk):
    """单品种最大手数/敞口限制."""

    def __init__(
        self,
        max_qty_per_product: Optional[dict[str, float]] = None,
        max_notional: Optional[float] = None,
        products: Optional[dict[str, str]] = None,
        registry: Optional[CtaContractRegistry] = None,
    ):
        self._max_qty = max_qty_per_product or {}
        self._max_notional = max_notional
        self._products = products or {}
        self._registry = registry or CtaContractRegistry()

    def apply(self, targets, mark_prices, index, dates, prices):
        out = dict(targets)
        # 单品种手数上限
        if self._max_qty:
            for symbol, qty in list(out.items()):
                product = self._products.get(symbol, symbol)
                cap = self._max_qty.get(product)
                if cap is not None and abs(qty) > cap:
                    out[symbol] = cap if qty > 0 else -cap
        # 总名义价值上限
        if self._max_notional:
            total = 0.0
            for symbol, qty in out.items():
                price = mark_prices.get(symbol)
                if price is None or price <= 0:
                    continue
                spec = self._registry.get_or_default(self._products.get(symbol, symbol))
                total += abs(qty) * price * spec.contract_multiplier
            if total > self._max_notional:
                scale = self._max_notional / total
                out = {s: q * scale for s, q in out.items()}
        return out


class CompositeRisk(CtaRisk):
    """链式组合风控."""

    def __init__(self, risks: list[CtaRisk]):
        self._risks = risks

    def apply(self, targets, mark_prices, index, dates, prices):
        out = targets
        for risk in self._risks:
            out = risk.apply(out, mark_prices, index, dates, prices)
        return out

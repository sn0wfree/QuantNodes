# coding=utf-8
"""CTA 期货/期权经纪商 — 仓位与现金管理.

基于合约乘数计算名义价值与保证金占用:
    - 名义价值 = 价格 × 合约乘数 × 手数
    - 保证金   = 名义价值 × margin_ratio
    - 现金约束: 开仓需保证金 ≤ 可用现金 (支持杠杆: 现金可 < 名义价值)

逐笔成交记录 TradeRecord, 支持换月 (平旧仓 + 开新仓).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cta_contract import CtaContract, CtaContractRegistry


@dataclass(slots=True)
class Position:
    """单合约持仓."""

    symbol: str
    product: str
    qty: float = 0.0           # 手数 (正=多, 负=空)
    avg_price: float = 0.0     # 开仓均价
    realized_pnl: float = 0.0  # 累计已实现盈亏


@dataclass(slots=True)
class TradeRecord:
    """单笔成交."""

    date: str
    symbol: str
    product: str
    side: str                 # 'buy' / 'sell' / 'close_long' / 'close_short'
    qty: float                # 手数 (正数)
    price: float
    fee: float
    notional: float           # 名义价值
    margin: float             # 保证金占用


class CtaBroker:
    """期货经纪商 (逐日结算).

    Args:
        registry: 合约规格注册表 (乘数/保证金)
        initial_cash: 初始资金
        commission_rate: 手续费比例 (default 0.0001)
        allow_short: 允许做空 (CTA 默认 True)
        margin_call: 保证金不足时拒绝开仓
    """

    def __init__(
        self,
        registry: CtaContractRegistry,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0001,
        allow_short: bool = True,
        margin_call: bool = False,
    ):
        self._registry = registry
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._commission_rate = commission_rate
        self._allow_short = allow_short
        self._margin_call = margin_call

        self._positions: dict[str, Position] = {}
        self._trades: list[TradeRecord] = []
        self._equity_history: list[float] = []

    # ── 查询 ────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    @property
    def trades(self) -> list[TradeRecord]:
        return self._trades

    def notional_value(self, symbol: str) -> float:
        """单合约持仓名义价值 (正负随方向)."""
        pos = self._positions.get(symbol)
        if pos is None or pos.qty == 0:
            return 0.0
        spec = self._registry.get_or_default(pos.product)
        return pos.qty * pos.avg_price * spec.contract_multiplier

    def margin_used(self) -> float:
        """全部持仓占用保证金 (按最新价, 简化: 用持仓均价)."""
        total = 0.0
        for pos in self._positions.values():
            if pos.qty == 0:
                continue
            spec = self._registry.get_or_default(pos.product)
            total += abs(pos.qty) * pos.avg_price * spec.contract_multiplier * spec.margin_ratio
        return total

    def equity(self, mark_prices: Optional[dict[str, float]] = None) -> float:
        """账户权益 = 现金 + 保证金占用 + 浮动盈亏.

        说明: place_order 开仓时已将保证金从现金划出, 但保证金仍是账户资产,
        故 equity = cash + margin_used + floating (相对当前市价).
        """
        # 按当前市价计算浮动
        cash = self._cash
        floating = 0.0
        margin = 0.0
        for pos in self._positions.values():
            if pos.qty == 0:
                continue
            spec = self._registry.get_or_default(pos.product)
            price = mark_prices.get(pos.symbol, pos.avg_price) if mark_prices else pos.avg_price
            floating += pos.qty * (price - pos.avg_price) * spec.contract_multiplier
            margin += abs(pos.qty) * price * spec.contract_multiplier * spec.margin_ratio
        return cash + margin + floating

    # ── 交易 ────────────────────────────────────────────────────────────

    def place_order(
        self,
        date: str,
        symbol: str,
        product: str,
        target_qty: float,
        price: float,
        contract: Optional[CtaContract] = None,
    ) -> bool:
        """执行目标手数调仓 (当前 qty → target_qty).

        Args:
            date: 交易日
            symbol: 合约代码 (e.g. 'IF2608')
            product: 产品代码 (e.g. 'IF')
            target_qty: 目标持仓手数 (可为 0 平仓)
            price: 成交价
            contract: 可选合约规格 (用于校验)

        Returns:
            bool 是否全部成交
        """
        cur = self._positions.get(symbol)
        cur_qty = cur.qty if cur else 0.0
        if abs(target_qty - cur_qty) < 1e-9:
            return True

        spec = self._registry.get_or_default(product)
        mult = spec.contract_multiplier
        margin = spec.margin_ratio
        fee_rate = self._commission_rate

        # ── 第 1 步: 平仓 (翻转平旧仓, 或同向减仓) ──────────────────────────
        if cur is not None and abs(cur_qty) > 1e-9:
            reversing = cur_qty * target_qty < 0
            reducing = not reversing and abs(target_qty) < abs(cur_qty)
            if reversing or reducing:
                close_qty = abs(cur_qty) if reversing else abs(cur_qty - target_qty)
                if close_qty > 1e-9:
                    fee_c = close_qty * price * mult * fee_rate
                    sign = 1 if cur_qty > 0 else -1
                    realized = close_qty * (price - cur.avg_price) * mult * sign
                    released_margin = close_qty * cur.avg_price * mult * margin
                    cur.realized_pnl += realized
                    self._cash += realized - fee_c + released_margin
                    self._record(date, symbol, product, "close", close_qty, price, fee_c)
                    cur.qty = cur_qty - close_qty * (1 if cur_qty > 0 else -1)
                    if abs(cur.qty) < 1e-9:
                        self._positions.pop(symbol, None)

        # ── 第 2 步: 开仓 (加仓/反手/新开) ──────────────────────────────────
        cur = self._positions.get(symbol)  # 平仓后可能已变
        cur_qty = cur.qty if cur else 0.0
        if abs(target_qty - cur_qty) < 1e-9:
            return True  # 已完全平仓 (target=0)

        open_qty = abs(target_qty - cur_qty)
        if not self._allow_short and target_qty < 0:
            return False

        # 保证金检查
        margin_req = abs(target_qty) * price * mult * margin
        if margin_req > self._cash + 1e-6 and self._margin_call:
            return False

        # 新开仓均价 (加仓取加权, 新开/反手用现价)
        if cur is not None and cur_qty * target_qty > 0:
            new_avg = (abs(cur_qty) * cur.avg_price + open_qty * price) / abs(target_qty)
        else:
            new_avg = price

        fee = open_qty * price * mult * fee_rate
        margin_deposit = open_qty * price * mult * margin
        self._cash -= margin_deposit + fee

        self._positions[symbol] = Position(
            symbol=symbol, product=product, qty=target_qty, avg_price=new_avg,
        )
        self._record(date, symbol, product, "open", open_qty, price, fee)
        return True

    def mark_to_market(self, mark_prices: dict[str, float], date: str) -> None:
        """日终按市价结算: 更新保证金占用与浮动盈亏.

        简化: 更新持仓均价到市价 (已实现/浮动合一), 现金调整浮动盈亏.
        """
        for pos in self._positions.values():
            price = mark_prices.get(pos.symbol)
            if price is None or price <= 0:
                continue
            spec = self._registry.get_or_default(pos.product)
            unrealized = pos.qty * (price - pos.avg_price) * spec.contract_multiplier
            # 浮动盈亏计入现金 (简化逐日结算), 均价重置为市价
            self._cash += unrealized
            pos.avg_price = price
        self._equity_history.append(self.equity())

    def _record(self, date, symbol, product, side, qty, price, fee):
        spec = self._registry.get_or_default(product)
        self._trades.append(TradeRecord(
            date=date, symbol=symbol, product=product, side=side, qty=abs(qty),
            price=price, fee=fee,
            notional=abs(qty) * price * spec.contract_multiplier,
            margin=abs(qty) * price * spec.contract_multiplier * spec.margin_ratio,
        ))

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
    """单合约持仓 (期货/期权)."""

    symbol: str
    product: str
    qty: float = 0.0           # 手数 (正=多, 负=空; 期权买方 qty≥0)
    avg_price: float = 0.0     # 开仓均价 (期货: 合约价; 期权: 权利金)
    realized_pnl: float = 0.0  # 累计已实现盈亏
    kind: str = "future"       # "future" | "option_buy" | "option_sell"
    option_spec: object = None # 期权时填入 (OptionContractSpec), 期货 None


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
        """全部持仓占用保证金 (按最新价, 简化: 用持仓均价).

        期权买方无保证金; 卖方按简化模型计算.
        """
        total = 0.0
        for pos in self._positions.values():
            if pos.qty == 0:
                continue
            if pos.kind == "option_buy":
                continue
            if pos.kind == "option_sell" and pos.option_spec is not None:
                total += abs(pos.qty) * pos.avg_price * pos.option_spec.contract_multiplier * 0.15
                continue
            spec = self._registry.get_or_default(pos.product)
            total += abs(pos.qty) * pos.avg_price * spec.contract_multiplier * spec.margin_ratio
        return total

    def equity(self, mark_prices: Optional[dict[str, float]] = None) -> float:
        """账户权益 = 现金 + 保证金占用 + 浮动盈亏.

        说明: place_order 开仓时已将保证金从现金划出, 但保证金仍是账户资产,
        故 equity = cash + margin_used + floating (相对当前市价).
        """
        cash = self._cash
        floating = 0.0
        margin = 0.0
        for pos in self._positions.values():
            if pos.qty == 0:
                continue
            price = mark_prices.get(pos.symbol, pos.avg_price) if mark_prices else pos.avg_price
            if pos.kind == "option_buy" and pos.option_spec is not None:
                mult = pos.option_spec.contract_multiplier
                floating += pos.qty * (price - pos.avg_price) * mult
                continue  # 买方无保证金
            if pos.kind == "option_sell" and pos.option_spec is not None:
                mult = pos.option_spec.contract_multiplier
                # 卖方浮动: 卖价 − 现价 (正 = 赚)
                floating += (pos.avg_price - price) * abs(pos.qty) * mult
                # 卖方保证金 (简化: strike × mult × qty × margin_ratio)
                margin += abs(pos.qty) * pos.avg_price * mult * 0.15
                continue
            spec = self._registry.get_or_default(pos.product)
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
        """执行目标手数调仓 (当前 qty → target_qty). 期货/期权统一入口.

        期权合约通过 contract.is_option=True 走买方分支:
            开仓扣权利金 (qty × price × multiplier), 无保证金;
            平仓按当前权利金结算已实现盈亏.

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
        is_option = contract is not None and contract.is_option

        # ── 期权分支 ────────────────────────────────────────────────────────
        if is_option:
            if target_qty >= 0:
                return self._place_option_buy(
                    date, symbol, product, target_qty, price, contract, fee_rate,
                )
            else:
                return self._place_option_sell(
                    date, symbol, product, target_qty, price, contract, fee_rate,
                )

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
            if pos.kind == "option_buy" and pos.option_spec is not None:
                mult = pos.option_spec.contract_multiplier
                unrealized = pos.qty * (price - pos.avg_price) * mult
                self._cash += unrealized
                pos.avg_price = price
                continue
            if pos.kind == "option_sell" and pos.option_spec is not None:
                mult = pos.option_spec.contract_multiplier
                # 卖方: 价格下降赚钱, 价格上升亏钱
                unrealized = (pos.avg_price - price) * abs(pos.qty) * mult
                self._cash += unrealized
                pos.avg_price = price
                continue
            spec = self._registry.get_or_default(pos.product)
            unrealized = pos.qty * (price - pos.avg_price) * spec.contract_multiplier
            # 浮动盈亏计入现金 (简化逐日结算), 均价重置为市价
            self._cash += unrealized
            pos.avg_price = price
        self._equity_history.append(self.equity())

    def _record(self, date, symbol, product, side, qty, price, fee, mult=None):
        spec = self._registry.get_or_default(product)
        m = mult if mult else spec.contract_multiplier
        self._trades.append(TradeRecord(
            date=date, symbol=symbol, product=product, side=side, qty=abs(qty),
            price=price, fee=fee,
            notional=abs(qty) * price * m,
            margin=abs(qty) * price * spec.contract_multiplier * spec.margin_ratio,
        ))

    # ── 期权买方逻辑 ──────────────────────────────────────────────────

    def _place_option_buy(
        self,
        date: str,
        symbol: str,
        product: str,
        target_qty: float,
        price: float,
        contract: Optional[CtaContract],
        fee_rate: float,
    ) -> bool:
        """期权买方调仓 (qty ≥ 0).

        开仓: 扣权利金 = qty × price × 乘数 (一次支出, 无保证金).
        平仓: 按当前权利金结算已实现盈亏.
        """
        cur = self._positions.get(symbol)
        cur_qty = cur.qty if cur else 0.0
        if target_qty < 0:
            return False  # MVP 买方 only, 不卖期权
        opt_spec = contract.option_spec
        mult = opt_spec.contract_multiplier if opt_spec else 1.0

        # 平仓 (target < cur)
        if target_qty < cur_qty and cur is not None and cur_qty > 0:
            close_qty = cur_qty - target_qty
            premium_in = close_qty * price * mult
            # 已实现盈亏 = (当前权利金 - 开仓权利金) × 手数 × 乘数
            realized = close_qty * (price - cur.avg_price) * mult
            fee_c = close_qty * price * mult * fee_rate
            cur.realized_pnl += realized
            self._cash += premium_in - fee_c
            self._record(date, symbol, product, "close", close_qty, price, fee_c, mult=mult)
            cur.qty = target_qty
            if abs(cur.qty) < 1e-9:
                self._positions.pop(symbol, None)
            return True

        # 开仓 (target > cur)
        if target_qty > cur_qty:
            open_qty = target_qty - cur_qty
            fee = open_qty * price * mult * fee_rate
            premium_out = open_qty * price * mult
            if premium_out + fee > self._cash + 1e-6 and self._margin_call:
                return False
            self._cash -= premium_out + fee
            if cur is not None and cur_qty > 0:
                new_avg = (cur_qty * cur.avg_price + open_qty * price) / target_qty
            else:
                new_avg = price
            self._positions[symbol] = Position(
                symbol=symbol, product=product, qty=target_qty, avg_price=new_avg,
                kind="option_buy", option_spec=opt_spec,
            )
            self._record(date, symbol, product, "open", open_qty, price, fee, mult=mult)
            return True
        return True

    def _place_option_sell(
        self,
        date: str,
        symbol: str,
        product: str,
        target_qty: float,
        price: float,
        contract: Optional[CtaContract],
        fee_rate: float,
    ) -> bool:
        """期权卖方调仓 (qty ≤ 0).

        开仓 (卖出): 收权利金, 扣保证金.
        平仓 (买回): 付权利金, 释放保证金.
        Put Spread 保证金 = spread_width × mult × qty (defined risk).
        """
        cur = self._positions.get(symbol)
        cur_qty = cur.qty if cur else 0.0
        opt_spec = contract.option_spec
        mult = opt_spec.contract_multiplier if opt_spec else 1.0
        fee_rate_s = fee_rate  # 手续费率

        # ── 平仓 (买回, target_qty > cur_qty, 即从更负向 0 移动) ──
        if cur is not None and cur_qty < 0 and target_qty > cur_qty:
            close_qty = abs(cur_qty - target_qty)  # 买回手数
            fee_c = close_qty * price * mult * fee_rate_s
            # 已实现盈亏 = (卖价 − 买回价) × 手数 × 乘数
            realized = close_qty * (cur.avg_price - price) * mult
            premium_out = close_qty * price * mult
            cur.realized_pnl += realized
            self._cash -= premium_out + fee_c
            self._record(date, symbol, product, "close", close_qty, price, fee_c, mult=mult)
            cur.qty = cur_qty + close_qty
            if abs(cur.qty) < 1e-9:
                self._positions.pop(symbol, None)
            return True

        # ── 开仓 (卖出, target_qty < cur_qty, 即向更负方向移动) ──
        if target_qty < cur_qty:
            open_qty = abs(cur_qty - target_qty)  # 卖出手数
            fee = open_qty * price * mult * fee_rate_s
            premium_in = open_qty * price * mult  # 权利金收入
            # 保证金 (简化: 0.15 × 不是名义)
            margin_dep = open_qty * price * mult * 0.15
            if margin_dep > self._cash + 1e-6 and self._margin_call:
                return False
            self._cash += premium_in - fee - margin_dep
            # 加权均价
            if cur is not None and cur_qty < 0:
                new_avg = (abs(cur_qty) * cur.avg_price + open_qty * price) / abs(target_qty)
            else:
                new_avg = price
            self._positions[symbol] = Position(
                symbol=symbol, product=product, qty=target_qty, avg_price=new_avg,
                kind="option_sell", option_spec=opt_spec,
            )
            self._record(date, symbol, product, "open", open_qty, price, fee, mult=mult)
            return True
        return True

    def _expire_options(self, date: str, settle_prices: dict[str, float]) -> None:
        """到期结算: 期权到期日按内在价值结算.

        settle_prices: {option_symbol: 结算价} (内在价值):
            Call: max(S - K, 0);  Put: max(K - S, 0)
        买方: 收取结算价; 卖方: 支付结算价.
        """
        for symbol, pos in list(self._positions.items()):
            if pos.kind not in ("option_buy", "option_sell"):
                continue
            spec = pos.option_spec
            if spec is None or spec.expiry_date != date:
                continue
            settle = settle_prices.get(symbol)
            if settle is None:
                continue
            mult = spec.contract_multiplier
            if pos.kind == "option_buy":
                realized = pos.qty * (settle - pos.avg_price) * mult
                self._cash += pos.qty * settle * mult
            else:
                # 卖方: 支付内在价值 (settle_per_unit = 内在价值/乘数, 已含在 settle 中)
                realized = (pos.avg_price * abs(pos.qty) - settle * abs(pos.qty)) * mult
                self._cash -= settle * abs(pos.qty) * mult
            pos.realized_pnl += realized
            self._record(date, symbol, pos.product, "expire", abs(pos.qty), settle, 0.0,
                         mult=mult)
            self._positions.pop(symbol)

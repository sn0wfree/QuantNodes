# coding=utf-8
"""CTA 逐日回测引擎.

核心: 逐日循环, 每个交易日:
    1. 计算信号 (策略回调)
    2. 生成目标持仓
    3. 经纪人执行调仓 (含保证金/手续费)
    4. 日终按市价结算 (mark-to-market)
    5. 累计 NAV

复用 momentum_etf_rotation.common.metrics 计算指标.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .cta_broker import CtaBroker
from .cta_contract import CtaContractRegistry
from .cta_risk import CtaRisk


@dataclass(slots=True)
class CtaBacktestConfig:
    """CTA 回测配置."""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0001
    allow_short: bool = True
    margin_call: bool = False     # True: 保证金不足拒绝开仓; False: 允许超配
    slippage_engine: object = None  # SlippageEngine 实例, None=无滑点


@dataclass
class CtaBacktestResult:
    """CTA 回测结果."""

    nav: pd.Series                    # 日频 NAV
    equity: list[float]
    cash: list[float]
    dates: list[pd.Timestamp]
    positions: dict[str, list[float]]  # {symbol: [qty]} 逐日
    trades: list                        # TradeRecord 列表
    metrics: dict = field(default_factory=dict)


class CtaStrategy:
    """CTA 策略回调基类.

    策略子类实现 target_positions(), 返回每个交易日每个合约的目标持仓.
    """

    def target_positions(self, date, data: dict) -> dict[str, float]:
        """给定日期返回 {symbol: target_qty}.

        data: 预加载的行情面板 (由调用方构造)
        """
        raise NotImplementedError


def run_cta_backtest(
    dates: list[pd.Timestamp],
    prices: dict[str, list[float]],       # {symbol: [price] × len(dates)}
    products: dict[str, str],              # {symbol: product}
    strategy: CtaStrategy,
    config: CtaBacktestConfig | None = None,
    registry: CtaContractRegistry | None = None,
    risk: CtaRisk | None = None,
    option_contracts: dict[str, object] | None = None,
    option_settle: dict[str, dict[str, float]] | None = None,
) -> CtaBacktestResult:
    """运行 CTA 回测.

    Args:
        dates: 交易日序列
        prices: {symbol: 价格序列}  (symbol → 每日价格, 与 dates 等长)
        products: {symbol: product}
        strategy: 策略回调 (target_positions)
        config: 回测配置
        registry: 合约规格
        risk: 风控 (可选, 拦截目标持仓)
        option_contracts: {option_symbol: CtaContract} 期权合约 (place_order 传 contract)
        option_settle: {option_symbol: {date: 结算价}} 到期结算价 (内在价值)
    """
    config = config or CtaBacktestConfig()
    registry = registry or CtaContractRegistry()
    broker = CtaBroker(
        registry=registry,
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        allow_short=config.allow_short,
        margin_call=config.margin_call,
        slippage_engine=config.slippage_engine,
    )

    n = len(dates)
    nav = [1.0] * n
    equity_hist: list[float] = []
    cash_hist: list[float] = []
    pos_hist: dict[str, list[float]] = {s: [0.0] * n for s in prices}

    for i, date in enumerate(dates):
        mark_prices = {s: p[i] for s, p in prices.items() if p[i] == p[i]}
        date_str = str(date.date())

        # 1. 信号 → 目标持仓
        targets = strategy.target_positions(date, {"prices": prices, "dates": dates})

        # 2. 风控
        if risk is not None:
            targets = risk.apply(targets, mark_prices, i, dates, prices)

        # 3. 执行调仓 (逐 symbol)
        for symbol, target_qty in targets.items():
            price = mark_prices.get(symbol)
            if price is None or price <= 0:
                continue
            contract = None
            if option_contracts and symbol in option_contracts:
                contract = option_contracts[symbol]
            broker.place_order(
                date=date_str, symbol=symbol,
                product=products[symbol], target_qty=target_qty, price=price,
                contract=contract,
            )

        # 3.5 期权到期结算 (买方按内在价值)
        if option_settle:
            settle_prices = {
                sym: daily.get(date_str, 0.0)
                for sym, daily in option_settle.items()
            }
            broker._expire_options(date_str, settle_prices)

        # 4. 日终结算
        broker.mark_to_market(mark_prices, date_str)
        equity_hist.append(broker.equity())
        cash_hist.append(broker.cash)
        for s in pos_hist:
            pos = broker.positions.get(s)
            pos_hist[s][i] = pos.qty if pos else 0.0

        nav[i] = equity_hist[-1] / config.initial_cash

    nav_series = pd.Series(nav, index=pd.DatetimeIndex(dates), name="nav")

    # 指标
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics
    metrics = compute_metrics(nav_series)

    return CtaBacktestResult(
        nav=nav_series,
        equity=equity_hist,
        cash=cash_hist,
        dates=dates,
        positions=pos_hist,
        trades=broker.trades,
        metrics=metrics,
    )

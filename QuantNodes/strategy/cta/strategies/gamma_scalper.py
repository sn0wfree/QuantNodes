# coding=utf-8
"""Gamma Scalper: 买入 ATM Straddle + 每日 Delta 对冲.

策略流程:
    1. 入场: 买入 ATM Straddle (call + put, 同 strike 同到期)
    2. 每日: 计算 portfolio delta → 用 futures 对冲至 delta-neutral
    3. 展期: 到期前 roll_days 天平旧仓 + 开新仓
    4. P&L 分解: theta 衰减 + gamma 利润 + delta 对冲 P&L

数据依赖:
    - 期权 OHLCV (v_all_options)
    - 标的期货收盘价 (continuous_main)
    - ATM IV 面板 (option_iv_daily)

输入: 品种列表 + 回测日期范围
输出: CtaBacktestResult (NAV/指标/P&L 分解)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_contract import CtaContractRegistry
from ..options.selector import select_straddle, select_option_for_delta, StraddleSelection
from ..options.delta_hedge import DeltaHedgeManager
from ..options.pnl_decompose import PnlDecomposer
from ..options.roll_manager import RollManager
from ..options.slippage import estimate_option_spread, apply_slippage
from ..options.underlying_map import UnderlyingMap
from ..options.expiry import OptionExpiry
from ..options.greeks import compute_greeks_for_option, days_to_expiry


@dataclass
class GammaScalperConfig:
    """Gamma Scalper 配置."""

    products: list[str] = field(default_factory=lambda: ["RB"])
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0001
    margin_call: bool = False

    # straddle 参数
    min_days: int = 30
    max_days: int = 60
    roll_days: int = 7       # 到期前 N 天展期

    # delta 对冲
    hedge_threshold: float = 0.3   # |delta| > threshold × multiplier 时调仓
    delta_hedge_mode: str = "option"  # "option" | "futures" (预留)
    delta_adj_side: str = "buy"     # "buy" = 买方调, "sell" = 卖方调


@dataclass
class StraddleHolding:
    """当前持有的 straddle 状态."""

    product: str
    call_symbol: str
    put_symbol: str
    strike: float
    expiry: pd.Timestamp
    cost: float           # 建仓成本 (权利金)
    entry_date: str

    # Greeks (每日更新)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0


class GammaScalper(CtaStrategy):
    """Gamma Scalping CTA 策略.

    使用方式:
        cfg = GammaScalperConfig(products=["RB"], min_days=30, max_days=60)
        scalper = GammaScalper(cfg)
        result = scalper.run()
    """

    def __init__(self, config: GammaScalperConfig | None = None):
        self._cfg = config or GammaScalperConfig()
        self._loader = CtaDataLoader()
        self._registry = CtaContractRegistry()
        self._um = UnderlyingMap(self._loader)
        self._oe = OptionExpiry(self._loader)

        # 状态
        self._holdings: dict[str, StraddleHolding] = {}  # product → holding
        self._delta_mgr = DeltaHedgeManager(self._registry)
        self._pnl = PnlDecomposer()
        self._roll_mgr = RollManager(roll_days=self._cfg.roll_days)

        # 预加载数据
        self._option_data: dict[str, pd.DataFrame] = {}  # product → option df
        self._price_data: dict[str, pd.Series] = {}      # product → underlying price series
        self._iv_data: pd.DataFrame | None = None

    # ── 数据加载 ──────────────────────────────────────────────────────────

    def load_data(self, start: str, end: str):
        """预加载期权行情 + 标的价格 + IV.

        Args:
            start: 回测开始日 'YYYY-MM-DD'
            end: 回测结束日 'YYYY-MM-DD'
        """
        # 期权 OHLCV (保留原始 case, 与 select_straddle 一致)
        for prod in self._cfg.products:
            df = self._loader.load_v_all_options(
                products=[prod], start=start, end=end,
            )
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                self._option_data[prod] = df

        # 标的期货价格 (主力连续)
        for prod in self._cfg.products:
            try:
                px = self._loader.load_continuous_main(prod)
                if px is not None and not px.empty:
                    px["trade_date"] = pd.to_datetime(px["trade_date"])
                    px = px.set_index("trade_date")
                    self._price_data[prod] = px["close"]
            except Exception:
                pass

        # IV 面板
        try:
            self._iv_data = load_iv_panel(start=start, end=end)
        except Exception:
            self._iv_data = None

    # ── 期权数据查询 ──────────────────────────────────────────────────────

    def _get_option_day(self, product: str, date: pd.Timestamp) -> pd.DataFrame:
        """获取某品种某日的期权行情."""
        df = self._option_data.get(product)
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df["trade_date"] == date]

    def _get_underlying_price(self, product: str, date: pd.Timestamp) -> float | None:
        """获取标的期货收盘价."""
        series = self._price_data.get(product)
        if series is None or series.empty:
            return None
        if date in series.index:
            return float(series.loc[date])
        # 找最近的交易日
        idx = series.index.get_indexer([date], method="ffill")
        if idx[0] >= 0:
            return float(series.iloc[idx[0]])
        return None

    def _get_iv(self, product: str, date: pd.Timestamp) -> float | None:
        """获取 ATM IV."""
        if self._iv_data is None or self._iv_data.empty:
            return None
        if date in self._iv_data.index and product in self._iv_data.columns:
            v = self._iv_data.loc[date, product]
            return float(v) if pd.notna(v) and v > 0 else None
        return None

    def _compute_straddle_greeks(
        self, product: str, date: pd.Timestamp,
        holding: StraddleHolding,
    ) -> None:
        """更新 straddle 的 Greeks (用当日市价反解 IV)."""
        spot = self._get_underlying_price(product, date)
        if spot is None or spot <= 0:
            return

        ttm = days_to_expiry(date, holding.expiry)
        if ttm <= 0:
            return

        # 获取当日两腿市价
        day_df = self._get_option_day(product, date)
        if day_df.empty:
            return

        call_row = day_df[day_df["symbol"] == holding.call_symbol]
        put_row = day_df[day_df["symbol"] == holding.put_symbol]
        if call_row.empty or put_row.empty:
            return

        call_price = float(call_row.iloc[0]["close"])
        put_price = float(put_row.iloc[0]["close"])
        if call_price <= 0 or put_price <= 0:
            return

        g_call = compute_greeks_for_option(
            symbol=holding.call_symbol, option_type="C",
            strike_price=holding.strike, underlying_price=spot,
            market_price=call_price, time_to_maturity=ttm,
            risk_free_rate=0.02,
        )
        g_put = compute_greeks_for_option(
            symbol=holding.put_symbol, option_type="P",
            strike_price=holding.strike, underlying_price=spot,
            market_price=put_price, time_to_maturity=ttm,
            risk_free_rate=0.02,
        )

        if g_call.valid and g_put.valid:
            holding.delta = g_call.delta + g_put.delta
            holding.gamma = g_call.gamma + g_put.gamma
            holding.theta = g_call.theta + g_put.theta
            holding.vega = g_call.vega + g_put.vega
            holding.iv = (g_call.iv + g_put.iv) / 2.0

    # ── 核心: target_positions ────────────────────────────────────────────

    def target_positions(self, date, data: dict) -> dict[str, float]:
        """每日调仓: 展期检查 → straddle 入场 → delta 对冲.

        Returns:
            {symbol: target_qty} — 包含期货 (delta hedge) + 期权 (straddle 腿)
        """
        targets: dict[str, float] = {}
        date_str = str(date.date())

        for product in self._cfg.products:
            spot = self._get_underlying_price(product, date)
            if spot is None or spot <= 0:
                continue

            # 1. 更新现有 straddle Greeks
            if product in self._holdings:
                self._compute_straddle_greeks(product, date, self._holdings[product])

            # 2. 展期检查
            holding = self._holdings.get(product)
            if holding is not None:
                days_left = (holding.expiry - date).days
                if days_left <= self._cfg.roll_days:
                    # 平旧仓
                    targets[holding.call_symbol] = 0.0
                    targets[holding.put_symbol] = 0.0
                    self._holdings.pop(product, None)
                    holding = None

            # 3. 入场: 无持仓时买新 straddle
            if holding is None:
                straddle = select_straddle(
                    product, date, spot,
                    min_days=self._cfg.min_days,
                    max_days=self._cfg.max_days,
                    loader=self._loader, um=self._um, oe=self._oe,
                )
                if straddle is not None:
                    self._holdings[product] = StraddleHolding(
                        product=product,
                        call_symbol=straddle.call.symbol,
                        put_symbol=straddle.put.symbol,
                        strike=straddle.strike,
                        expiry=pd.Timestamp(straddle.expiry),
                        cost=straddle.cost,
                        entry_date=date_str,
                        delta=straddle.delta,
                        gamma=straddle.gamma,
                        theta=straddle.theta,
                        vega=straddle.vega,
                    )
                    targets[straddle.call.symbol] = 1.0
                    targets[straddle.put.symbol] = 1.0

            # 4. Delta 对冲 (用期权调整)
            holding = self._holdings.get(product)
            if holding is not None:
                spec = self._registry.get_or_default(product)
                mult = spec.contract_multiplier
                total_delta = holding.delta * 1.0 * mult  # 1 lot

                # |delta| 超过阈值 → 选调整期权
                if abs(total_delta) > self._cfg.hedge_threshold * mult:
                    adj_type, adj_qty, _ = self._delta_mgr.compute_option_delta_adjustment(
                        current_delta=total_delta,
                        target_delta=0.0,
                        product=product,
                        side=self._cfg.delta_adj_side,
                    )
                    if adj_qty > 0 and adj_type:
                        adj = select_option_for_delta(
                            product, date, spot, adj_type,
                            loader=self._loader, um=self._um, oe=self._oe,
                        )
                        if adj is not None:
                            targets[adj.symbol] = adj_qty

        return targets

    # ── 结果 ──────────────────────────────────────────────────────────────

    def run(self, start: str, end: str) -> CtaBacktestResult:
        """运行完整回测.

        Args:
            start: 回测开始日 'YYYY-MM-DD'
            end: 回测结束日 'YYYY-MM-DD'

        Returns:
            CtaBacktestResult (含 NAV/指标)
        """
        self.load_data(start, end)

        # 构造价格面板 (标的期货)
        price_panel = pd.DataFrame(self._price_data)
        price_panel = price_panel.dropna(how="all")
        if price_panel.empty:
            raise ValueError("无可用价格数据")

        dates = list(price_panel.index)
        prices_map = {p: price_panel[p].tolist() for p in price_panel.columns}
        products_map = {p: p for p in price_panel.columns}

        # 构造 option_contracts + option 价格 (保留原始 case, 与 select_straddle 一致)
        option_contracts = {}
        for prod, df in self._option_data.items():
            if df.empty:
                continue
            for sym in df["symbol"].unique():
                sym_df = df[df["symbol"] == sym]
                if sym_df.empty:
                    continue
                row = sym_df.iloc[0]
                opt_type = str(row.get("option_type", "")).upper()
                strike = float(row["strike_price"])
                underlying = str(row.get("underlying", ""))

                from ..backtest.cta_contract import CtaContract, OptionContractSpec
                opt_spec = OptionContractSpec(
                    exchange="", product=prod,
                    underlying=underlying,
                    option_type=opt_type,
                    strike=strike,
                    expiry_date="",
                    contract_multiplier=self._registry.get_or_default(prod).contract_multiplier,
                )
                fut_spec = self._registry.get_or_default(prod)
                contract = CtaContract(
                    product=prod, symbol=sym,
                    expiry_year=0, expiry_month=0,
                    spec=fut_spec, option_spec=opt_spec,
                )
                option_contracts[sym] = contract

                # option price series (对齐 dates)
                sym_ts = sym_df.set_index("trade_date")
                sym_prices = sym_ts["close"].reindex(dates)
                prices_map[sym] = sym_prices.tolist()
                products_map[sym] = prod

        return run_cta_backtest(
            dates=dates,
            prices=prices_map,
            products=products_map,
            strategy=self,
            config=CtaBacktestConfig(
                initial_cash=self._cfg.initial_cash,
                commission_rate=self._cfg.commission_rate,
                margin_call=self._cfg.margin_call,
            ),
            registry=self._registry,
            option_contracts=option_contracts,
        )

    def pnl_summary(self) -> dict:
        """返回 P&L 分解统计."""
        return self._pnl.summary()

    def hedge_records(self) -> pd.DataFrame:
        """返回 delta hedge 调仓记录."""
        return self._delta_mgr.get_records()

    def roll_records(self) -> pd.DataFrame:
        """返回展期记录."""
        return self._roll_mgr.get_records()

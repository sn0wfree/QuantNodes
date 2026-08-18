# coding=utf-8
"""Gamma-Vega 卖方策略: 大跌后裸卖 OTM put + delta 管理 (docs/108 §5).

核心逻辑:
    触发: 连续4日跌 + VRP分位>0.5 + 品种∈good_products
    开仓: 一次性裸卖 OTM put (|delta|<0.3), 享受高 IV + 恐慌溢价
    delta 管控: delta 市值超15%时, 用买权降敞口 → 逐步变 spread
    持有: 持有到期 (卖权吃时间价值), 但加止损控左尾

与之前方案的区别:
    - 单次开仓 (非分步)
    - 裸卖 OTM put (非 spread), 让 delta 随行情自然管理
    - 超标后用买权补 → 动态变为 spread
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy


@dataclass
class GammaVegaConfig:
    """Gamma-Vega 卖方配置 (裸卖 OTM put + delta 管理)."""

    # 触发
    down_days: int = 3                # 连续下跌日数 (与之前一致)
    vrp_min: float = 0.50             # VRP 分位阈值
    good_products: list[str] = field(
        default_factory=lambda: ["I", "AG", "CU", "SR", "TA", "RU", "M"])

    # 卖 put 参数
    short_put_strike_mult: float = 0.95  # OTM put strike 倍数 (0.95×spot = delta≈0.3)
    min_dte: int = 15
    max_dte: int = 40
    target_delta_pct: float = 0.30    # 卖方 delta 市值上限 = NAV × 30%

    # 买权 delta 管理
    hedge_call_strike_mult: float = 1.00  # 买权 hedge strike 倍数 (ATM)
    hedge_trigger_pct: float = 0.10   # delta 超标阈值: 当前delta市值/NAV > 10% → 买权降敞口
    hedge_reduction_pct: float = 0.30  # 每次买权降敞口 30% (买权 delta ≈ 0.5)

    # 持有 + 风控
    hold_to_expiry: bool = True       # 持有到期吃theta
    stop_loss_pct: float = 0.20       # 裸卖亏损 >20% → 止损 (优化后更紧)


class GammaVegaSeller(FactorModuleComboStrategy):
    """大跌后裸卖 OTM put + 动态 delta 管理 (叠加 fut_v1)."""

    def __init__(
        self,
        gv_config: GammaVegaConfig | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        option_contracts: dict | None = None,
        greeks_cache=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gv_cfg = gv_config or GammaVegaConfig()
        self._option_contracts = option_contracts or {}
        self._iv_panel: pd.DataFrame | None = None
        self._precomputed_lots: pd.DataFrame | None = None
        self._greeks_cache = greeks_cache
        self._vrp_pct: pd.DataFrame | None = None
        self._ret: pd.DataFrame | None = None
        self._down_cnt: pd.DataFrame | None = None
        self._chains: dict[str, pd.DataFrame] = {}
        for prod, df in (option_data or {}).items():
            if df is not None and not df.empty:
                self._chains[prod] = df.set_index("trade_date").sort_index()
        self._expiry_map: dict[str, pd.Timestamp] = {}
        self._build_expiry_map(option_data or {})
        self.stats = {"entries": 0, "hedges": 0, "exits": 0,
                      "stops": 0, "skips": 0}

    def _build_expiry_map(self, option_data):
        for prod, df in option_data.items():
            if df is None or df.empty:
                continue
            for sym, dt in df.groupby("symbol")["trade_date"].max().items():
                self._expiry_map[str(sym).upper()] = pd.Timestamp(dt)

    # ── 信号 ──────────────────────────────────────────────────────────

    def _load_iv_panel(self):
        if self._iv_panel is not None:
            return
        try:
            from .iv_panel import load_iv_panel
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            self._iv_panel = None

    def _compute_signal_panels(self):
        if self._vrp_pct is not None or self._iv_panel is None:
            return
        iv = self._iv_panel.reindex(index=self._price_panel.index,
                                    columns=self._price_panel.columns)
        ret = self._price_panel.pct_change()
        hv20 = ret.rolling(20).std() * np.sqrt(252)
        vrp = iv - hv20
        self._vrp_pct = vrp.rolling(252, min_periods=60).rank(pct=True)
        self._ret = ret
        self._down_cnt = (ret < 0).rolling(10).sum()

    # ── 期权链工具 ────────────────────────────────────────────────────

    def _day_rows(self, product, date):
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        d = chain.loc[date]
        return d.to_frame().T if isinstance(d, pd.Series) else d

    def _pick(self, product, date, spot, opt_type, sm):
        d = self._day_rows(product, date)
        if d is None:
            return None
        sub = d[d["option_type"].str.upper() == opt_type].copy()
        if sub.empty:
            return None
        liq = sub[(sub["volume"].fillna(0) > 0) | (sub["open_interest"].fillna(0) > 0)]
        if not liq.empty:
            sub = liq
        exp = sub["symbol"].str.upper().map(self._expiry_map)
        ttm = (exp - date).dt.days
        win = sub[(ttm >= self._gv_cfg.min_dte) & (ttm <= self._gv_cfg.max_dte)]
        if win.empty:
            return None
        dist = (win["strike_price"] - spot * sm).abs()
        row = win.iloc[int(dist.to_numpy().argmin())]
        sym = str(row["symbol"]).upper()
        if sym not in self._option_contracts:
            return None
        px = float(row["close"]) if pd.notna(row["close"]) else 0.0
        if px <= 0:
            return None
        return {"symbol": sym, "px": px, "strike": float(row["strike_price"]),
                "expiry": self._expiry_map.get(sym)}

    def _delta_of(self, product, date, pick):
        if self._greeks_cache is not None:
            rec = self._greeks_cache.lookup(product, pick["symbol"], date)
            if rec is not None and pd.notna(rec.get("delta")):
                return float(rec["delta"])
        return 0.0

    def _price_of(self, product, date, sym):
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        d = chain.loc[date]
        if isinstance(d, pd.Series):
            d = d.to_frame().T
        m = d["symbol"].str.upper() == sym.upper()
        if not m.any():
            return None
        v = d.loc[m].iloc[0]["close"]
        return float(v) if pd.notna(v) else None

    # ── 状态机 ────────────────────────────────────────────────────────

    def _precompute_gv_targets(self):
        cfg = self._gv_cfg
        dates = self._precomputed_lots.index
        targets_by_date = {}
        holdings = {}

        for i, date in enumerate(dates):
            targets = {}
            has_px = date in self._price_panel.index

            # 1. 到期/止损平仓
            for product in list(holdings.keys()):
                h = holdings[product]
                close = False
                if cfg.hold_to_expiry and h.get("expiry"):
                    if date >= pd.Timestamp(h["expiry"]):
                        close = True
                if not close and cfg.stop_loss_pct > 0:
                    cur = self._price_of(product, date, h["short_sym"])
                    if cur is not None:
                        loss = (cur - h["entry_premium"]) / h["entry_premium"]
                        if loss >= cfg.stop_loss_pct:
                            close = True
                            self.stats["stops"] += 1
                if close:
                    targets[h["short_sym"]] = 0.0
                    if h.get("hedge_sym"):
                        targets[h["hedge_sym"]] = 0.0
                    self.stats["exits"] += 1
                    holdings.pop(product)
                    continue

                # 2. delta 管理: 当前 delta 市值 > 阈值 → 买权降敞口
                spot = self._price_panel.loc[date, product] if has_px else np.nan
                if not np.isnan(spot) and spot > 0:
                    short_px = self._price_of(product, date, h["short_sym"])
                    if short_px is not None:
                        short_delta = abs(self._delta_of(product, date, {
                            "symbol": h["short_sym"], "px": short_px}))
                        if short_delta < 1e-6:
                            short_delta = 0.3
                        cur_delta_mkt = abs(short_delta * h["qty"] * spot *
                                            self._registry.get_or_default(product).contract_multiplier)
                        equity = getattr(self._config, "initial_cash", 5_000_000)
                        cur_delta_pct = cur_delta_mkt / equity if equity > 0 else 0

                        # 如果 delta 超标且未 hedge
                        if cur_delta_pct > cfg.hedge_trigger_pct and not h.get("hedge_sym"):
                            hedge = self._pick(product, date, spot, "C",
                                               cfg.hedge_call_strike_mult)
                            if hedge is not None:
                                hedge_delta = abs(self._delta_of(product, date, hedge))
                                if hedge_delta < 1e-6:
                                    hedge_delta = 0.5
                                hedge_qty = int(cur_delta_mkt * cfg.hedge_reduction_pct
                                                / (spot * self._registry.get_or_default(product)
                                                   .contract_multiplier * hedge_delta))
                                if hedge_qty > 0:
                                    targets[hedge["symbol"]] = float(hedge_qty)
                                    h["hedge_sym"] = hedge["symbol"]
                                    h["hedge_qty"] = hedge_qty
                                    h["hedge_entry"] = hedge["px"]
                                    self.stats["hedges"] += 1

            # 3. 新触发 (一次性开仓)
            if has_px and self._vrp_pct is not None:
                for product in self._products:
                    if product in holdings:
                        continue
                    if product not in cfg.good_products:
                        continue
                    if date not in self._vrp_pct.index:
                        continue
                    v = self._vrp_pct.loc[date, product]
                    if pd.isna(v) or v <= cfg.vrp_min:
                        continue
                    spot = self._price_panel.loc[date, product]
                    if np.isnan(spot) or spot <= 0:
                        continue
                    if self._chains.get(product) is None:
                        continue
                    dc = np.nan
                    if date in self._down_cnt.index:
                        dc = self._down_cnt.loc[date, product]
                    if pd.isna(dc) or dc < cfg.down_days:
                        self.stats["skips"] += 1
                        continue
                    # 选 OTM put (|delta| ~0.3)
                    put = self._pick(product, date, spot, "P",
                                     cfg.short_put_strike_mult)
                    if put is None:
                        continue
                    put_delta = abs(self._delta_of(product, date, put))
                    if put_delta < 1e-6:
                        put_delta = 0.3
                    spec = self._registry.get_or_default(product)
                    mult = spec.contract_multiplier
                    equity = getattr(self._config, "initial_cash", 5_000_000)
                    qty = int(equity * cfg.target_delta_pct /
                              (spot * mult * put_delta))
                    if qty < 1:
                        continue
                    targets[put["symbol"]] = -float(qty)
                    holdings[product] = {
                        "short_sym": put["symbol"], "short_px": put["px"],
                        "qty": qty, "expiry": put.get("expiry"),
                        "entry_premium": put["px"], "entry_i": i,
                        "hedge_sym": None, "hedge_qty": 0, "hedge_entry": 0,
                    }
                    self.stats["entries"] += 1

            targets_by_date[date] = targets

        self._precomputed_gv = targets_by_date
        return targets_by_date

    # ── 回测接口 ──────────────────────────────────────────────────────

    def target_positions(self, date, data):
        if self._precomputed_lots is None or date not in self._precomputed_lots.index:
            return {}
        out = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                out[product] = float(row[product])
        if hasattr(self, "_precomputed_router") and date in self._precomputed_router:
            out.update(self._precomputed_router[date])
        if hasattr(self, "_precomputed_gv") and date in self._precomputed_gv:
            out.update(self._precomputed_gv[date])
        return out

    def _run_with_strategy(self) -> 'CtaBacktestResult':
        """使用 target_positions() 策略运行回测 (包含期权目标)."""
        from ..backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
        from ..backtest.cta_risk import PositionLimitRisk
        from ..backtest.cta_slippage import SlippageEngine, SquareRootImpact
        from .slippage import build_option_prices

        cfg = self._config
        price_panel = self._price_panel

        # 构建滑点引擎
        slippage_engine = None
        if hasattr(cfg, 'slippage_bps') and cfg.slippage_bps is not None:
            model = SquareRootImpact(scale=cfg.slippage_bps)
            slippage_engine = SlippageEngine(
                model=model,
                roll_cost_bps=getattr(cfg, 'roll_cost_bps', 0.0),
                roll_days=getattr(cfg, 'roll_days', 5),
            )
            price_panels = {}
            for p in price_panel.columns:
                if p in self._ohlc_map and p in self._vols.columns:
                    ohlc = self._ohlc_map[p].copy()
                    ohlc["volume"] = self._vols[p]
                    price_panels[p] = ohlc
            slippage_engine.load(price_panels, list(price_panels.keys()))
            if slippage_engine.roll_cost_bps > 0:
                slippage_engine.load_roll_dates(list(price_panels.keys()), self._dl)

        # 构建价格映射: 期货 + 期权
        opt_prices, opt_settle = build_option_prices(
            list(price_panel.index), self._chains
        )
        all_prices = {p: price_panel[p].tolist() for p in price_panel.columns}
        all_prices.update(opt_prices)

        # 构建产品映射: 期货 + 期权
        all_products = {p: p for p in price_panel.columns}
        for sym, c in self._option_contracts.items():
            all_products[sym] = c.product

        # 使用 self.target_positions 作为策略回调
        strategy = _GVSellerStrategy(self)
        risk = PositionLimitRisk(
            products={p: p for p in price_panel.columns}, registry=self._registry
        )

        return run_cta_backtest(
            dates=list(price_panel.index),
            prices=all_prices,
            products=all_products,
            strategy=strategy,
            config=CtaBacktestConfig(
                initial_cash=cfg.initial_cash, commission_rate=cfg.commission_rate,
                margin_call=cfg.margin_call, slippage_engine=slippage_engine,
            ),
            registry=self._registry,
            risk=risk,
            option_contracts=self._option_contracts,
            option_settle=opt_settle,
        )

    def run(self):
        self.load_panel()
        self._load_iv_panel()
        self._compute_signal_panels()
        signals = self._signals()
        exec_sig = self._exec_sig(signals)
        lots = self._signal_to_lots(exec_sig, self._config)
        self._precomputed_lots = lots
        self._precompute_gv_targets()
        return self._run_with_strategy()


class _GVSellerStrategy:
    """包装 GammaVegaSeller.target_positions 为 CtaStrategy 回调."""

    def __init__(self, seller: GammaVegaSeller):
        self._seller = seller
        self._last = {}

    def target_positions(self, date, data):
        raw = self._seller.target_positions(date, data)
        # min_position_change 过滤
        filtered = {}
        for p in set(raw) | set(self._last):
            new_qty = raw.get(p, 0)
            old_qty = self._last.get(p, 0)
            def _sgn(x):
                return 1 if x > 0 else (-1 if x < 0 else 0)
            if _sgn(new_qty) == _sgn(old_qty):
                max_qty = max(abs(old_qty), abs(new_qty), 1)
                if abs(new_qty - old_qty) / max_qty < self._seller._config.min_position_change:
                    filtered[p] = old_qty
                    continue
            filtered[p] = new_qty
        self._last = filtered
        return filtered

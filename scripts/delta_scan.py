#!/usr/bin/env python3
"""delta_notional_pct 扫描 (组合模式, long7/short15)."""
import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, "/home/ll/Public/QuantNodes")
from dataclasses import replace
import numpy as np
from QuantNodes.strategy.cta.options.vrp_router import VrpRouterStrategy, VrpRouterConfig
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

PRODUCTS = ["RB","I","AG","AU","CU","M","TA","SR","RU","PP"]
cfg = ComboConfig(initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
    momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
    module_weighting="inv_vol", continuous=True, cont_z_window=252, cont_z_clip=3.0,
    margin_call=True, execution_lag=1, aggregation="tvpr")
od, oc = load_option_data(PRODUCTS, "2023-01-01", "2025-12-31")
gc = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
s0 = VrpRouterStrategy(router_config=VrpRouterConfig(), option_data=od, option_contracts=oc,
    greeks_cache=gc, products=PRODUCTS, config=cfg, start="2018-01-01", end="2025-12-31")
s0.load_panel()
sig = s0._signals(); es = s0._exec_sig(sig); lots = s0._signal_to_lots(es, cfg)
s0._precomputed_lots = lots; s0._load_iv_panel(); s0._compute_signal_panels()
op, os_ = build_option_prices(list(s0._price_panel.index), od)
ap = {p: s0._price_panel[p].tolist() for p in PRODUCTS}; ap.update(op)
ap2 = {p: p for p in PRODUCTS}
for sym, c in oc.items():
    ap2[sym] = c.product
reg = CtaContractRegistry()
risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=reg)

print(f"{'delta':>6} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>7} {'Calmar':>7} {'buy':>4} {'exit':>5}")
for dp in [0.15, 0.18, 0.20, 0.25, 0.30]:
    t = time.time()
    cfg2 = replace(VrpRouterConfig(), delta_notional_pct=dp, long_hold=7, short_hold=15)
    st = VrpRouterStrategy(router_config=cfg2, option_data=od, option_contracts=oc,
        greeks_cache=gc, products=PRODUCTS, config=cfg, start="2018-01-01", end="2025-12-31")
    st._price_panel = s0._price_panel
    st._ohlc_map = s0._ohlc_map; st._vols = s0._vols; st._ois = s0._ois
    st._iv_panel = s0._iv_panel; st._vrp_pct = s0._vrp_pct
    st._mom20 = s0._mom20; st._iv_pct = s0._iv_pct
    st._precomputed_lots = lots
    st._precompute_router_targets()
    res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap, products=ap2,
        strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000,
            commission_rate=0.0001, margin_call=True),
        registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
    r = res.nav.pct_change().dropna()
    a = (1 + res.nav.iloc[-1] / res.nav.iloc[0] - 1) ** (1 / max(len(r) / 252, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    dd = ((res.nav - res.nav.cummax()) / res.nav.cummax()).min()
    cal = a / abs(dd) if abs(dd) > 0 else 0
    sh = (a - 0.02) / vol if vol > 0 else 0
    s = st.stats
    print(f"{dp:>6.2f} {sh:>7.2f} {a:>8.1%} {dd:>7.1%} {cal:>7.2f} "
          f"{s['long_buys']+s['short_buys']:>4} {s['exits']:>5}  [{time.time()-t:.0f}s]",
          flush=True)

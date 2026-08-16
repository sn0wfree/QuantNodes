#!/usr/bin/env python3
"""最优参数最终验证 + OOS 2025 (long7/short15, delta 0.20)."""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np

from QuantNodes.strategy.cta.options.vrp_router import VrpRouterStrategy, VrpRouterConfig
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]


def metrics(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann / abs(dd) if abs(dd) > 0 else 0
    return {"AnnRet": ann, "Sharpe": sharpe, "MaxDD": dd, "Calmar": calmar}


def run_one(start, end, opt_data, opt_contracts, gcache, tag):
    cfg_base = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
    )
    s0 = VrpRouterStrategy(
        router_config=VrpRouterConfig(), option_data=opt_data,
        option_contracts=opt_contracts, greeks_cache=gcache, products=PRODUCTS,
        config=cfg_base, start=start, end=end,
    )
    s0.load_panel()
    sig = s0._signals()
    es = s0._exec_sig(sig)
    lots = s0._signal_to_lots(es, cfg_base)
    s0._precomputed_lots = lots
    s0._load_iv_panel()
    s0._compute_signal_panels()

    op, os_ = build_option_prices(list(s0._price_panel.index), opt_data)
    ap = {p: s0._price_panel[p].tolist() for p in PRODUCTS}
    ap.update(op)
    allp = {p: p for p in PRODUCTS}
    for sym, c in opt_contracts.items():
        allp[sym] = c.product
    reg = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=reg)

    rr = VrpRouterConfig(long_hold=7, short_hold=15, delta_notional_pct=0.25)
    st = VrpRouterStrategy(
        router_config=rr, option_data=opt_data, option_contracts=opt_contracts,
        greeks_cache=gcache, products=PRODUCTS, config=cfg_base, start=start, end=end,
    )
    st._price_panel = s0._price_panel
    st._ohlc_map = s0._ohlc_map
    st._vols = s0._vols
    st._ois = s0._ois
    st._iv_panel = s0._iv_panel
    st._vrp_pct = s0._vrp_pct
    st._mom20 = s0._mom20
    st._iv_pct = s0._iv_pct
    st._precomputed_lots = lots
    t0 = time.time()
    st._precompute_router_targets()
    res = run_cta_backtest(
        dates=list(st._price_panel.index), prices=ap, products=allp,
        strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000,
                                              commission_rate=0.0001, margin_call=True),
        registry=reg, risk=risk, option_contracts=opt_contracts, option_settle=os_,
    )
    m = metrics(res.nav)
    s = st.stats
    print(f"{tag}: AnnRet={m['AnnRet']:.1%} Sharpe={m['Sharpe']:.2f} "
          f"MaxDD={m['MaxDD']:.1%} Calmar={m['Calmar']:.2f} "
          f"L={s['long_buys']} S={s['short_buys']} exits={s['exits']} [{time.time()-t0:.0f}s]")


def main():
    option_data, option_contracts = load_option_data(PRODUCTS, "2023-01-01", "2025-12-31")
    gcache = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    print(f"contracts={len(option_contracts)}, greeks={gcache.size:,}")
    print("=== 最优参数 (long7/short15, delta 0.20) ===")
    run_one("2018-01-01", "2025-12-31", option_data, option_contracts, gcache, "全时段")
    print("=== OOS: 2025 独立测试 ===")
    run_one("2025-01-01", "2025-12-31", option_data, option_contracts, gcache, "2025 OOS")


if __name__ == "__main__":
    main()

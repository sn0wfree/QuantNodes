#!/usr/bin/env python3
"""双向 VRP 路由参数扫描: IV过滤 × 持有期 × delta市值 × 白名单."""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
from dataclasses import replace
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np

from QuantNodes.strategy.cta.options.vrp_router import VrpRouterStrategy, VrpRouterConfig
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

BASE = {"AnnRet": 0.235, "Sharpe": 2.09, "MaxDD": -0.091, "Calmar": 2.57}


def metrics(nav):
    r = nav.pct_change().dropna()
    if len(r) == 0:
        return {}
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann / abs(dd) if abs(dd) > 0 else 0
    return {"AnnRet": ann, "Sharpe": sharpe, "MaxDD": dd, "Calmar": calmar}


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    cfg_base = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
    )

    print("Loading option data once...")
    option_data, option_contracts = load_option_data(products, "2023-01-01", "2025-12-31")
    gcache = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    print(f"  contracts: {len(option_contracts)}, greeks: {gcache.size:,}")

    strat0 = VrpRouterStrategy(
        router_config=VrpRouterConfig(),
        option_data=option_data, option_contracts=option_contracts,
        greeks_cache=gcache, products=products, config=cfg_base,
        start="2018-01-01", end="2025-12-31",
    )
    strat0.load_panel()
    print("Computing fut_v1 signals once...")
    signals = strat0._signals()
    exec_sig = strat0._exec_sig(signals)
    lots = strat0._signal_to_lots(exec_sig, cfg_base)
    strat0._load_iv_panel()
    strat0._compute_signal_panels()

    opt_prices, opt_settle = build_option_prices(list(strat0._price_panel.index), option_data)
    all_prices = {p: strat0._price_panel[p].tolist() for p in products}
    all_prices.update(opt_prices)
    all_products = {p: p for p in products}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product

    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    variants = [
        ("base", {}),
        ("iv<0.25", {"iv_rank_max": 0.25}),
        ("iv<0.20", {"iv_rank_max": 0.20}),
        ("delta.10", {"delta_notional_pct": 0.10}),
        ("delta.20", {"delta_notional_pct": 0.20}),
        ("long7/short15", {"long_hold": 7, "short_hold": 15}),
        ("long3/short7", {"long_hold": 3, "short_hold": 7}),
        ("iv<0.25+d.10", {"iv_rank_max": 0.25, "delta_notional_pct": 0.10}),
        ("iv<0.20+d.10", {"iv_rank_max": 0.20, "delta_notional_pct": 0.10}),
        ("short+MTA", {"short_products": ["M", "TA"]}),
        ("short+MTA+iv", {"short_products": ["M", "TA"], "iv_rank_max": 0.25}),
    ]

    print(f"\n{'variant':<18} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>7} {'Calmar':>7} "
          f"{'L':>4} {'S':>4} {'exits':>5}")
    print("-" * 80)
    print(f"{'BASELINE':<18} {BASE['Sharpe']:>7.2f} {BASE['AnnRet']:>8.1%} "
          f"{BASE['MaxDD']:>7.1%} {BASE['Calmar']:>7.2f}")
    for name, overrides in variants:
        rr_cfg = replace(VrpRouterConfig(), **overrides)
        strat = VrpRouterStrategy(
            router_config=rr_cfg,
            option_data=option_data, option_contracts=option_contracts,
            greeks_cache=gcache, products=products, config=cfg_base,
            start="2018-01-01", end="2025-12-31",
        )
        strat._price_panel = strat0._price_panel
        strat._ohlc_map = strat0._ohlc_map
        strat._vols = strat0._vols
        strat._ois = strat0._ois
        strat._iv_panel = strat0._iv_panel
        strat._vrp_pct = strat0._vrp_pct
        strat._mom20 = strat0._mom20
        strat._iv_pct = strat0._iv_pct
        strat._precomputed_lots = lots
        t0 = time.time()
        strat._precompute_router_targets()
        result = run_cta_backtest(
            dates=list(strat._price_panel.index),
            prices=all_prices, products=all_products, strategy=strat,
            config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001,
                                     margin_call=True),
            registry=registry, risk=risk,
            option_contracts=option_contracts, option_settle=opt_settle,
        )
        m = metrics(result.nav)
        s = strat.stats
        print(f"{name:<18} {m['Sharpe']:>7.2f} {m['AnnRet']:>8.1%} {m['MaxDD']:>7.1%} "
              f"{m['Calmar']:>7.2f} {s['long_buys']:>4} {s['short_buys']:>4} "
              f"{s['exits']:>5}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

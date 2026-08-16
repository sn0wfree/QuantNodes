#!/usr/bin/env python3
"""Diagonal Lifecycle 参数扫描 — 定位负贡献来源.

变体: 放宽止损 / 单腿(无far) / 关scalping / 止盈止损组合.
数据加载与期权价格构建只做一次, 循环复用.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
from dataclasses import replace
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np

from QuantNodes.strategy.cta.options.diagonal_lifecycle import (
    DiagonalLifecycleStrategy, DiagonalLifecycleConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices


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
    return {"AnnRet": ann, "Sharpe": sharpe, "MaxDD": dd}


BASE = {"AnnRet": 0.235, "Sharpe": 2.09, "MaxDD": -0.091}


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    cfg_base = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
    )

    print("Loading option data once...")
    t0 = time.time()
    option_data, option_contracts = load_option_data(products, "2023-01-01", "2025-12-31")
    print(f"  contracts: {len(option_contracts)} ({time.time()-t0:.0f}s)")

    # fut_v1 lots 只算一次
    strat0 = DiagonalLifecycleStrategy(
        lifecycle_config=DiagonalLifecycleConfig(),
        option_data=option_data, option_contracts=option_contracts,
        products=products, config=cfg_base,
        start="2018-01-01", end="2025-12-31",
    )
    strat0.load_panel()
    print("Computing fut_v1 signals once...")
    t0 = time.time()
    signals = strat0._signals()
    exec_sig = strat0._exec_sig(signals)
    lots = strat0._signal_to_lots(exec_sig, cfg_base)
    strat0._load_iv_panel()
    print(f"  signals done ({time.time()-t0:.0f}s)")

    t0 = time.time()
    opt_prices, opt_settle = build_option_prices(
        list(strat0._price_panel.index), option_data)
    all_prices = {p: strat0._price_panel[p].tolist() for p in products}
    all_prices.update(opt_prices)
    all_products = {p: p for p in products}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product
    print(f"  prices built ({time.time()-t0:.0f}s)")

    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    variants = [
        ("delta.15", {"delta_notional_pct": 0.15}),
        ("delta.30", {"delta_notional_pct": 0.30}),
        ("delta.50", {"delta_notional_pct": 0.50}),
        ("delta.30单腿", {"delta_notional_pct": 0.30, "far_enabled": False}),
        ("delta.30 no-scalp", {"delta_notional_pct": 0.30, "scalp_threshold": 999.0}),
        ("delta.30 far.3", {"delta_notional_pct": 0.30, "far_share": 0.3}),
        ("delta.30 far.7", {"delta_notional_pct": 0.30, "far_share": 0.7}),
    ]

    print(f"\n{'variant':<20} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>7} "
          f"{'entry':>6} {'conv':>5} {'scalp':>6} {'tp':>4} {'sl':>4} {'tmo':>4}")
    print("-" * 90)
    print(f"{'BASELINE(fut_v1)':<20} {BASE['Sharpe']:>7.2f} {BASE['AnnRet']:>8.1%} "
          f"{BASE['MaxDD']:>7.1%}")
    # 共享 parquet 缓存 (变体间只加载一次)
    from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
    gcache = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    print(f"  greeks cache loaded: {gcache.size:,}")
    for name, overrides in variants:
        lc_cfg = replace(DiagonalLifecycleConfig(), **overrides)
        strat = DiagonalLifecycleStrategy(
            lifecycle_config=lc_cfg,
            option_data=option_data, option_contracts=option_contracts,
            greeks_cache_path="/tmp/opencode/dl_greeks_cache.pkl",
            greeks_cache=gcache,
            products=products, config=cfg_base,
            start="2018-01-01", end="2025-12-31",
        )
        # 复用已加载的面板/信号/IV
        strat._price_panel = strat0._price_panel
        strat._ohlc_map = strat0._ohlc_map
        strat._vols = strat0._vols
        strat._ois = strat0._ois
        strat._iv_panel = strat0._iv_panel
        strat._precomputed_lots = lots
        t0 = time.time()
        strat._precompute_lifecycle_targets()
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
        print(f"{name:<20} {m['Sharpe']:>7.2f} {m['AnnRet']:>8.1%} {m['MaxDD']:>7.1%} "
              f"{s['entries']:>6} {s['conversions']:>5} {s['scalps']:>6} "
              f"{s['tp_exits']:>4} {s['sl_exits']:>4} {s['timeout_exits']:>4}  "
              f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

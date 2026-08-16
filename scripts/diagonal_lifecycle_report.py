#!/usr/bin/env python3
"""Diagonal Lifecycle 完整业绩指标报告.

输出: 年化收益 / 年化波动 / Sharpe / MaxDD / Calmar
对比: fut_v1 baseline vs delta.15 (最优) vs delta.30.
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
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices


def full_metrics(nav):
    r = nav.pct_change().dropna()
    if len(r) == 0:
        return {}
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann_ret = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 0 else 0
    return {
        "AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe,
        "MaxDD": dd, "Calmar": calmar,
    }


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
    print(f"  contracts: {len(option_contracts)}")

    # 共享 parquet greeks 缓存
    from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
    gcache = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    print(f"  greeks cache: {gcache.size:,}")

    # ── Baseline (全时段回测, 独立) ──
    print("\nRunning baseline fut_v1...")
    t0 = time.time()
    base = FactorModuleComboStrategy(products=products, config=cfg_base,
                                     start="2018-01-01", end="2025-12-31")
    res_base = base.run()
    m_base = full_metrics(res_base.nav)
    print(f"  done ({time.time()-t0:.0f}s)")

    # ── fut_v1 lots 一次 ──
    strat0 = DiagonalLifecycleStrategy(
        lifecycle_config=DiagonalLifecycleConfig(),
        option_data=option_data, option_contracts=option_contracts,
        greeks_cache=gcache, products=products, config=cfg_base,
        start="2018-01-01", end="2025-12-31",
    )
    strat0.load_panel()
    signals = strat0._signals()
    exec_sig = strat0._exec_sig(signals)
    lots = strat0._signal_to_lots(exec_sig, cfg_base)
    strat0._load_iv_panel()

    opt_prices, opt_settle = build_option_prices(list(strat0._price_panel.index), option_data)
    all_prices = {p: strat0._price_panel[p].tolist() for p in products}
    all_prices.update(opt_prices)
    all_products = {p: p for p in products}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product

    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    variants = [
        ("delta.15", {"delta_notional_pct": 0.15}),
        ("delta.30", {"delta_notional_pct": 0.30}),
    ]

    print(f"\n{'strategy':<14} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>7} "
          f"{'MaxDD':>8} {'Calmar':>7}")
    print("-" * 60)
    m = m_base
    print(f"{'baseline fut_v1':<14} {m['AnnRet']:>8.1%} {m['AnnVol']:>8.1%} "
          f"{m['Sharpe']:>7.2f} {m['MaxDD']:>8.1%} {m['Calmar']:>7.2f}")

    for name, overrides in variants:
        lc_cfg = replace(DiagonalLifecycleConfig(), **overrides)
        strat = DiagonalLifecycleStrategy(
            lifecycle_config=lc_cfg,
            option_data=option_data, option_contracts=option_contracts,
            greeks_cache=gcache, products=products, config=cfg_base,
            start="2018-01-01", end="2025-12-31",
        )
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
        m = full_metrics(result.nav)
        s = strat.stats
        print(f"{name:<14} {m['AnnRet']:>8.1%} {m['AnnVol']:>8.1%} "
              f"{m['Sharpe']:>7.2f} {m['MaxDD']:>8.1%} {m['Calmar']:>7.2f}  "
              f"[entry={s['entries']} conv={s['conversions']} scalp={s['scalps']} "
              f"tp={s['tp_exits']} sl={s['sl_exits']} ({time.time()-t0:.0f}s)]")


if __name__ == "__main__":
    main()

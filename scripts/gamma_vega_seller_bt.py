#!/usr/bin/env python3
"""Gamma-Vega 卖方回测: 大跌后分步建仓卖 put spread (docs/108 §5).

对比: fut_v1 基线 vs fut_v1 + GammaVegaSeller.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np

from QuantNodes.strategy.cta.options.gamma_vega_seller import (
    GammaVegaSeller, GammaVegaConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices


def compute_full_metrics(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann_ret = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 0 else 0
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_drawdown": dd, "calmar": calmar}


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    start = "2018-01-01"
    end = "2025-12-31"

    print("Gamma-Vega Seller — 大跌后分步建仓卖 put spread (docs/108)")
    print("=" * 70)

    option_data, option_contracts = load_option_data(products, "2023-01-01", "2025-12-31")
    print(f"  Option contracts: {len(option_contracts)}")
    gcache = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    print(f"  greeks cache: {gcache.size:,}")

    cfg_base = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
    )

    print("\n--- Baseline fut_v1 ---")
    t0 = time.time()
    base = FactorModuleComboStrategy(products=products, config=cfg_base,
                                     start=start, end=end)
    res_base = base.run()
    m_base = compute_full_metrics(res_base.nav)
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"MaxDD={m_base['max_drawdown']:>6.1%}  ({time.time()-t0:.0f}s)")

    # Gamma-Vega 卖方
    print("\n--- fut_v1 + GammaVegaSeller ---")
    t0 = time.time()
    gv_cfg = GammaVegaConfig(
        down_days=3, vrp_min=0.50,
        good_products=["I", "AG", "CU", "SR", "TA", "RU", "M"],
        max_batches=3, batch_add_if_down=True, batch_size=1.0 / 3.0,
        atm_mult=1.00, put_otm_mult=0.88, min_dte=15, max_dte=40,
        hold_days=7, delta_notional_pct=0.10,
    )
    strat = GammaVegaSeller(
        gv_config=gv_cfg,
        option_data=option_data, option_contracts=option_contracts,
        greeks_cache=gcache,
        products=products, config=cfg_base, start=start, end=end,
    )
    strat.load_panel()
    signals = strat._signals()
    exec_sig = strat._exec_sig(signals)
    lots = strat._signal_to_lots(exec_sig, cfg_base)
    strat._precomputed_lots = lots
    strat._load_iv_panel()
    strat._compute_signal_panels()
    print("  Precomputing gv targets...")
    n_opt = strat._precompute_gv_targets()
    n_days = sum(1 for v in n_opt.values() if v)
    print(f"  Precomputed: {n_days} days with targets ({time.time()-t0:.0f}s)")
    print(f"  Stats: {strat.stats}")

    opt_prices, opt_settle = build_option_prices(list(strat._price_panel.index), option_data)
    all_prices = {p: strat._price_panel[p].tolist() for p in products}
    all_prices.update(opt_prices)
    all_products = {p: p for p in products}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product

    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)
    t0 = time.time()
    result = run_cta_backtest(
        dates=list(strat._price_panel.index),
        prices=all_prices, products=all_products, strategy=strat,
        config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001,
                                 margin_call=True),
        registry=registry, risk=risk,
        option_contracts=option_contracts, option_settle=opt_settle,
    )
    m = compute_full_metrics(result.nav)
    print(f"  AnnRet={m['ann_return']:>6.1%}  Sharpe={m['sharpe']:>6.2f}  "
          f"MaxDD={m['max_drawdown']:>6.1%}  Trades={len(result.trades):>5d}  "
          f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  {'':>15} {'Baseline':>12} {'GV Seller':>12} {'Improvement':>12}")
    print("  " + "-" * 50)
    for key in ["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar"]:
        v1 = m_base.get(key, 0)
        v2 = m.get(key, 0)
        diff = v2 - v1
        if key in ("ann_return", "ann_vol", "max_drawdown"):
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagonal Lifecycle 回测 — fut_v1 + 对角线生命周期期权 (docs/103).

对比: fut_v1 基线 vs fut_v1 + 三腿对角线生命周期 (scalping 期货调整).
输出: 组合指标 + 状态机统计 (入场/转换/scalp/退出).
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.diagonal_lifecycle import (
    DiagonalLifecycleStrategy, DiagonalLifecycleConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import (
    CtaContractRegistry, CtaContract, OptionContractSpec,
)
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk


def compute_full_metrics(nav):
    returns = nav.pct_change().dropna()
    if len(returns) == 0:
        return {}
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0
    cummax = nav.cummax()
    max_dd = ((nav - cummax) / cummax).min()
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_drawdown": max_dd, "calmar": calmar,
    }


def load_option_data(products, start, end):
    """加载期权链 + contracts (groupby 加速)."""
    loader = CtaDataLoader()
    registry = CtaContractRegistry()
    option_data = {}
    option_contracts = {}
    for prod in products:
        try:
            opt_df = loader.load_v_all_options(products=[prod], start=start, end=end)
            if opt_df is not None and not opt_df.empty:
                opt_df["trade_date"] = pd.to_datetime(opt_df["trade_date"])
                opt_df["symbol"] = opt_df["symbol"].str.upper()
                option_data[prod] = opt_df
                spec = registry.get_or_default(prod)
                for sym, row in opt_df.groupby("symbol").first().iterrows():
                    opt_spec = OptionContractSpec(
                        exchange="", product=prod,
                        underlying=str(row.get("underlying", "")),
                        option_type=str(row.get("option_type", "")).upper(),
                        strike=float(row["strike_price"]),
                        expiry_date="",
                        contract_multiplier=spec.contract_multiplier,
                    )
                    option_contracts[sym] = CtaContract(
                        product=prod, symbol=sym, expiry_year=0, expiry_month=0,
                        spec=spec, option_spec=opt_spec,
                    )
        except Exception:
            pass
    return option_data, option_contracts


def build_option_prices(dates, option_data):
    """期权逐日价格序列 {symbol: [price]} + 到期结算价 map.

    关键修复: 期权 symbol 必须进 prices dict, 否则引擎 mark_prices.get(None)
    → place_order 被跳过, 期权层从未真正执行.
    """
    idx = pd.DatetimeIndex(dates)
    prices = {}
    settle = {}
    for prod, odf in option_data.items():
        for sym, grp in odf.groupby("symbol"):
            ser = grp.set_index("trade_date")["close"].sort_index()
            ser = ser[~ser.index.duplicated(keep="last")]
            prices[sym] = ser.reindex(idx, method="ffill").fillna(0.0).tolist()
            settle[sym] = {d.strftime("%Y-%m-%d"): v for d, v in ser.items()}
    return prices, settle


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    start = "2018-01-01"
    end = "2025-12-31"
    opt_start = "2023-01-01"
    opt_end = "2025-12-31"

    print("Diagonal Lifecycle — fut_v1 overlay (docs/103)")
    print("=" * 70)

    print("\nLoading option data (2023-2025)...")
    t0 = time.time()
    option_data, option_contracts = load_option_data(products, opt_start, opt_end)
    print(f"  Option contracts: {len(option_contracts)} ({time.time()-t0:.0f}s)")

    cfg_base = ComboConfig(
        initial_cash=5_000_000,
        target_vol=0.30,
        max_leverage=3.0,
        vol_lookback=20,
        momentum_filter_window=60,
        momentum_filter_coef=0.0,
        filter_enabled=True,
        module_weighting="inv_vol",
        continuous=True,
        cont_z_window=252,
        cont_z_clip=3.0,
        margin_call=True,
        execution_lag=1,
        aggregation="tvpr",
    )

    # ── 基线 ──
    print("\n--- Baseline fut_v1 ---")
    t0 = time.time()
    strat_base = FactorModuleComboStrategy(
        products=products, config=cfg_base, start=start, end=end,
    )
    result_base = strat_base.run()
    m_base = compute_full_metrics(result_base.nav)
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"MaxDD={m_base['max_drawdown']:>6.1%}  ({time.time()-t0:.0f}s)")

    # ── 对角线生命周期 ──
    print("\n--- fut_v1 + Diagonal Lifecycle ---")
    t0 = time.time()
    lc_cfg = DiagonalLifecycleConfig(
        loc_window=20, loc_threshold=0.15, iv_rank_max=0.70,
        near_min_dte=15, near_max_dte=40, far_min_dte=45, far_max_dte=75,
        buy_strike_mult=1.00, sell_strike_mult=1.00,
        confirm_window=5, confirm_min_up=0.002, confirm_max_up=0.03,
        scalp_threshold=0.5,
        roll_days=7, take_profit=0.50, stop_loss=0.30,
        delta_notional_pct=0.30, far_share=0.5, far_enabled=True,
    )
    strat = DiagonalLifecycleStrategy(
        lifecycle_config=lc_cfg,
        option_data=option_data,
        option_contracts=option_contracts,
        greeks_cache_path="/tmp/opencode/dl_greeks_cache.pkl",
        greeks_parquet="data/cache/option_greeks.parquet",
        products=products,
        config=cfg_base,
        start=start,
        end=end,
    )
    strat.load_panel()
    signals = strat._signals()
    exec_sig = strat._exec_sig(signals)
    lots = strat._signal_to_lots(exec_sig, cfg_base)
    strat._precomputed_lots = lots
    strat._loc_cache = None
    strat._load_iv_panel()
    print("  Precomputing lifecycle targets...")
    n_opt = strat._precompute_lifecycle_targets()
    n_days = sum(1 for v in n_opt.values() if v)
    n_targets = sum(len(v) for v in n_opt.values())
    print(f"  Precomputed: {len(n_opt)} days, {n_days} days with targets, "
          f"{n_targets} option targets ({time.time()-t0:.0f}s)")
    print(f"  Stats: {strat.stats}")

    # 期权价格 + 到期结算
    t0 = time.time()
    opt_prices, opt_settle = build_option_prices(
        list(strat._price_panel.index), option_data,
    )
    all_prices = {p: strat._price_panel[p].tolist() for p in products}
    all_prices.update(opt_prices)
    # products map 覆盖期权 symbol (引擎逐 symbol 查 product)
    all_products = {p: p for p in products}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product
    print(f"  Option prices built: {len(opt_prices)} syms ({time.time()-t0:.0f}s)")

    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)
    t0 = time.time()
    result = run_cta_backtest(
        dates=list(strat._price_panel.index),
        prices=all_prices,
        products=all_products,
        strategy=strat,
        config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001,
                                 margin_call=True),
        registry=registry,
        risk=risk,
        option_contracts=option_contracts,
        option_settle=opt_settle,
    )
    m_lc = compute_full_metrics(result.nav)
    print(f"  AnnRet={m_lc['ann_return']:>6.1%}  Sharpe={m_lc['sharpe']:>6.2f}  "
          f"MaxDD={m_lc['max_drawdown']:>6.1%}  Trades={len(result.trades):>5d}  "
          f"({time.time()-t0:.0f}s)")

    # ── 对比 ──
    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  {'':>15} {'Baseline':>12} {'Lifecycle':>12} {'Improvement':>12}")
    print("  " + "-" * 50)
    for key in ["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar"]:
        v1 = m_base.get(key, 0)
        v2 = m_lc.get(key, 0)
        diff = v2 - v1
        if key in ("ann_return", "ann_vol", "max_drawdown"):
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Options-Boosted fut_v1 回测 — 裸买 Call/Put 增强.

对比: fut_v1 vs fut_v1 + 期权因子 vs fut_v1 + 裸买期权
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import pandas as pd
import numpy as np
from QuantNodes.strategy.cta.options.options_boosted_fut_v1 import (
    OptionsBoostedFutV1,
    OptionsBoostConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy,
    ComboConfig,
)
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry, CtaContract, OptionContractSpec


def compute_full_metrics(nav):
    returns = nav.pct_change().dropna()
    if len(returns) == 0:
        return {}
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (ann_return - 0.02) / down_vol if down_vol > 0 else 0
    cummax = nav.cummax()
    max_dd = ((nav - cummax) / cummax).min()
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0
    win_rate = (returns > 0).sum() / len(returns)
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar, "win_rate": win_rate,
    }


def load_option_data(products, start, end):
    """加载期权数据 + option_contracts."""
    loader = CtaDataLoader()
    registry = CtaContractRegistry()

    option_data = {}
    option_contracts = {}

    for prod in products:
        try:
            opt_df = loader.load_v_all_options(products=[prod], start=start, end=end)
            if opt_df is not None and not opt_df.empty:
                opt_df["trade_date"] = pd.to_datetime(opt_df["trade_date"])
                option_data[prod] = opt_df

                for sym in opt_df["symbol"].unique():
                    row = opt_df[opt_df["symbol"] == sym].iloc[0]
                    opt_spec = OptionContractSpec(
                        exchange="", product=prod,
                        underlying=str(row.get("underlying", "")),
                        option_type=str(row.get("option_type", "")).upper(),
                        strike=float(row["strike_price"]),
                        expiry_date="",
                        contract_multiplier=registry.get_or_default(prod).contract_multiplier,
                    )
                    contract = CtaContract(
                        product=prod, symbol=sym, expiry_year=0, expiry_month=0,
                        spec=registry.get_or_default(prod), option_spec=opt_spec,
                    )
                    option_contracts[sym] = contract
        except Exception:
            pass

    return option_data, option_contracts


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    start = "2023-01-01"
    end = "2025-12-31"

    print("Options-Boosted fut_v1 — Multi-Product Portfolio")
    print("=" * 70)

    # 加载期权数据
    print("\nLoading option data...")
    option_data, option_contracts = load_option_data(products, start, end)
    print(f"  Option contracts: {len(option_contracts)}")
    print(f"  Products with options: {list(option_data.keys())}")

    # ═══ 策略 1: 基线 fut_v1 ═══
    print("\n--- Baseline fut_v1 (72 factors) ---")
    t0 = time.time()
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
    strat_base = FactorModuleComboStrategy(
        products=products, config=cfg_base, start=start, end=end,
    )
    result_base = strat_base.run()
    m_base = compute_full_metrics(result_base.nav)
    t1 = time.time() - t0
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"MaxDD={m_base['max_drawdown']:>6.1%}  Calmar={m_base['calmar']:>6.2f}  ({t1:.0f}s)")

    # ═══ 策略 2: fut_v1 + 裸买期权 ═══
    print("\n--- fut_v1 + Naked Long Options (VRP boost) ---")
    t0 = time.time()
    boost_cfg = OptionsBoostConfig(
        boost_notional_pct=0.3,
        boost_max_lots=5,
        iv_high_threshold=0.7,
        iv_zscore_max=2.0,
        min_dte=30, max_dte=60, roll_dte=10,
        vrp_boost_high=0.8, vrp_boost_mid=0.5, vrp_boost_low=0.2,
        vrp_min_threshold=0.2,
        enable_put_spread=False,  # 先禁用 Put Spread 测其他升级
    )
    strat_boost = OptionsBoostedFutV1(
        boost_config=boost_cfg,
        option_contracts=option_contracts,
        option_data=option_data,
        products=products,
        config=cfg_base,
        start=start,   # 关键: 限制回测日期范围, 避免加载全量数据
        end=end,
    )

    # 运行回测
    registry = CtaContractRegistry()
    from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    # 先跑信号 → 预计算 lots + 期权目标
    # 关键修复: 必须走 _exec_sig (T+1 lag), 与 fut_v1 原版 _run_signal 一致
    strat_boost.load_panel()
    signals = strat_boost._signals()
    exec_sig = strat_boost._exec_sig(signals)  # T+1: T 日信号 → T+1 日成交
    lots = strat_boost._signal_to_lots(exec_sig, cfg_base)
    strat_boost._precomputed_lots = lots  # 供 target_positions 使用
    print("  Precomputing option targets (T+1)...")
    n_opt = strat_boost._precompute_option_targets()
    n_days_with_opts = sum(1 for v in n_opt.values() if v)
    n_opts = sum(len(v) for v in n_opt.values())
    print(f"  Precomputed: {len(n_opt)} days, {n_days_with_opts} days with options, {n_opts} option targets")

    t0 = time.time()
    result_boost = run_cta_backtest(
        dates=list(strat_boost._price_panel.index),
        prices={p: strat_boost._price_panel[p].tolist() for p in products},
        products={p: p for p in products},
        strategy=strat_boost,  # 直接使用增强策略的 target_positions
        config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001, margin_call=True),
        registry=registry,
        risk=risk,
        option_contracts=option_contracts,
    )
    m_boost = compute_full_metrics(result_boost.nav)
    t2 = time.time() - t0
    print(f"  AnnRet={m_boost['ann_return']:>6.1%}  Sharpe={m_boost['sharpe']:>6.2f}  "
          f"MaxDD={m_boost['max_drawdown']:>6.1%}  Calmar={m_boost['calmar']:>6.2f}  "
          f"Trades={len(result_boost.trades):>4d}  ({t2:.0f}s)")

    # 对比
    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  {'':>15} {'Baseline':>12} {'Boosted':>12} {'Improvement':>12}")
    print("  " + "-" * 50)
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]:
        v1 = m_base.get(key, 0)
        v2 = m_boost.get(key, 0)
        diff = v2 - v1
        if key in ["ann_return", "ann_vol", "max_drawdown", "win_rate"]:
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()

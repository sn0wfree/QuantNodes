#!/usr/bin/env python3
"""Options-Enhanced fut_v1 回测对比.

对比: fut_v1 (72因子) vs fut_v1 + 期权因子
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import pandas as pd
import numpy as np
from QuantNodes.strategy.cta.options.options_enhanced_fut_v1 import (
    OptionsEnhancedFactorModule,
    OptionsFactorConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy,
    ComboConfig,
)
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_engine import CtaBacktestConfig


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


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]

    print("Options-Enhanced fut_v1 — Multi-Product Portfolio")
    print("=" * 70)

    # 基线 fut_v1
    print("\nBaseline fut_v1 (72 factors, no options):")
    t0 = time.time()
    cfg_base = ComboConfig(
        initial_cash=10_000_000,
        target_vol=0.15,
        max_leverage=3.0,
        vol_lookback=20,
        momentum_filter_window=60,
        momentum_filter_coef=0.5,
        filter_enabled=True,
        module_weighting="inv_vol",
    )
    strat_base = FactorModuleComboStrategy(products=products, config=cfg_base)
    result_base = strat_base.run()
    m_base = compute_full_metrics(result_base.nav)
    t1 = time.time() - t0
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"Sortino={m_base['sortino']:>6.2f}  MaxDD={m_base['max_drawdown']:>6.1%}  "
          f"Calmar={m_base['calmar']:>6.2f}  WinRate={m_base['win_rate']:>5.1%}  ({t1:.0f}s)")

    # Options-Enhanced fut_v1
    print("\nOptions-Enhanced fut_v1 (72 factors + options factors):")
    t0 = time.time()
    opt_cfg = OptionsFactorConfig()
    strat_enh = OptionsEnhancedFactorModule(
        options_config=opt_cfg,
        products=products,
        config=cfg_base,
    )
    result_enh = strat_enh.run()
    m_enh = compute_full_metrics(result_enh.nav)
    t2 = time.time() - t0
    print(f"  AnnRet={m_enh['ann_return']:>6.1%}  Sharpe={m_enh['sharpe']:>6.2f}  "
          f"Sortino={m_enh['sortino']:>6.2f}  MaxDD={m_enh['max_drawdown']:>6.1%}  "
          f"Calmar={m_enh['calmar']:>6.2f}  WinRate={m_enh['win_rate']:>5.1%}  ({t2:.0f}s)")

    # 对比
    print("\n" + "=" * 70)
    print("Improvement:")
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]:
        v1 = m_base.get(key, 0)
        v2 = m_enh.get(key, 0)
        diff = v2 - v1
        if key in ["ann_return", "ann_vol", "max_drawdown", "win_rate"]:
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()

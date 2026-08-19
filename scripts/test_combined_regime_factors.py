"""三方案组合测试:
1. 树模型震荡过滤 (降低震荡市仓位)
2. TurnPointBreakout (快速捕捉拐点)
3. ShortTermReversalTrend (震荡市超卖反弹)
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS,
    TrendMomentumModuleStrategy, TurnPointBreakoutFactor, FastTurnPointFactor,
    ShortTermReversalTrendFactor,
)
from QuantNodes.strategy.cta.factors.volatility import vol_target_positions
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

PRODUCTS_10 = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]


def load_prices():
    """加载价格数据."""
    cfg = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting='inv_vol', continuous=True, cont_z_window=252, cont_z_clip=3.0,
        margin_call=True, execution_lag=1, aggregation='tvpr',
        commission_rate=0.0002, min_position_change=0.30, slippage_bps=0.5,
        position_ema_alpha=0.15, tail_risk_enabled=True, tail_risk_corr_threshold=0.30,
        enable_cluster_filter=False, execution_price_type='open',
    )
    s = FactorModuleComboStrategy(products=PRODUCTS_10, config=cfg, start='2018-01-01', end='2025-12-31')
    s.load_panel()
    return s, cfg


def run_test(name, s, cfg, choppy_prob=None, regime_th=(0.10, 0.20), custom_factors=None):
    """运行单个测试."""
    if custom_factors is not None:
        original = TrendMomentumModuleStrategy.FACTORS.copy()
        TrendMomentumModuleStrategy.FACTORS = custom_factors

    if choppy_prob is not None:
        original_method = s._signal_to_lots
        prob_ref = {'prob': choppy_prob}

        def patched(signals, cfg):
            weights = vol_target_positions(
                signals, s._price_panel, target_vol=cfg.target_vol,
                vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
            )
            if s._filter is not None:
                weights = weights * s._filter.coefficients(s._price_panel)
            weights = s._tail_risk_filter(weights)

            prob = prob_ref['prob']
            scale = pd.Series(1.0, index=weights.index)
            for idx in weights.index:
                if idx in prob.index:
                    p = prob.loc[idx]
                    if p > regime_th[1]:
                        scale.loc[idx] = 0.3
                    elif p > regime_th[0]:
                        scale.loc[idx] = 0.6
            weights = weights.mul(scale, axis=0)
            return s._weights_to_lots(weights, cfg)

        s._signal_to_lots = patched

    t0 = time.time()
    r = s.run()
    m = compute_metrics(r.nav)
    elapsed = time.time() - t0

    nav_2024 = r.nav[r.nav.index.year == 2024]
    ret_2024 = float(nav_2024.iloc[-1] / nav_2024.iloc[0] - 1) if len(nav_2024) > 10 else 0

    if choppy_prob is not None:
        s._signal_to_lots = original_method
    if custom_factors is not None:
        TrendMomentumModuleStrategy.FACTORS = original

    return m, ret_2024, elapsed


def main():
    print("=" * 80)
    print("三方案组合测试")
    print("=" * 80)

    s, cfg = load_prices()

    # 加载震荡概率
    print("\n加载震荡概率...")
    choppy_prob = pd.read_parquet('output/market_regime_choppy_prob.parquet')['choppy_prob']

    # 获取原始因子列表
    original_factors = TrendMomentumModuleStrategy.FACTORS.copy()

    # 因子组合配置
    factor_tests = {
        "原版 (有turn_point)": [
            (n, f, t) for (n, f, t) in original_factors
            if n not in ("turn_point_breakout", "fast_turn_point", "short_term_reversal_trend")
        ] + [(n, f, t) for (n, f, t) in original_factors if n in ("turn_point_breakout",)],
        "+ TurnPoint": original_factors,
        "+ TurnPoint + ShortTerm": original_factors,
        "+ TurnPoint + ShortTerm + 树模型": original_factors,
    }

    # 补充ShortTerm到组合中
    factor_tests["+ TurnPoint + ShortTerm"] = [
        (n, f, t) for (n, f, t) in original_factors
        if n != "short_term_reversal_trend"
    ]
    factor_tests["+ TurnPoint + ShortTerm + 树模型"] = factor_tests["+ TurnPoint + ShortTerm"]

    results = []

    for name, factors in factor_tests.items():
        if "树模型" in name:
            m, ret_2024, elapsed = run_test(name, s, cfg, choppy_prob=choppy_prob, custom_factors=factors)
        else:
            m, ret_2024, elapsed = run_test(name, s, cfg, custom_factors=factors)
        results.append((name, m, ret_2024, elapsed))
        print(f"  {name}: Sharpe={m['Sharpe']:.3f}  AnnRet={m['AnnRet']:.1%}  MaxDD={m['MaxDD']:.1%}  2024={ret_2024:>+.1%}  ({elapsed:.0f}s)")

    # 对比
    print(f"\n" + "=" * 80)
    print("对比汇总")
    print("=" * 80)
    print(f"  {'配置':<50} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} {'2024':>8}")
    print("-" * 85)
    for name, m, r2024, _ in results:
        print(f"  {name:<50} {m['Sharpe']:>8.3f} {m['AnnRet']:>7.1%} {m['MaxDD']:>7.1%} {r2024:>+7.1%}")


if __name__ == "__main__":
    main()

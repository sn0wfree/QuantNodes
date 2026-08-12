"""
Trading Cost Verification for G/D/F Schemes
=============================================

Tests G vs D vs F under multiple cost scenarios:
- 0bp (no cost, as current NAV)
- 6bp (realistic A-share ETF round-trip/2)
- 10bp (current codebase default)
- 15bp (high cost scenario)
- 20bp (worst case)

Outputs:
- Per-scheme Sharpe, Calmar, MaxDD at each cost level
- Break-even cost (where G still beats D)
- Annual cost drag analysis
"""

import sys
sys.path.insert(0, '/home/ll/Public/QuantNodes')

import numpy as np
import pandas as pd
from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
    scheme_d_signal_weighted,
    scheme_f_conservative,
    scheme_g_trend_adaptive,
    load_navs,
    compute_nav,
)
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics


def compute_nav_with_cost(prices, weights_history, cost_bp: int = 10):
    """NAV with explicit cost_bp (replaces default 10)."""
    rets = prices.pct_change().fillna(0)
    port_ret = (weights_history.shift(1).fillna(method='ffill').fillna(0) * rets).sum(axis=1)
    turnover = weights_history.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * cost_bp / 10000
    nav = (1 + port_ret - cost).cumprod()
    nav.iloc[0] = 1.0
    return nav


def analyze_cost_sensitivity(prices, schemes, cost_levels):
    """Analyze all schemes under different cost scenarios."""
    results = []

    for scheme_name, scheme_func in schemes.items():
        weights = scheme_func(prices)

        for cost_bp in cost_levels:
            nav = compute_nav_with_cost(prices, weights, cost_bp)
            nav_oos = nav.loc['2022-01-01':].dropna()
            m = compute_metrics(nav_oos)

            # Turnover stats
            turnover = weights.diff().abs().sum(axis=1).fillna(0)
            avg_turnover = turnover.loc['2022-01-01':].mean()
            max_turnover = turnover.loc['2022-01-01':].max()

            # Annual cost drag
            annual_cost = avg_turnover * cost_bp / 10000 * 252  # daily turnover * cost * trading days

            results.append({
                'scheme': scheme_name,
                'cost_bp': cost_bp,
                'sharpe': m.get('Sharpe', 0),
                'calmar': m.get('Calmar', 0),
                'maxdd': m.get('MaxDD', 0),
                'total_return': m.get('Total Return', 0),
                'annual_cost_drag': annual_cost,
                'avg_turnover': avg_turnover,
                'max_turnover': max_turnover,
            })

    return pd.DataFrame(results)


def find_breakeven_cost(prices, scheme_a, scheme_b, scheme_a_name, scheme_b_name):
    """Find the cost level where scheme_a stops beating scheme_b."""
    costs = np.arange(0, 30, 0.5)
    for cost_bp in costs:
        nav_a = compute_nav_with_cost(prices, scheme_a, cost_bp)
        nav_b = compute_nav_with_cost(prices, scheme_b, cost_bp)

        m_a = compute_metrics(nav_a.loc['2022-01-01':].dropna())
        m_b = compute_metrics(nav_b.loc['2022-01-01':].dropna())

        if m_a.get('Sharpe', 0) < m_b.get('Sharpe', 0):
            return cost_bp
    return 30.0  # A always wins


def main():
    print("=" * 80)
    print("Trading Cost Verification: G vs D vs F Schemes")
    print("=" * 80)

    prices = load_navs()
    print(f"Prices: {prices.shape[0]} days, {prices.shape[1]} ETFs")

    # Define schemes
    schemes = {
        'D (signal-weighted)': scheme_d_signal_weighted,
        'F (conservative)': scheme_f_conservative,
        'G (trend-adaptive)': scheme_g_trend_adaptive,
    }

    # Cost scenarios
    cost_levels = [0, 5, 6, 8, 10, 12, 15, 20]

    # Full analysis
    print("\n" + "=" * 80)
    print("1. Cost Sensitivity Analysis (OOS: 2022-01-01 ~ 2026-05-29)")
    print("=" * 80)

    df = analyze_cost_sensitivity(prices, schemes, cost_levels)

    print(f"\n{'Scheme':<22} {'Cost':>6} {'Sharpe':>8} {'Calmar':>8} {'MaxDD':>8} {'TotalRet':>9} {'AnnCost':>8} {'AvgTO':>8} {'MaxTO':>8}")
    print("-" * 100)

    for _, row in df.iterrows():
        print(f"{row['scheme']:<22} {row['cost_bp']:>5.0f}bp {row['sharpe']:>8.3f} {row['calmar']:>8.3f} {row['maxdd']:>7.2%} {row['total_return']:>8.2%} {row['annual_cost_drag']:>7.2%} {row['avg_turnover']:>7.2%} {row['max_turnover']:>7.2%}")

    # Break-even analysis
    print("\n" + "=" * 80)
    print("2. Break-even Cost Analysis")
    print("=" * 80)

    weights_g = scheme_g_trend_adaptive(prices)
    weights_d = scheme_d_signal_weighted(prices)
    weights_f = scheme_f_conservative(prices)

    be_g_vs_d = find_breakeven_cost(prices, weights_g, weights_d, 'G', 'D')
    be_f_vs_d = find_breakeven_cost(prices, weights_f, weights_d, 'F', 'D')

    print(f"\nG vs D: G stops beating D at cost = {be_g_vs_d:.1f}bp")
    print(f"F vs D: F stops beating D at cost = {be_f_vs_d:.1f}bp")
    print(f"Realistic cost (~6bp): G {'still beats' if be_g_vs_d > 6 else 'loses to'} D")

    # Marginal cost impact
    print("\n" + "=" * 80)
    print("3. Marginal Cost Impact (Sharpe loss per 1bp additional cost)")
    print("=" * 80)

    for scheme_name in ['D (signal-weighted)', 'G (trend-adaptive)']:
        base = df[(df['scheme'] == scheme_name) & (df['cost_bp'] == 0)]['sharpe'].values[0]
        high = df[(df['scheme'] == scheme_name) & (df['cost_bp'] == 20)]['sharpe'].values[0]
        marginal = (base - high) / 20
        print(f"  {scheme_name}: Sharpe drops {marginal:.4f} per 1bp additional cost")

    # Annual cost in dollars (assuming $100k portfolio)
    print("\n" + "=" * 80)
    print("4. Annual Cost in USD (assuming $100,000 portfolio)")
    print("=" * 80)

    for scheme_name in ['D (signal-weighted)', 'G (trend-adaptive)']:
        avg_to = df[(df['scheme'] == scheme_name) & (df['cost_bp'] == 10)]['avg_turnover'].values[0]
        annual_cost_6 = avg_to * 6 / 10000 * 252 * 100000
        annual_cost_10 = avg_to * 10 / 10000 * 252 * 100000
        annual_cost_15 = avg_to * 15 / 10000 * 252 * 100000
        print(f"  {scheme_name}:")
        print(f"    @ 6bp:  ${annual_cost_6:,.0f}/year")
        print(f"    @ 10bp: ${annual_cost_10:,.0f}/year")
        print(f"    @ 15bp: ${annual_cost_15:,.0f}/year")

    # Save results
    output_path = '/home/ll/Public/QuantNodes/reports/momentum_etf_rotation/v10_3/v10_3_cost_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[Saved] {output_path}")


if __name__ == '__main__':
    main()

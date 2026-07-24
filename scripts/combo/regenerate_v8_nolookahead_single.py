# coding=utf-8
"""v8 Jump Model: 逐方法跑 (避免超时)."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
    jump_model_periodic_retrain, jump_model_true_rolling, compute_features,
)
from v8_probabilistic_experiment import probabilistic_jump_model
from v8_integrated_comparison import load_v7_14_portfolio, compute_integrated_nav
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics

OOS_START = pd.Timestamp('2021-08-01')

def load_v56():
    return pd.read_parquet(REPO / "data/high_freq_macro" / "v56_expanded_daily.parquet")

def build_signals_periodic(weekly_weights, daily_returns):
    """方法 A: jump_model_periodic_retrain, 月频重训."""
    signals = {}
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    for i, code in enumerate(common_codes):
        returns = daily_returns[code].dropna()
        if len(returns) < 1000:
            continue
        states = jump_model_periodic_retrain(returns, retrain_every=30)
        n = len(states)
        probs = np.zeros((n, 2))
        for j, s in enumerate(states.values):
            probs[j, s] = 1.0
        probs_df = pd.DataFrame(probs, index=states.index, columns=['P_bull', 'P_bear'])
        bear_pct = states.rolling(60, min_periods=1).mean()
        signals[code] = {'bear_pct': bear_pct, 'prob_2state': probs_df, 'prob_3state': probs_df}
        if (i + 1) % 10 == 0:
            print(f"  periodic: {i+1}/{len(common_codes)} 完成")
    return signals

def build_signals_probabilistic(weekly_weights, daily_returns):
    """方法 B: probabilistic_jump_model, 月频重训."""
    signals = {}
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    for i, code in enumerate(common_codes):
        returns = daily_returns[code].dropna()
        if len(returns) < 1000:
            continue
        feats = compute_features(returns).dropna()
        common = returns.index.intersection(feats.index)
        rets_aligned = returns.loc[common]
        feats_aligned = feats.loc[common]
        states, probs = probabilistic_jump_model(rets_aligned, feats_aligned, retrain_every=30)
        cols = ['P_bull', 'P_bear'] if probs.shape[1] == 2 else ['P_bull', 'P_neutral', 'P_bear']
        probs_df = pd.DataFrame(probs, index=feats_aligned.index, columns=cols[:probs.shape[1]])
        bear_pct = probs_df['P_bear'].rolling(60, min_periods=1).mean()
        signals[code] = {'bear_pct': bear_pct, 'prob_2state': probs_df, 'prob_3state': probs_df}
        if (i + 1) % 10 == 0:
            print(f"  probabilistic: {i+1}/{len(common_codes)} 完成")
    return signals

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='periodic', choices=['periodic', 'probabilistic'])
    args = parser.parse_args()

    print(f"=== v8 无未来函数: {args.method} ===")
    daily_returns = load_v56()
    print(f"v56: {daily_returns.shape}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    print(f"v7.14: {weekly_weights.shape}")

    t0 = time.time()
    if args.method == 'periodic':
        signals = build_signals_periodic(weekly_weights, daily_returns)
    else:
        signals = build_signals_probabilistic(weekly_weights, daily_returns)
    print(f"\n信号生成耗时: {time.time()-t0:.0f}s, {len(signals)} ETF")

    # 跑 v8_method_b + v8_prob_2state, cost=5
    for version in ['v8_method_b', 'v8_prob_2state']:
        nav = compute_integrated_nav(weekly_weights, daily_returns, signals, version=version, cost_bp=5)
        oos = nav.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')
        print(f"{version:20s}: Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")

    # 保存 NAV
    nav_b = compute_integrated_nav(weekly_weights, daily_returns, signals, version='v8_method_b', cost_bp=5)
    out = REPO / f"reports/momentum_etf_rotation/combo/v8_{args.method}_nav_v56.parquet"
    nav_b.to_frame('v8_method_b').to_parquet(out)
    print(f"NAV 已保存: {out}")

if __name__ == "__main__":
    main()
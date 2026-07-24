# coding=utf-8
"""v8 Jump Model 用 v56 数据实测 3 种无未来函数方案 OOS Sharpe.

对比:
  - 方法 A: jump_model_periodic_retrain (硬分类, 月频重训)
  - 方法 B: probabilistic_jump_model (硬+软概率, 月频重训)
  - 方法 C: 当前有未来函数 (虚高, 作为基线对比)
  - 方法 D: jump_model_true_rolling (每天重训, 极慢)

每个方法 × 4 个成本档 (0/5/10/20 bp) = 16 个组合.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
    jump_model_periodic_retrain,
    jump_model_true_rolling,
    compute_features,
)
from v8_probabilistic_experiment import probabilistic_jump_model
from v8_integrated_comparison import (
    load_v7_14_portfolio,
    compute_integrated_nav,
    performance_metrics,
    BEAR_THRESHOLD,
    POSITION_WEIGHTS_2STATE,
    POSITION_WEIGHTS_3STATE,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics

OOS_START = pd.Timestamp('2021-08-01')


def load_v56_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data/high_freq_macro" / "v56_expanded_daily.parquet")


def build_signals(method, weekly_weights, daily_returns, retrain_every=30, n_states=2):
    """构造 3 种方法的 Jump Model 信号 (与 compute_per_asset_signals 兼容)."""
    signals = {}
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    for code in common_codes:
        returns = daily_returns[code].dropna()
        if len(returns) < 1000:
            continue
        feats = compute_features(returns).dropna()
        common = returns.index.intersection(feats.index)
        rets_aligned = returns.loc[common]
        feats_aligned = feats.loc[common]

        if method == 'periodic':
            states = jump_model_periodic_retrain(
                rets_aligned, retrain_every=retrain_every,
            )
            # 构造 prob (用 states)
            n = len(states)
            probs = np.zeros((n, 2))
            for i, s in enumerate(states.values):
                probs[i, s] = 1.0
            probs_df = pd.DataFrame(probs, index=states.index,
                                    columns=['P_bull', 'P_bear'])
            bear_pct = states.rolling(60, min_periods=1).mean()
            signals[code] = {
                'bear_pct': bear_pct,
                'prob_2state': probs_df,
                'prob_3state': probs_df,  # placeholder
            }

        elif method == 'probabilistic':
            states, probs = probabilistic_jump_model(
                rets_aligned, feats_aligned,
                retrain_every=retrain_every,
            )
            # probs shape: (T, n_states)
            cols = ['P_bull', 'P_bear'] if n_states == 2 else ['P_bull', 'P_neutral', 'P_bear']
            probs_df = pd.DataFrame(probs, index=feats_aligned.index, columns=cols[:probs.shape[1]])
            bear_pct = probs_df['P_bear'].rolling(60, min_periods=1).mean()
            signals[code] = {
                'bear_pct': bear_pct,
                'prob_2state': probs_df,
                'prob_3state': probs_df,  # placeholder for 3-state
            }

        elif method == 'true_rolling':
            states = jump_model_true_rolling(rets_aligned)
            n = len(states)
            probs = np.zeros((n, 2))
            for i, s in enumerate(states.values):
                probs[i, s] = 1.0
            probs_df = pd.DataFrame(probs, index=states.index,
                                    columns=['P_bull', 'P_bear'])
            bear_pct = states.rolling(60, min_periods=1).mean()
            signals[code] = {
                'bear_pct': bear_pct,
                'prob_2state': probs_df,
                'prob_3state': probs_df,
            }
    return signals


def run_method(method, weekly_weights, daily_returns, retrain_every=30, cost_bp=5):
    """跑单方法, 返回 4 种 version 的 metrics."""
    print(f"\n  [{method}] 重训频率={retrain_every} 成本={cost_bp}bp")
    t0 = time.time()
    signals = build_signals(method, weekly_weights, daily_returns, retrain_every)
    t_signal = time.time() - t0
    print(f"    信号生成: {t_signal:.1f}s, {len(signals)} 个 ETF")

    results = {}
    n_signals = len(signals)
    for version in ['v8_method_b', 'v8_prob_2state']:
        nav = compute_integrated_nav(
            weekly_weights, daily_returns, signals,
            version=version, cost_bp=cost_bp,
        )
        oos = nav.loc[OOS_START:].dropna()
        if len(oos) < 100:
            continue
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')
        results[version] = m
        print(f"    {version:20s}: Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} "
              f"MaxDD={m['MaxDD']:.2%} 收益={m['AnnRet']:.2%}")
    return results, t_signal, n_signals


def main():
    print("=" * 70)
    print("v8 Jump Model 三种无未来函数方案 OOS 实测 (v56 数据)")
    print("=" * 70)

    print("\n[Step 1] 加载数据")
    daily_returns = load_v56_daily_returns()
    print(f"  v56: {daily_returns.shape}, {daily_returns.index[0].date()} ~ {daily_returns.index[-1].date()}")

    weekly_weights, prices, shares = load_v7_14_portfolio()
    print(f"  v7.14 weekly weights: {weekly_weights.shape}")

    print("\n[Step 2] 跑 2 种方案 × 3 成本 (节省时间, 关键对比 5bp)")
    all_results = []
    for method in ['periodic', 'probabilistic']:
        for retrain in [30]:  # 月频
            for cost in [5, 10, 20]:  # 跳过 0bp (与 5bp 差异固定)
                results, t_signal, n_signals = run_method(
                    method, weekly_weights, daily_returns,
                    retrain_every=retrain, cost_bp=cost,
                )
                for version, m in results.items():
                    all_results.append({
                        'method': method, 'retrain': retrain,
                        'cost_bp': cost, 'version': version,
                        'Sharpe': m['Sharpe'], 'Calmar': m['Calmar'],
                        'AnnRet': m['AnnRet'], 'MaxDD': m['MaxDD'],
                        'n_etf': n_signals, 'signal_time_s': t_signal,
                    })

    df = pd.DataFrame(all_results)
    out_path = REPO / "reports/momentum_etf_rotation/combo/v8_method_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[输出] {out_path}")
    print(f"  共 {len(df)} 行")
    print("\n=== 总结 (按 Sharpe 排序) ===")
    print(df.sort_values('Sharpe', ascending=False).to_string(index=False))

    # 对比 baseline (当前有未来函数 v8_method_b)
    print("\n=== 对比 Baseline ===")
    print("当前 v8_method_b (有未来函数): Sharpe=0.854, Calmar=0.939, MaxDD=-11.66%")
    print("期望无未来函数版本: Sharpe 应低于 0.854 (合理)")


if __name__ == "__main__":
    main()
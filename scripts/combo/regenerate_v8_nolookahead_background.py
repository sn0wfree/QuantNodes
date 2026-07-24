# coding=utf-8
"""v8 Jump Model 全方法后台测试脚本 (tmux 版).

用法:
  tmux new-session -d -s v8_bg 'python3 scripts/combo/regenerate_v8_nolookahead_background.py 2>&1 | tee /tmp/v8_background.log'

监控:
  tail -f /tmp/v8_background.log

产出:
  reports/momentum_etf_rotation/combo/v8_periodic_{cost}bp.parquet
  reports/momentum_etf_rotation/combo/v8_probabilistic_{cost}bp.parquet
  reports/momentum_etf_rotation/combo/v8_method_comparison.csv
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
    jump_model_periodic_retrain, compute_features,
)
from v8_probabilistic_experiment import probabilistic_jump_model
from v8_integrated_comparison import load_v7_14_portfolio, compute_integrated_nav
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics

OOS_START = pd.Timestamp('2021-08-01')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_v56():
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v56_expanded_daily.parquet")


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
        if (i + 1) % 5 == 0:
            log(f"  periodic: {i+1}/{len(common_codes)} ETF 完成")
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
        if (i + 1) % 5 == 0:
            log(f"  probabilistic: {i+1}/{len(common_codes)} ETF 完成")
    return signals


def run_single(method, cost_bp, weekly_weights, daily_returns):
    """跑单个组合, 返回 metrics."""
    log(f"=== 开始: {method} 成本={cost_bp}bp ===")
    t0 = time.time()

    if method == 'periodic':
        signals = build_signals_periodic(weekly_weights, daily_returns)
    else:
        signals = build_signals_probabilistic(weekly_weights, daily_returns)

    t_signal = time.time() - t0
    log(f"  信号生成耗时: {t_signal:.0f}s, {len(signals)} ETF")

    nav = compute_integrated_nav(weekly_weights, daily_returns, signals, version='v8_method_b', cost_bp=cost_bp)
    oos = nav.loc[OOS_START:].dropna()
    rets = oos.pct_change().dropna()
    m = compute_metrics(rets, freq='D')

    out_path = OUT_DIR / f"v8_{method}_{cost_bp}bp.parquet"
    nav.to_frame('v8_method_b').to_parquet(out_path)
    log(f"  NAV 已保存: {out_path}")
    log(f"  Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")
    log(f"=== 完成: {method} 成本={cost_bp}bp 耗时={time.time()-t0:.0f}s ===")

    return {
        'method': method, 'cost_bp': cost_bp,
        'Sharpe': m['Sharpe'], 'Calmar': m['Calmar'],
        'AnnRet': m['AnnRet'], 'MaxDD': m['MaxDD'],
        'signal_time_s': t_signal, 'n_etf': len(signals),
    }


def main():
    log("=" * 70)
    log("v8 Jump Model 全方法后台测试 (tmux 版)")
    log("=" * 70)
    log("加载数据...")
    daily_returns = load_v56()
    log(f"v56: {daily_returns.shape}, {daily_returns.index[0].date()} ~ {daily_returns.index[-1].date()}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    combos = []
    for method in ['periodic', 'probabilistic']:
        for cost in [0, 5, 10, 20]:
            combos.append((method, cost))

    log(f"共 {len(combos)} 个组合待跑")
    log("")

    all_results = []
    for i, (method, cost) in enumerate(combos):
        log(f"--- 组合 {i+1}/{len(combos)} ---")
        result = run_single(method, cost, weekly_weights, daily_returns)
        all_results.append(result)
        log("")

    df = pd.DataFrame(all_results)
    csv_path = OUT_DIR / "v8_method_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("=" * 70)
    log("全部完成!")
    log(f"对比表已保存: {csv_path}")
    log("")
    log("=== 总结 (按 Sharpe 排序) ===")
    log(df.sort_values('Sharpe', ascending=False).to_string(index=False))
    log("")
    log("=== 对比 Baseline (有未来函数) ===")
    log("当前 v8_method_b (有未来函数): Sharpe=0.854, Calmar=0.939, MaxDD=-11.66%")
    log("=" * 70)


if __name__ == "__main__":
    main()
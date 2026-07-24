# coding=utf-8
"""v8 Jump Model: probabilistic + Sigmoid 仓位函数验证.

核心改动: 用 Sigmoid 函数替代原线性仓位公式, 大幅降低换手率.

原公式: adj = 1 - (P_bear - 0.25) / 0.75  (P_bear 微小变化触发大幅调仓)
新公式: adj = 1 / (1 + exp((P_bear - threshold) * steepness))

用法:
  python3 scripts/combo/regenerate_v8_sigmoid.py

产出:
  reports/momentum_etf_rotation/combo/v8_sigmoid_t{0.40,0.35,0.45}_s{10,15,10,20}.parquet
  reports/momentum_etf_rotation/combo/v8_sigmoid_comparison.csv
  signals_prob.pkl (复用)
"""
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import compute_features
from v8_probabilistic_experiment import probabilistic_jump_model
from v8_integrated_comparison import load_v7_14_portfolio
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics

OOS_START = pd.Timestamp('2021-08-01')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"
HF_DIR = REPO / "data" / "high_freq_macro"
SIGNAL_PKL = Path(__file__).resolve().parent / "signals_prob.pkl"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_v56():
    return pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")


def sigmoid_adj(P_bear, threshold=0.40, steepness=10):
    """Sigmoid 仓位函数: 范围 (0, 1).
    - P_bear << threshold: adj → 1 (满仓)
    - P_bear >> threshold: adj → 0 (空仓)
    - threshold 附近: 平滑过渡
    """
    x = (P_bear - threshold) * steepness
    return 1.0 / (1.0 + np.exp(x))


def build_or_load_signals(weekly_weights, daily_returns):
    """生成或加载 probabilistic 信号 (pickle 复用)."""
    if SIGNAL_PKL.exists():
        log(f"加载已有信号: {SIGNAL_PKL}")
        with open(SIGNAL_PKL, 'rb') as f:
            signals = pickle.load(f)
        log(f"  信号数量: {len(signals)} ETF")
        return signals

    log("生成 probabilistic 信号 (首次, ~8 分钟)...")
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
        signals[code] = probs_df
        if (i + 1) % 10 == 0:
            log(f"  probabilistic: {i+1}/{len(common_codes)} ETF 完成")

    with open(SIGNAL_PKL, 'wb') as f:
        pickle.dump(signals, f)
    log(f"信号已保存: {SIGNAL_PKL}")
    return signals


def compute_nav_sigmoid(weekly_weights, daily_returns, signals,
                         threshold=0.40, steepness=10, cost_bp=20):
    """用 Sigmoid 仓位函数计算 NAV."""
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建 weekly_bear_pct (用 reindex ffill 对齐)
    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear = bear_pct.reindex(weekly_dates, method='ffill')
            weekly_bear_pct[code] = weekly_bear

    date_to_adjusted_weights = {}

    for i, wd in enumerate(weekly_dates):
        after = all_dates[all_dates > wd]
        if len(after) == 0:
            continue
        start = after[0]
        if i + 1 < len(weekly_dates):
            next_wd = weekly_dates[i + 1]
            before_next = all_dates[all_dates <= next_wd]
            if len(before_next) == 0:
                continue
            end = before_next[-1]
        else:
            end = all_dates[-1]

        # 计算每个资产的 Sigmoid 仓位调整
        adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            if asset in weekly_bear_pct:
                current_bear = weekly_bear_pct[asset].loc[wd]
            else:
                current_bear = 0.0
            if pd.isna(current_bear):
                current_bear = 0.0
            adj_weights[asset] *= sigmoid_adj(current_bear, threshold, steepness)

        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adj_weights.copy()

    # 计算 NAV
    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_adjusted_weights.get(d)
        if w is not None:
            row = daily_returns.loc[d]
            if row[common_codes].isna().all():
                nav.iloc[i] = nav.iloc[i - 1]
            else:
                ret = row.fillna(0.0)
                port_ret = float((w * ret).sum())
                cost_factor = 1.0
                if cost_bp > 0:
                    turnover = float((w - prev_w).abs().sum())
                    cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
                nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
                prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]

    nav_rets = nav.pct_change().dropna()
    cost_drag = nav_rets.mean() * 252
    implied_turnover = cost_drag / (cost_bp / 10000) if cost_bp > 0 else 0

    return nav, implied_turnover


def main():
    log("=" * 70)
    log("v8 probabilistic + Sigmoid 仓位函数验证")
    log("=" * 70)

    log("加载数据...")
    daily_returns = load_v56()
    log(f"v56: {daily_returns.shape}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    signals = build_or_load_signals(weekly_weights, daily_returns)

    # 4 种 Sigmoid 参数
    configs = [
        {'threshold': 0.40, 'steepness': 10, 'name': 'S1'},
        {'threshold': 0.35, 'steepness': 15, 'name': 'S2'},
        {'threshold': 0.45, 'steepness': 10, 'name': 'S3'},
        {'threshold': 0.40, 'steepness': 20, 'name': 'S4'},
    ]

    # 先跑 0bp 基线 (无成本)
    log("\n=== 0bp 基线 (S1, threshold=0.40, steepness=10) ===")
    nav_0bp, _ = compute_nav_sigmoid(
        weekly_weights, daily_returns, signals,
        threshold=0.40, steepness=10, cost_bp=0,
    )
    oos_0bp = nav_0bp.loc[OOS_START:].dropna()
    rets_0bp = oos_0bp.pct_change().dropna()
    m_0bp = compute_metrics(rets_0bp, freq='D')
    log(f"  0bp Sharpe={m_0bp['Sharpe']:.3f} Calmar={m_0bp['Calmar']:.3f} "
        f"AnnRet={m_0bp['AnnRet']:.2%} MaxDD={m_0bp['MaxDD']:.2%}")

    results = []
    for cfg in configs:
        log(f"\n=== {cfg['name']}: threshold={cfg['threshold']}, steepness={cfg['steepness']} ===")
        t0 = time.time()
        nav_20bp, turnover = compute_nav_sigmoid(
            weekly_weights, daily_returns, signals,
            threshold=cfg['threshold'], steepness=cfg['steepness'], cost_bp=20,
        )
        elapsed = time.time() - t0

        oos = nav_20bp.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')

        out_path = OUT_DIR / f"v8_sigmoid_{cfg['name']}_t{cfg['threshold']}_s{cfg['steepness']}.parquet"
        nav_20bp.to_frame('v8_sigmoid').to_parquet(out_path)

        log(f"  Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} "
            f"AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")
        log(f"  隐含换手率: {turnover:.1f}x 耗时: {elapsed:.1f}s")
        log(f"  NAV 已保存: {out_path}")

        results.append({
            'name': cfg['name'],
            'threshold': cfg['threshold'],
            'steepness': cfg['steepness'],
            'cost_bp': 20,
            'Sharpe': m['Sharpe'],
            'Calmar': m['Calmar'],
            'AnnRet': m['AnnRet'],
            'MaxDD': m['MaxDD'],
            'turnover_x': turnover,
            'elapsed_s': elapsed,
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_sigmoid_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("全部完成!")
    log(f"对比表: {csv_path}")
    log("\n=== 总结 ===")
    log(df.to_string(index=False))
    log("\n=== 对比基线 ===")
    log(f"原函数 0bp (线性): Sharpe=0.904, 换手率=47.2x")
    log(f"原函数 20bp (线性): Sharpe=0.354, 换手率≈30x (估算)")
    log("=" * 70)


if __name__ == "__main__":
    main()
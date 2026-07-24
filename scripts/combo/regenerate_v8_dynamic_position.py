# coding=utf-8
"""v8 Jump Model: per-asset 月末调仓 + 动态仓位 risk_scalar.

Phase A: v8 per-asset + v9 银河方案 risk_scalar 整合.

Layer 1 (per-asset sigmoid 月末调仓):
    每周评估每只 ETF 的 P_bear → sigmoid 仓位调整
    修复后 Sharpe 0.871, MaxDD -18.14%

Layer 2 (动态仓位 risk_scalar, 借鉴 v9):
    5 真实宏观因子 → 熵权综合得分 → risk_scalar(t)
    final_position = per_asset_adj × risk_scalar(t)
    期望: 改进 Sharpe 至 0.95+

5 × 4 = 20 组合测试:
    5 风险偏好: R1极保守 / R2标准(v9 默认) / R3温和 / R4激进 / R5保守防御
    4 成本档: 5/10/15/20bp

输入: scripts/combo/signals_prob.pkl (P_bear 信号, 复用)
输出: reports/momentum_etf_rotation/combo/v8_dynamic_position_*.parquet
      reports/momentum_etf_rotation/combo/v8_dynamic_position_comparison.csv
"""
import sys, time, pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from v8_integrated_comparison import load_v7_14_portfolio
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_factor_score,
    compute_risk_scalar,
)

OOS_START = pd.Timestamp('2021-08-01')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"
HF_DIR = REPO / "data" / "high_freq_macro"
SIGNAL_PKL = Path(__file__).resolve().parent / "signals_prob.pkl"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sigmoid_adj(P_bear, threshold=0.50, steepness=10):
    if pd.isna(P_bear):
        return 1.0
    x = (P_bear - threshold) * steepness
    return 1.0 / (1.0 + np.exp(x))


def _get_risk_scalar_for_date(rs_series: pd.Series, wd, weekly_dates, i, clip_low, clip_high):
    """取 wd 当周的 risk_scalar, 回退到上一非空."""
    if wd in rs_series.index:
        v = float(rs_series.loc[wd])
        if np.isnan(v):
            return 1.0
        return float(np.clip(v, clip_low, clip_high))
    if rs_series.notna().any():
        non_nan = rs_series[rs_series.notna()]
        prior = non_nan[non_nan.index <= wd]
        if len(prior) > 0:
            return float(np.clip(prior.iloc[-1], clip_low, clip_high))
    return 1.0


def compute_nav_two_layer(
    weekly_weights, daily_returns, signals, risk_scalar_series,
    sigmoid_threshold=0.50, sigmoid_steepness=10,
    cost_bp=20,
    clip_low=0.3, clip_high=1.5,
):
    """per-asset sigmoid 月末 (Layer 1) × risk_scalar 整体 (Layer 2).

    final_position[d] = per_asset_adj[d] × risk_scalar(wd)

    参数:
        weekly_weights: (T, N) 周频权重 (来自 v7.14)
        daily_returns: (T_d, N) 日频收益 (v56)
        signals: {code: DataFrame with 'P_bear'}
        risk_scalar_series: 周频 risk_scalar (因子 + 熵权综合)
        clip_low/high: risk_scalar 上下限
    """
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear_pct[code] = bear_pct.reindex(weekly_dates, method='ffill')

    date_to_adjusted_weights = {}
    last_ww = None
    last_per_asset_adj = {code: 1.0 for code in common_codes}

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

        is_month_end = (i + 1 >= len(weekly_dates)) or (wd.month != next_wd.month)

        if is_month_end:
            last_ww = weekly_weights.loc[wd].copy()
            for asset in common_codes:
                if asset not in weekly_bear_pct:
                    continue
                p_bear = weekly_bear_pct[asset].loc[wd]
                if pd.isna(p_bear):
                    p_bear = 0.0
                last_per_asset_adj[asset] = sigmoid_adj(p_bear, sigmoid_threshold, sigmoid_steepness)

        if last_ww is not None:
            adj_weights = last_ww.copy()
        else:
            adj_weights = weekly_weights.loc[wd].copy()

        for asset in common_codes:
            if asset in last_per_asset_adj:
                adj_weights[asset] *= last_per_asset_adj[asset]

        # Layer 2: 动态仓位 risk_scalar
        rs = _get_risk_scalar_for_date(risk_scalar_series, wd, weekly_dates, i, clip_low, clip_high)
        adj_weights = adj_weights * rs

        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adj_weights.copy()

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

    return nav


def main():
    log("=" * 70)
    log("Phase A: v8 per-asset + 动态仓位 risk_scalar 整合")
    log("=" * 70)

    log("加载 daily returns...")
    daily_returns = pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")
    log(f"v56: {daily_returns.shape}")

    log("加载 weekly weights (v7.14)...")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    log("加载 P_bear 信号 (signals_prob.pkl)...")
    with open(SIGNAL_PKL, 'rb') as f:
        signals = pickle.load(f)

    log("计算 factor_score (5 宏观因子 + 熵权)...")
    t0 = time.time()
    factor_score = compute_factor_score(daily_returns)
    log(f"  factor_score: {len(factor_score)} 周 ({factor_score.index[0].date()} ~ {factor_score.index[-1].date()})  ({time.time()-t0:.1f}s)")

    log("计算 risk_scalar (default params)...")
    risk_scalar = compute_risk_scalar(factor_score)
    log(f"  risk_scalar: {len(risk_scalar)} 周 ({risk_scalar.index[0].date()} ~ {risk_scalar.index[-1].date()})")

    # 5 风险偏好 × 4 成本档 = 20 组合
    risk_profiles = [
        {'name': 'R1_极保守', 'clip_low': 0.5, 'clip_high': 1.0, 'desc': 'clip_low=0.5 激进时只允许小幅减仓'},
        {'name': 'R2_标准',   'clip_low': 0.3, 'clip_high': 1.5, 'desc': 'v9 默认 [0.3, 1.5]'},
        {'name': 'R3_温和',   'clip_low': 0.4, 'clip_high': 1.3, 'desc': '温和动态 [0.4, 1.3]'},
        {'name': 'R4_激进',   'clip_low': 0.1, 'clip_high': 2.0, 'desc': '极宽 [0.1, 2.0]'},
        {'name': 'R5_保守防御', 'clip_low': 0.6, 'clip_high': 1.2, 'desc': '窄幅 [0.6, 1.2]'},
    ]
    cost_tiers = [5, 10, 15, 20]

    results = []
    for profile in risk_profiles:
        for cost_bp in cost_tiers:
            t0 = time.time()
            nav = compute_nav_two_layer(
                weekly_weights, daily_returns, signals, risk_scalar,
                cost_bp=cost_bp,
                clip_low=profile['clip_low'],
                clip_high=profile['clip_high'],
            )
            elapsed = time.time() - t0

            oos = nav.loc[OOS_START:].dropna()
            rets = oos.pct_change().dropna()

            # 用 v8_integrated_comparison 风格的 ann_ret/vol/Sharpe
            # (避免 v9 compute_metrics 的 rf 减除差异)
            total = oos.iloc[-1] / oos.iloc[0] - 1
            n_years = len(rets) / 252
            ann_ret = (1 + total) ** (1 / max(n_years, 1e-9)) - 1
            vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / vol if vol > 0 else 0.0
            peak = oos.cummax()
            max_dd = float((oos / peak - 1).min())
            calmar = ann_ret / abs(max_dd) if max_dd < -1e-6 else 0.0

            rs_applied = risk_scalar.clip(profile['clip_low'], profile['clip_high'])
            rs_mean = float(rs_applied.mean())
            rs_min = float(rs_applied.min())
            rs_max = float(rs_applied.max())

            log(f"  {profile['name']:14s} cost={cost_bp:2d}bp  "
                f"Sharpe={sharpe:.3f} Calmar={calmar:.3f} "
                f"AnnRet={ann_ret:.2%} MaxDD={max_dd:.2%} "
                f"rs [min,max]=[{rs_min:.2f},{rs_max:.2f}] {elapsed:.1f}s")

            out_path = OUT_DIR / f"v8_dynamic_position_{profile['name']}_C{cost_bp}.parquet"
            nav.to_frame('nav').to_parquet(out_path)

            results.append({
                'profile': profile['name'],
                'clip_low': profile['clip_low'],
                'clip_high': profile['clip_high'],
                'cost_bp': cost_bp,
                'desc': profile['desc'],
                'Sharpe': float(sharpe),
                'Calmar': float(calmar),
                'AnnRet': float(ann_ret),
                'MaxDD': max_dd,
                'rs_mean': rs_mean,
                'rs_min': rs_min,
                'rs_max': rs_max,
            })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_dynamic_position_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("✅ Phase A 完成")
    log(f"对比表: {csv_path}")
    log("=" * 70)

    log("\n=== 5 × 4 = 20 组合 Sharpe 矩阵 ===")
    pivot = df.pivot(index='profile', columns='cost_bp', values='Sharpe')
    log(pivot.to_string())
    log("\n=== 5 × 4 = 20 组合 MaxDD 矩阵 ===")
    pivot_dd = df.pivot(index='profile', columns='cost_bp', values='MaxDD')
    log(pivot_dd.to_string())

    log("\n=== 完整对比表 ===")
    log(df.sort_values(['Sharpe'], ascending=False).to_string(index=False))

    log("\n=== baseline 对比 (Sharpe) ===")
    log(f"  v7.10 TV-PR 5bp               : 0.922")
    log(f"  v8 per-asset 5bp (Phase B 终点): 0.871")
    log(f"  v8 per-asset + dynamic best    : {df['Sharpe'].max():.3f}")


if __name__ == "__main__":
    main()

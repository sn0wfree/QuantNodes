# coding=utf-8
"""scripts/combo/regenerate_v8_optimized_nav.py — 用 v56 重跑 v8 优化版 NAV.

v8 优化版参数 (来自 docs/46 和 scripts/v8_with_smoothing.py):
  - 基于 v8_method_b
  - smooth_weekly_weights(alpha=0.7, min_trade_threshold=0.02)
  - cost_bp = 10
  - bear_threshold = 0.25

输出:
  reports/momentum_etf_rotation/combo/v8_optimized_nav_v56.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 复用 v8 集成框架的函数
from v8_integrated_comparison import (
    load_v7_14_portfolio,
    compute_per_asset_signals,
    compute_integrated_nav,
    compute_position_adjustment,
)


def load_v56_daily_returns() -> pd.DataFrame:
    """加载 v56 日频 ETF 收益 (替代 v7_6)."""
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v56_expanded_daily.parquet")


def smooth_weekly_weights(weights_df, alpha=0.7, min_trade_threshold=0.02):
    """主代码 smooth 函数复制."""
    smoothed = weights_df.copy()
    for t in range(1, len(smoothed)):
        prev_w = smoothed.iloc[t - 1]
        new_w = weights_df.iloc[t]
        blended = alpha * new_w + (1 - alpha) * prev_w
        diff = blended - prev_w
        diff[diff.abs() < min_trade_threshold] = 0.0
        smoothed.iloc[t] = prev_w + diff
    row_sums = smoothed.sum(axis=1)
    mask = row_sums > 1.0
    smoothed.loc[mask] = smoothed.loc[mask].div(row_sums[mask], axis=0)
    return smoothed


def compute_optimized_nav(weekly_weights, daily_returns, signals,
                          alpha=0.7, threshold=0.02, cost_bp=10.0):
    """v8 优化版: v8_method_b + smooth + 10bp 成本."""
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建调整后权重 (v8_method_b 逻辑)
    date_to_adjusted_weights = {}
    adjusted_w_list = []
    weekly_dates_used = []
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

        # v8_method_b 调整
        adj = compute_position_adjustment("v8_method_b", signals, wd)
        adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            if asset in adj:
                adj_weights[asset] *= adj[asset]
        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        adjusted_w_list.append(adj_weights)
        weekly_dates_used.append(wd)

    adjusted_weights = pd.DataFrame(adjusted_w_list, index=weekly_dates_used)

    # 应用平滑 (smooth)
    adjusted_weights = smooth_weekly_weights(
        adjusted_weights, alpha=alpha, min_trade_threshold=threshold,
    )

    # 构建 date → weights 映射
    for i, wd in enumerate(weekly_dates_used):
        if i + 1 < len(weekly_dates_used):
            next_wd = weekly_dates_used[i + 1]
            before_next = all_dates[all_dates <= next_wd]
            if len(before_next) == 0:
                continue
            end = before_next[-1]
        else:
            end = all_dates[-1]

        after = all_dates[all_dates > wd]
        if len(after) == 0:
            continue
        start = after[0]

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adjusted_weights.loc[wd]

    # 计算日频 NAV (含 10bp 成本 + 中国假期处理)
    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_adjusted_weights.get(d)
        if w is not None:
            row = daily_returns.loc[d]
            # 中国假期判断: ETF 收益全 NaN 当日, 跳过
            if row[common_codes].isna().all():
                nav.iloc[i] = nav.iloc[i - 1]
            else:
                ret = row.fillna(0.0)
                port_ret = float((w * ret).sum())
                # 成本
                turnover = float((w - prev_w).abs().sum())
                cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
                nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
                prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]

    return nav


def compute_metrics(nav, freq="D"):
    """日频或周频 NAV 计算 Sharpe/Calmar/MaxDD."""
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return {}
    freq_map = {"D": 252, "W": 52}
    periods = freq_map.get(freq, 252)
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (periods / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(periods)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0
    dd = (nav / nav.cummax() - 1).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 1e-10 else 0
    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "max_dd": dd,
    }


def main():
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("重新生成 v8 Jump Model 优化版 NAV (基于 v56)")
    print("=" * 70)

    # 1. 加载 v56 日频 ETF 收益
    print("\n[Step 1] 加载 v56 日频 ETF 收益")
    daily_returns = load_v56_daily_returns()
    print(f"  v56 shape: {daily_returns.shape}")

    # 2. 加载 v7.14 TV-PR 周频权重
    print("\n[Step 2] 加载 v7.14 TV-PR 周频权重")
    weekly_weights, prices, shares = load_v7_14_portfolio()
    print(f"  weekly_weights: {weekly_weights.shape}")

    # 3. 计算每个资产的 Jump Model 信号
    print("\n[Step 3] 计算每资产 Jump Model 信号")
    signals = compute_per_asset_signals(weekly_weights, daily_returns)
    print(f"  共 {len(signals)} 个资产有信号")

    # 4. 计算 v8 优化版 NAV (alpha=0.7, threshold=0.02, cost=10bp)
    print("\n[Step 4] 计算 v8 优化版 NAV")
    nav = compute_optimized_nav(
        weekly_weights, daily_returns, signals,
        alpha=0.7, threshold=0.02, cost_bp=10.0,
    )

    # OOS 指标
    oos = nav.loc['2021-08-01':'2026-06-30']
    m = compute_metrics(oos, freq='D')
    print(f"  v8 优化版 OOS (2021-08~2026-06):")
    print(f"  Sharpe={m['sharpe']:.3f}, Calmar={m['calmar']:.3f}, "
          f"AnnRet={m['ann_ret']:.2%}, MaxDD={m['max_dd']:.2%}")

    # 5. 保存
    print("\n[Step 5] 保存 v8 优化版 NAV parquet")
    out_path = out_dir / "v8_optimized_nav_v56.parquet"
    nav.to_frame('v8 Jump Model 优化版 (v56)').to_parquet(out_path)
    print(f"  保存: {out_path}")
    print(f"  shape: {nav.shape}")

    # 同时保存一个 CSV 方便对比
    df_metrics = pd.DataFrame([{
        'version': 'v8_optimized',
        'alpha': 0.7,
        'threshold': 0.02,
        'cost_bp': 10.0,
        'sharpe': m['sharpe'],
        'calmar': m['calmar'],
        'ann_ret': m['ann_ret'],
        'ann_vol': m['ann_vol'],
        'max_dd': m['max_dd'],
    }])
    csv_path = out_dir / "v8_optimized_v56_results.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n保存: {csv_path}")

    print("\n" + "=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()
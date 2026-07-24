# coding=utf-8
"""scripts/combo/regenerate_v7_14_nav_with_v56.py — 用 v56 数据重新生成 v7.14 NAV.

目的: 让 v7.14 和 v9 用同一份数据 (v56_expanded_daily.parquet) 做公平对比.

v7.14 之前用 v7_6_daily_etf_returns.parquet (简单收益 + 跳中国假期),
   跟 v9 用的 v56_expanded_daily.parquet (对数收益 + 含中国假期) 不一致.

输出:
  reports/momentum_etf_rotation/combo/v7_14_nav_v56.parquet
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

# 复用 v8 集成框架的 v7.14 加载 + NAV 计算
from v8_integrated_comparison import load_v7_14_portfolio


def load_v56_daily_returns() -> pd.DataFrame:
    """加载 v56 日频 ETF 收益 (替代 v7_6)."""
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v56_expanded_daily.parquet")


def compute_daily_nav(weekly_weights, daily_returns, cost_bp=0.0):
    """与 v8/integration.py 一致的日频 NAV 计算逻辑."""
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    date_to_weights = {}
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

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_weights[d] = weekly_weights.loc[wd]

    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_weights.get(d)
        if w is not None:
            row = daily_returns.loc[d]
            # 中国假期判断: ETF 收益全 NaN 当日, 跳过 (与 v7_6 数据一致)
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
    print("重新生成 v7.14 TV-PR NAV (基于 v56 数据)")
    print("=" * 70)

    # 1. 加载 v56 日频 ETF 收益
    print("\n[Step 1] 加载 v56 日频 ETF 收益")
    daily_returns = load_v56_daily_returns()
    print(f"  v56 shape: {daily_returns.shape}")

    # 2. 加载 v7.14 TV-PR 周频权重
    print("\n[Step 2] 加载 v7.14 TV-PR 周频权重")
    weekly_weights, prices, shares = load_v7_14_portfolio()
    print(f"  weekly_weights: {weekly_weights.shape}")

    # 3. 跑 v7.14 日频 NAV (3 个成本档)
    print("\n[Step 3] 计算 v7.14 日频 NAV")
    navs = {}
    for cost_bp in [0, 5, 10]:
        nav = compute_daily_nav(weekly_weights, daily_returns, cost_bp)
        col = f"v7.14 TV-PR (v56, cost={cost_bp}bp)"
        navs[col] = nav
        # OOS 指标
        oos = nav.loc['2021-08-01':'2026-06-30']
        m = compute_metrics(oos, freq="D")
        print(f"  cost={cost_bp}bp: Sharpe={m['sharpe']:.3f} Calmar={m['calmar']:.3f} "
              f"AnnRet={m['ann_ret']:.2%} MaxDD={m['max_dd']:.2%}")

    # 4. 保存
    print("\n[Step 4] 保存 v7.14 NAV parquet")
    out_df = pd.DataFrame(navs)
    out_path = out_dir / "v7_14_nav_v56.parquet"
    out_df.to_parquet(out_path)
    print(f"  保存: {out_path}")
    print(f"  shape: {out_df.shape}")

    # 5. 完整指标对比 (新数据 vs 旧数据)
    print("\n[Step 5] 对比新旧数据")
    old = pd.read_parquet(out_dir / "v7_14_nav.parquet")
    print(f"  旧数据 NAV shape: {old.shape}, 起点 {old.iloc[0, 0]:.4f}")
    print(f"  新数据 NAV shape: {out_df.shape}, 起点 {out_df.iloc[0, 0]:.4f}")

    old_oos = old.iloc[:, 0].loc['2021-08-01':'2026-06-30']
    new_oos = out_df['v7.14 TV-PR (v56, cost=0bp)'].loc['2021-08-01':'2026-06-30']

    m_old = compute_metrics(old_oos, freq="D")
    m_new = compute_metrics(new_oos, freq="D")

    print("\n=== OOS (2021-08~2026-06) 指标对比 ===")
    print(f"{'指标':15s} {'旧 (v7_6)':>15s} {'新 (v56)':>15s} {'差异':>15s}")
    for k in ['sharpe', 'calmar', 'ann_ret', 'max_dd']:
        o = m_old[k]
        n = m_new[k]
        diff = n - o
        if k in ['ann_ret', 'max_dd']:
            print(f"{k:15s} {o:>15.2%} {n:>15.2%} {diff:>+15.2%}")
        else:
            print(f"{k:15s} {o:>15.3f} {n:>15.3f} {diff:>+15.3f}")

    print("\n" + "=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()
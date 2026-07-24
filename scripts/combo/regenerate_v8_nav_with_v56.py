# coding=utf-8
"""scripts/combo/regenerate_v8_nav_with_v56.py — 用 v56 数据重新生成 v8 NAV.

目的: 让 v8 和 v9 用同一份数据 (v56_expanded_daily.parquet) 做公平对比.

步骤:
  1. 加载 v56 (日频 ETF 收益)
  2. 复用 v8 集成策略框架 (scripts/v8_integrated_comparison.py)
  3. 计算 v8_method_b 集成 NAV
  4. 保存到 reports/momentum_etf_rotation/combo/v8_method_b_nav_v56.parquet

输出: v8 Jump Model 方案B 基于 v56 数据的日频 NAV.
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

# 复用 v8 框架的函数
from v8_integrated_comparison import (
    load_v7_14_portfolio,
    compute_per_asset_signals,
    compute_integrated_nav,
    VERSIONS,
    COST_CANDIDATES,
    JUMP_PENALTY,
    N_RESTARTS,
    N_ITER,
    RANDOM_STATE,
)


def load_v56_daily_returns() -> pd.DataFrame:
    """加载 v56 日频 ETF 收益 (替代 v7_6)."""
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v56_expanded_daily.parquet")


def main():
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("重新生成 v8 Jump Model NAV (基于 v56 数据)")
    print("=" * 70)

    # 1. 加载 v56 日频 ETF 收益
    print("\n[Step 1] 加载 v56 日频 ETF 收益")
    daily_returns = load_v56_daily_returns()
    print(f"  v56 shape: {daily_returns.shape}")
    print(f"  范围: {daily_returns.index[0].date()} ~ {daily_returns.index[-1].date()}")

    # 2. 加载 v7.14 TV-PR 周频权重
    print("\n[Step 2] 加载 v7.14 TV-PR 周频权重")
    weekly_weights, prices, shares = load_v7_14_portfolio()
    print(f"  weekly_weights: {weekly_weights.shape}")

    # 3. 计算每个资产的 Jump Model 信号
    print("\n[Step 3] 计算每资产 Jump Model 信号")
    signals = compute_per_asset_signals(weekly_weights, daily_returns)
    print(f"  共 {len(signals)} 个资产有信号")

    # 4. 计算 v8_method_b 集成 NAV (无成本 + 10bp 成本)
    print("\n[Step 4] 计算 v8_method_b 集成 NAV")
    results = []
    navs = {}
    for cost_bp in COST_CANDIDATES:
        for version in VERSIONS:
            nav = compute_integrated_nav(
                weekly_weights, daily_returns, signals, version, cost_bp
            )
            navs[(version, cost_bp)] = nav
            # 计算 OOS 指标 (2021-08 ~ 2026-06)
            oos = nav.loc['2021-08-01':'2026-06-30']
            rets = oos.pct_change().dropna()
            ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0
            dd = (oos / oos.cummax() - 1).min()
            calmar = ann_ret / abs(dd) if abs(dd) > 1e-10 else 0
            results.append({
                'version': version,
                'cost_bp': cost_bp,
                'ann_ret': ann_ret,
                'ann_vol': ann_vol,
                'sharpe': sharpe,
                'max_dd': dd,
                'calmar': calmar,
            })
            print(f"  {version} (cost={cost_bp}bp): "
                  f"Sharpe={sharpe:.3f} Calmar={calmar:.3f} "
                  f"AnnRet={ann_ret:.2%} MaxDD={dd:.2%}")

    # 5. 保存 v8_method_b NAV (无成本版本, 用于对比)
    print("\n[Step 5] 保存 v8_method_b NAV (基于 v56)")
    df_results = pd.DataFrame(results)
    print("\n=== 各版本 OOS 指标 (2021-08-01 ~ 2026-06-30) ===")
    print(df_results.to_string(index=False))

    # 保存 v8_method_b (无成本) NAV
    nav_v8_mb = navs[('v8_method_b', 0.0)]
    out_path = out_dir / "v8_method_b_nav_v56.parquet"
    nav_v8_mb.to_frame('v8 Jump Model 方案B (v56)').to_parquet(out_path)
    print(f"\n保存: {out_path}")
    print(f"  shape: {nav_v8_mb.shape}")

    # 同时保存一个包含所有版本 NAV 的 parquet
    all_navs = pd.DataFrame()
    for (version, cost_bp), nav in navs.items():
        col = f"{version}_cost{cost_bp}bp"
        all_navs[col] = nav
    all_navs_path = out_dir / "v8_all_navs_v56.parquet"
    all_navs.to_parquet(all_navs_path)
    print(f"\n保存: {all_navs_path}")
    print(f"  shape: {all_navs.shape}")

    # 保存结果 CSV
    results_path = out_dir / "v8_v56_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n保存: {results_path}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
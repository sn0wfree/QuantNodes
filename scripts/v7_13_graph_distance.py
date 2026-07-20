#!/usr/bin/env python3
# coding=utf-8
"""v7.13 图谱距离因子生成 + IC 测试.

v7.13 = v7.10 (36 因子) + 10 图谱距离因子 = 46 因子.

三个组件:
  A. DCC Z-Score (每对资产, 截面): dcc_z_mean, dcc_z_max
  B. 网络拓扑 (7 维, 时序): avg_path, clustering_coeff, centrality_entropy,
                            density, largest_comp, spectral_radius, modularity
  C. 跨类别尾部依赖 (时序): cross_class_tail_dep

用法:
  python scripts/v7_13_graph_distance.py
  python scripts/v7_13_graph_distance.py --ic-only
  python scripts/v7_13_graph_distance.py --fast  # 快速版 DCC Z-Score
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data,
    load_daily_etf_returns,
    EXPANDED_COLS,
    EXPANDED_BOND_INDICES,
    EQUITY_ETF_COLS,
    COMMODITY_ETF_COLS,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.graph_distance_factors import (
    compute_all_graph_distance_factors,
    get_graph_distance_factor_names,
)

HF_DIR = REPO / "data" / "high_freq_macro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true", help="只计算 IC")
    parser.add_argument("--fast", action="store_true", help="快速版 DCC Z-Score")
    args = parser.parse_args()

    print("=" * 60)
    print("v7.13 = v7.10 + 10 图谱距离因子")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X_v710, Y, codes = load_v7_10_data()
    daily_returns = load_daily_etf_returns()
    daily_returns = daily_returns[[c for c in codes if c in daily_returns.columns]]
    print(f"  v7.10: {X_v710.shape}, Y: {Y.shape}")
    print(f"  日频: {daily_returns.shape}")

    # 权益/非权益分类
    equity_codes = [c for c in EQUITY_ETF_COLS if c in codes]
    bond_codes = [c for c in EXPANDED_BOND_INDICES if c in codes]
    commodity_codes = [c for c in COMMODITY_ETF_COLS if c in codes]
    non_equity_codes = bond_codes + commodity_codes
    print(f"  权益: {len(equity_codes)}, 非权益: {len(non_equity_codes)}")

    # 2. 计算图谱距离因子
    print("\n[2/5] 计算图谱距离因子...")
    t0 = time.time()
    factors = compute_all_graph_distance_factors(
        daily_returns, equity_codes, non_equity_codes, fast_mode=args.fast,
    )
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s")

    # 3. 对齐到 v7.10 时间索引
    print("\n[3/5] 对齐到周频...")
    Y_index = Y.index
    T = len(Y_index)
    N = len(codes)

    # 分类因子
    factor_groups = get_graph_distance_factor_names()
    all_names = []
    for group_names in factor_groups.values():
        all_names.extend(group_names)

    # 构建面板
    K_new = len(all_names)
    X_new = np.full((T, N, K_new), np.nan)

    for k, fname in enumerate(all_names):
        data = factors[fname]
        if isinstance(data, pd.DataFrame):
            # 截面因子 (每个资产不同)
            for i, target_date in enumerate(Y_index):
                diffs = abs(data.index - target_date)
                if len(diffs) > 0:
                    closest_idx = diffs.argmin()
                    if diffs[closest_idx].days <= 7:
                        for j, code in enumerate(codes):
                            if code in data.columns:
                                X_new[i, j, k] = data.iloc[closest_idx][code]
        elif isinstance(data, pd.Series):
            # 时序因子 (所有资产相同)
            for i, target_date in enumerate(Y_index):
                diffs = abs(data.index - target_date)
                if len(diffs) > 0:
                    closest_idx = diffs.argmin()
                    if diffs[closest_idx].days <= 7:
                        X_new[i, :, k] = data.iloc[closest_idx]

    print(f"  图谱距离面板: {X_new.shape}")

    # 4. 计算 IC
    print("\n[4/5] 计算 IC...")
    from scipy.stats import spearmanr, pearsonr

    Y_shifted = Y.shift(-1).iloc[:-1].values
    X_new_shifted = X_new[:-1]
    market_ret = Y.shift(-1).iloc[:-1].mean(axis=1).values

    # 截面因子 IC
    print(f"\n  截面因子 IC (每个资产不同):")
    print(f"  {'因子':<25} {'IC_mean':<10} {'IC_std':<10} {'ICIR':<10} {'pct_pos':<10}")
    print(f"  {'-'*65}")

    for k, fname in enumerate(all_names):
        # 检查是否时序因子
        data = factors[fname]
        is_ts = isinstance(data, pd.Series)
        if is_ts:
            continue  # 跳过时序因子, 后面单独处理

        ic_list = []
        for t in range(52, T - 1):
            x_t = X_new_shifted[t, :, k]
            y_t = Y_shifted[t]
            valid = ~np.isnan(x_t) & ~np.isnan(y_t)
            if valid.sum() > 10:
                corr, _ = spearmanr(x_t[valid], y_t[valid])
                ic_list.append(corr)

        if ic_list:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            pct_pos = sum(1 for x in ic_list if x > 0) / len(ic_list)
            print(f"  {fname:<25} {ic_mean:<10.4f} {ic_std:<10.4f} {icir:<10.4f} {pct_pos:<10.2%}")

    # 时序因子 IC
    print(f"\n  时序因子 IC (所有资产相同):")
    print(f"  {'因子':<25} {'IC (Pearson)':<14} {'p-value':<10} {'显著?':<8}")
    print(f"  {'-'*57}")

    for k, fname in enumerate(all_names):
        data = factors[fname]
        if not isinstance(data, pd.Series):
            continue

        factor_ts = X_new_shifted[52:T-1, 0, k]
        mkt_ts = market_ret[52:T-1]
        valid = ~np.isnan(factor_ts) & ~np.isnan(mkt_ts)
        if valid.sum() > 10:
            corr, pval = pearsonr(factor_ts[valid], mkt_ts[valid])
            sig = "✓" if pval < 0.05 else "✗"
            print(f"  {fname:<25} {corr:<14.4f} {pval:<10.4f} {sig}")

    # 5. 保存
    if not args.ic_only:
        print("\n[5/5] 保存 v7.13 数据...")
        X_v713 = np.concatenate([X_v710, X_new], axis=2)

        v710_names = (HF_DIR / "v7_10_factor_names.csv").read_text().strip().split("\n")[1:]
        v713_names = v710_names + all_names

        np.save(HF_DIR / "v7_13_X_panel.npy", X_v713)
        Y.to_parquet(HF_DIR / "v7_13_Y_weekly.parquet")
        (HF_DIR / "v7_13_codes.csv").write_text("\n".join(["code"] + codes))
        (HF_DIR / "v7_13_factor_names.csv").write_text("\n".join(["factor"] + v713_names))

        print(f"  v7_13_X_panel.npy: {X_v713.shape}")
        print(f"  v7_13_factor_names.csv: {len(v713_names)} factors")
    else:
        print("\n(--ic-only, 跳过保存)")


if __name__ == "__main__":
    main()

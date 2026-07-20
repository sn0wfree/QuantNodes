#!/usr/bin/env python3
# coding=utf-8
"""v7.11 数据生成 + IC 测试.

生成 v7.11 数据 (36 原有 + 10 新增 = 46 因子), 计算每个新因子的 IC.

用法:
  python scripts/v7_11_gen_data.py
  python scripts/v7_11_gen_data.py --ic-only  # 只计算 IC, 不重新生成数据
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
)
from QuantNodes.strategy.momentum_etf_rotation.v7.enhanced_factors_v7_11 import (
    compute_all_v7_11_factors,
    resample_factors_to_weekly,
    get_factor_names,
)

HF_DIR = REPO / "data" / "high_freq_macro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true", help="只计算 IC")
    args = parser.parse_args()

    print("=" * 60)
    print("v7.11 数据生成 + IC 测试")
    print("=" * 60)

    # 1. 加载 v7.10 数据
    print("\n[1/4] 加载 v7.10 数据...")
    X_v710, Y, codes = load_v7_10_data()
    print(f"  X: {X_v710.shape}, Y: {Y.shape}, codes: {len(codes)}")

    # 2. 加载日频数据
    print("\n[2/4] 加载日频数据...")
    daily_returns = load_daily_etf_returns()
    # 计算日频收盘价 (从收益累积)
    daily_close = (1 + daily_returns).cumprod()
    print(f"  daily_returns: {daily_returns.shape}")

    # 3. 计算 10 个新因子
    print("\n[3/4] 计算 10 个新因子 (日频 → 周频)...")
    t0 = time.time()
    daily_factors = compute_all_v7_11_factors(daily_returns, daily_close)
    weekly_factors = resample_factors_to_weekly(daily_factors)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # 对齐到 v7.10 的时间索引
    Y_index = Y.index
    new_factor_names = get_factor_names()

    # 构建新因子面板 (T, N, 10)
    T = len(Y_index)
    N = len(codes)
    K_new = len(new_factor_names)
    X_new = np.full((T, N, K_new), np.nan)

    for k, fname in enumerate(new_factor_names):
        df = weekly_factors[fname]
        # 对齐: 用最近的周频值 (容差 7 天)
        for i, target_date in enumerate(Y_index):
            # 找到最接近的日期
            diffs = abs(df.index - target_date)
            if len(diffs) > 0:
                closest_idx = diffs.argmin()
                if diffs[closest_idx].days <= 7:
                    for j, code in enumerate(codes):
                        if code in df.columns:
                            X_new[i, j, k] = df.iloc[closest_idx][code]

    print(f"  新因子面板: {X_new.shape}")

    # 4. 计算 IC
    print("\n[4/4] 计算 IC...")
    from scipy.stats import spearmanr

    # 时序因子列表 (所有资产值相同, 截面 IC 无意义)
    TIME_SERIES_FACTORS = {"return_dispersion"}

    # Y_shifted[t] = Y[t+1] (t→t+1 收益)
    Y_shifted = Y.shift(-1).iloc[:-1].values
    X_new_shifted = X_new[:-1]
    market_ret = Y.shift(-1).iloc[:-1].mean(axis=1).values  # 市场平均收益

    print(f"\n  截面因子 (每个资产不同):")
    print(f"  {'因子':<20} {'IC_mean':<10} {'IC_std':<10} {'ICIR':<10} {'pct_pos':<10}")
    print(f"  {'-'*60}")

    for k, fname in enumerate(new_factor_names):
        if fname in TIME_SERIES_FACTORS:
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
            print(f"  {fname:<20} {ic_mean:<10.4f} {ic_std:<10.4f} {icir:<10.4f} {pct_pos:<10.2%}")

    # 时序因子 IC: factor[t] vs market_return[t+1] (时间序列相关)
    print(f"\n  时序因子 (所有资产相同, 用时间序列 IC):")
    print(f"  {'因子':<20} {'IC (Pearson)':<14} {'p-value':<10}")
    print(f"  {'-'*44}")
    for k, fname in enumerate(new_factor_names):
        if fname not in TIME_SERIES_FACTORS:
            continue
        # 取第一个资产的值 (时序因子所有资产相同)
        factor_ts = X_new_shifted[52:T-1, 0, k]
        mkt_ts = market_ret[52:T-1]
        valid = ~np.isnan(factor_ts) & ~np.isnan(mkt_ts)
        if valid.sum() > 10:
            from scipy.stats import pearsonr
            corr, pval = pearsonr(factor_ts[valid], mkt_ts[valid])
            print(f"  {fname:<20} {corr:<14.4f} {pval:<10.4f}")
        else:
            print(f"  {fname:<20} {'N/A':<10}")

    # 5. 保存数据
    if not args.ic_only:
        print("\n[5/5] 保存 v7.11 数据...")
        # 合并 v7.10 (36) + 新因子 (10) = 46
        X_v711 = np.concatenate([X_v710, X_new], axis=2)

        # 因子名
        v710_names = (HF_DIR / "v7_10_factor_names.csv").read_text().strip().split("\n")[1:]
        v711_names = v710_names + new_factor_names

        np.save(HF_DIR / "v7_11_X_panel.npy", X_v711)
        Y.to_parquet(HF_DIR / "v7_11_Y_weekly.parquet")
        (HF_DIR / "v7_11_codes.csv").write_text("\n".join(["code"] + codes))
        (HF_DIR / "v7_11_factor_names.csv").write_text("\n".join(["factor"] + v711_names))

        print(f"  v7_11_X_panel.npy: {X_v711.shape}")
        print(f"  v7_11_Y_weekly.parquet: {Y.shape}")
        print(f"  v7_11_factor_names.csv: {len(v711_names)} factors")
    else:
        print("\n(--ic-only, 跳过保存)")


if __name__ == "__main__":
    main()

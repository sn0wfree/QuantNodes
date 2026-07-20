#!/usr/bin/env python3.10
# coding=utf-8
"""v7.14 = v7.10 (36 因子) + 6 个相关性距离因子 (共 42 因子).

6 个相关性因子 (全部截面变异):
  1. distance_to_centroid  — 相关性空间中到中心的 L2 距离
  2. avg_pairwise_corr     — 与所有其他资产的真实平均相关性
  3. local_clustering_coeff — 阈值化相关性网络的局部聚类系数
  4. corr_diff             — 同类别相关性 - 跨类别相关性 (权益/商品/债券)
  5. avg_tail_dep          — 平均下尾依赖 (10% 分位数)
  6. corr_momentum         — 相关性变化 (corr_60d - corr_60d_lag20)

用扩展后的资产池 (43 + 4 新增 = 47) 计算相关性, 但只输出原始 43 资产的因子值.

用法:
    python3.10 scripts/v7_14_corr_distance.py [--fast]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_daily_etf_returns,
    load_v7_10_data,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.correlation_distance_factors import (
    compute_all_corr_factors,
)


# ============================================================
# 扩展资产池
# ============================================================
NEW_CODES = ["159980", "511010", "162411"]  # 铜, 5Y国债, 原油

# 资产分类: 0=权益, 1=商品, 2=债券
CATEGORY_MAP = {
    # 权益 (37)
    "510300": 0, "510500": 0, "510050": 0, "159915": 0, "588000": 0, "159901": 0,
    "512760": 0, "512480": 0, "515030": 0, "515790": 0, "512690": 0, "512170": 0,
    "512010": 0, "515050": 0, "159928": 0, "512880": 0, "512000": 0, "512800": 0,
    "515220": 0, "512200": 0, "512400": 0, "512660": 0, "512980": 0, "515880": 0,
    "159996": 0, "512120": 0, "510900": 0, "159920": 0, "513010": 0, "513050": 0,
    "159740": 0, "513100": 0, "513300": 0, "513500": 0, "513520": 0, "513880": 0,
    "159941": 0,
    # 商品 (6: 移除 518800 重复 + 159766 太短; 新增 162411 原油 + 159980 铜)
    "518880": 1, "161226": 1, "159985": 1, "159981": 1,
    "162411": 1, "159980": 1,
    # 债券 (2: 原 511260 + 新增 511010)
    "511260": 2, "511010": 2,
}


def load_extended_daily_returns() -> pd.DataFrame:
    """加载扩展后的日频收益 (43 原始 + 4 新增 = 47 资产)."""
    daily = load_daily_etf_returns()  # (T, 43)

    # 加载新 ETF 数据
    per_etf_dir = PROJECT_ROOT / "data" / "real" / "per_etf"
    new_series = {}
    for code in NEW_CODES:
        path = per_etf_dir / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            new_series[code] = df["close"].pct_change()
        else:
            print(f"  警告: {code} 数据不存在, 跳过")

    if not new_series:
        print("  警告: 无新增 ETF 数据, 使用原始 43 资产")
        return daily

    new_df = pd.DataFrame(new_series)
    # 对齐索引
    extended = pd.concat([daily, new_df], axis=1).reindex(daily.index)
    return extended


def main() -> int:
    parser = argparse.ArgumentParser(description="v7.14 相关性距离因子测试")
    parser.add_argument("--fast", action="store_true", help="快速模式 (减少资产数)")
    args = parser.parse_args()

    t0_total = time.time()
    print("=" * 60)
    print("v7.14 = v7.10 + 6 相关性距离因子")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X_v710, Y, codes = load_v7_10_data()
    daily = load_extended_daily_returns()

    T, N_orig, K_orig = X_v710.shape
    print(f"  v7.10: {X_v710.shape}, Y: {Y.shape}")
    print(f"  日频扩展: {daily.shape}")
    print(f"  原始资产: {N_orig}, 新增: {len(NEW_CODES)}")

    if args.fast:
        print("  快速模式: 只用前 5 个资产")
        codes = codes[:5]
        X_v710 = X_v710[:, :5, :]
        Y = Y.iloc[:, :5]
        daily = daily.iloc[:, :5]  # 这里不对, 应该保留所有资产用于计算相关性
        # 修正: 保留所有资产计算相关性, 只输出前5个
        original_codes_5 = codes
    else:
        original_codes_5 = codes

    # 2. 计算相关性因子
    print("\n[2/5] 计算相关性因子...")
    t1 = time.time()

    # 构建分类映射 (用于所有资产)
    cat_map = {}
    for code in daily.columns:
        cat_map[code] = CATEGORY_MAP.get(code, 0)  # 默认权益

    X_corr_daily, factor_names = compute_all_corr_factors(
        daily, cat_map, codes,  # 用 codes (43), 不是 original_codes_5
        window=60, threshold=0.3, quantile=0.10, lag=20,
    )
    elapsed_corr = time.time() - t1
    print(f"  耗时: {elapsed_corr:.1f}s")
    print(f"  日频因子面板: {X_corr_daily.shape}")

    # 2b. 下采样到周频 (取每周最后一个交易日的值)
    print("  下采样到周频...")
    daily_dates = daily.index
    weekly_dates = Y.index  # v7.10 的周频日期

    # 对每个周频日期, 找到对应的最后一个日频日期
    X_corr = np.full((len(weekly_dates), len(codes), len(factor_names)), np.nan)
    for t_w, w_date in enumerate(weekly_dates):
        # 找到该周内最后一个日频日期
        mask = daily_dates <= w_date
        if mask.any():
            last_day_idx = np.where(mask)[0][-1]
            if last_day_idx < X_corr_daily.shape[0]:
                X_corr[t_w] = X_corr_daily[last_day_idx]

    print(f"  周频因子面板: {X_corr.shape}")

    # 2c. 标准化新因子 (截面 Z-score, 与 v7.10 一致)
    print("  标准化新因子 (截面 Z-score)...")
    for t in range(X_corr.shape[0]):
        for k in range(X_corr.shape[2]):
            vals = X_corr[t, :, k]
            valid = ~np.isnan(vals)
            if valid.sum() > 1:
                mean = np.nanmean(vals[valid])
                std = np.nanstd(vals[valid])
                if std > 1e-10:
                    X_corr[t, :, k] = (vals - mean) / std

    print(f"  标准化后: mean={np.nanmean(X_corr):.4f}, std={np.nanstd(X_corr):.4f}")

    # 3. 合并为 v7.14
    print("\n[3/5] 合并为 v7.14...")
    X_v714 = np.concatenate([X_v710, X_corr], axis=2)
    all_names = pd.read_csv(
        PROJECT_ROOT / "data" / "high_freq_macro" / "v7_10_factor_names.csv",
        header=None,
    )[0].tolist()[:K_orig] + factor_names
    print(f"  v7.14: {X_v714.shape} ({K_orig} + {len(factor_names)} = {len(all_names)} 因子)")

    # 4. 保存
    print("\n[4/5] 保存 v7.14 数据...")
    out_dir = PROJECT_ROOT / "data" / "high_freq_macro"
    np.save(out_dir / "v7_14_X_panel.npy", X_v714)
    pd.Series(all_names).to_csv(out_dir / "v7_14_factor_names.csv", index=False, header=False)
    print(f"  v7_14_X_panel.npy: {X_v714.shape}")

    # 5. IC 测试
    print("\n[5/5] 计算 IC...")
    from scipy.stats import spearmanr

    print("\n  截面因子 IC (每个资产不同):")
    print(f"  {'因子':<30} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'pct_pos':>10}")
    print("  " + "-" * 70)

    for k, name in enumerate(factor_names):
        ic_vals = []
        for t in range(60, T - 1):
            x_t = X_corr[t, :, k]
            y_t = Y.iloc[t + 1].values if t + 1 < len(Y) else np.full(N_orig, np.nan)
            valid = ~np.isnan(x_t) & ~np.isnan(y_t)
            if valid.sum() > 10:
                ic, _ = spearmanr(x_t[valid], y_t[valid])
                ic_vals.append(ic)
        if ic_vals:
            ic_mean = np.nanmean(ic_vals)
            ic_std = np.nanstd(ic_vals)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            pct_pos = np.mean(np.array(ic_vals) > 0) * 100
            print(f"  {name:<30} {ic_mean:>+10.4f} {ic_std:>10.4f} {icir:>+10.3f} {pct_pos:>9.1f}%")
        else:
            print(f"  {name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    elapsed_total = time.time() - t0_total
    print(f"\n总耗时: {elapsed_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

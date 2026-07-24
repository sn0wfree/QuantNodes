#!/usr/bin/env python3
# coding=utf-8
"""v7.7 数据重新生成 (修复 look-ahead bias).

原问题: v7.7 数据中 X[t] 和 Y[t] 都包含 t-1→t 的收益信息, 导致 f18_mom_short 与目标相关 0.96.
修复: Y[t] = t→t+1 的收益 (shift -1), 确保 X[t] 只用 t-1 之前的信息预测 t→t+1 的收益.

用法:
  python scripts/v7_7_regen_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_6_data,
    load_weekly_asset_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.enhanced_factors import (
    EnhancedFactorConfig,
)

HF_DIR = REPO / "data" / "high_freq_macro"


def main():
    print("=" * 60)
    print("v7.7 数据重新生成 (修复 look-ahead)")
    print("=" * 60)

    # 1. 加载 v7.6 数据 (39 因子, 包含增强因子)
    print("\n[1/5] 加载 v7.6 数据...")
    X_panel, Y, codes = load_v7_6_data(
        macro_use_log_return=True,
        standardize=None,  # 不标准化, 保持原始值
    )
    print(f"  X: {X_panel.shape}, Y: {Y.shape}, codes: {len(codes)}")

    # 2. 获取因子名
    from QuantNodes.strategy.momentum_etf_rotation.v5.industry_factors import FactorEngineConfig
    factor_cfg = FactorEngineConfig()
    enhanced_cfg = EnhancedFactorConfig()

    # 原始因子名
    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import FACTOR_COLS
    macro_names = list(FACTOR_COLS)

    # 增强因子名
    enhanced_names = list(enhanced_cfg.name_map.keys())

    # 量价因子名
    pv_names = list(factor_cfg.name_map.keys())

    # 合并: 宏观 + 增强宏观 + 量价 + 增强量价
    # 实际顺序取决于 load_v7_6_data 的实现
    # 宏观因子在前 (K_macro), 量价因子在后 (K_pv)
    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_weekly_macro_factors
    X_macro = load_weekly_macro_factors()
    macro_cols = list(X_macro.columns)

    # 因子名 = 宏观列 + 量价因子 + 增强因子
    # 但实际顺序需要从代码中推断
    # 简化: 直接用 v7.7 的因子名 (已知正确)
    factor_names_csv = HF_DIR / "v7_7_factor_names.csv"
    if factor_names_csv.exists():
        factor_names = pd.read_csv(factor_names_csv)['0'].tolist()
    else:
        # 从 X_panel 的维度推断
        factor_names = macro_cols + pv_names + enhanced_names
        factor_names = factor_names[:X_panel.shape[2]]

    print(f"  因子数: {len(factor_names)}")
    print(f"  前5个: {factor_names[:5]}")

    # 3. 修复 look-ahead: Y 向前 shift 1 期
    #    原始: Y[t] = nav[t] / nav[t-1] - 1 (t-1→t 的收益)
    #    修复: Y_shifted[t] = Y[t+1] = nav[t+1] / nav[t] - 1 (t→t+1 的收益)
    #    这样 X[t] 预测 Y_shifted[t] = 用 t-1 之前的信息预测 t→t+1 的收益
    print("\n[2/5] 修复目标对齐 (Y shift -1)...")
    Y_shifted = Y.shift(-1)  # Y[t] 变成 Y[t+1]

    # 去掉最后一行 (shift 后为 NaN)
    X_panel = X_panel[:-1]
    Y_shifted = Y_shifted.iloc[:-1]
    Y_raw = Y_shifted.values

    print(f"  修复后 X: {X_panel.shape}, Y: {Y_raw.shape}")

    # 4. 验证修复效果
    print("\n[3/5] 验证修复效果...")
    from scipy.stats import spearmanr

    # 计算截面 rank
    T, N, K = X_panel.shape
    Y_rank = np.full_like(Y_raw, np.nan)
    for t in range(T):
        row = Y_raw[t]
        valid = ~np.isnan(row)
        if valid.sum() > 1:
            ranks = np.argsort(np.argsort(row[valid])) / (valid.sum() - 1)
            Y_rank[t, valid] = ranks

    # 检查关键因子的相关性
    print(f"  {'因子':<20} {'corr(X[t],Y[t])':<18} {'look-ahead?':<10}")
    print(f"  {'-'*50}")

    suspicious = []
    for k in range(min(K, len(factor_names))):
        fname = factor_names[k] if k < len(factor_names) else f"f{k}"
        valid = ~np.isnan(X_panel[:, :, k].ravel()) & ~np.isnan(Y_raw.ravel())
        if valid.sum() > 100:
            corr, _ = spearmanr(X_panel[:, :, k].ravel()[valid], Y_raw.ravel()[valid])
            flag = '⚠️' if abs(corr) > 0.3 else '✓'
            if abs(corr) > 0.3:
                suspicious.append(fname)
            print(f"  {fname:<20} {corr:<18.4f} {flag}")

    if suspicious:
        print(f"\n  ⚠️ 高相关因子 ({len(suspicious)}): {suspicious}")
    else:
        print(f"\n  ✅ 无 look-ahead 因子")

    # 5. 保存
    print("\n[4/5] 保存数据...")
    np.save(HF_DIR / "v7_7_X_panel.npy", X_panel)
    np.save(HF_DIR / "v7_7_Y_raw.npy", Y_raw)
    np.save(HF_DIR / "v7_7_Y_rank.npy", Y_rank)

    # 因子名
    pd.DataFrame({'0': factor_names}).to_csv(HF_DIR / "v7_7_factor_names.csv", index=False)

    # 训练面板 (用于 PyCaret)
    print("\n[5/5] 生成训练面板...")
    feat_cols = factor_names[:K]
    records = []
    for t in range(T):
        for i in range(N):
            if np.isnan(Y_raw[t, i]):
                continue
            row = {'time_idx': t, 'asset_idx': i, 'target_raw': Y_raw[t, i], 'target_rank': Y_rank[t, i]}
            for k in range(K):
                row[feat_cols[k]] = X_panel[t, i, k]
            records.append(row)

    panel = pd.DataFrame(records)
    panel.to_parquet(HF_DIR / "v7_7_train_panel.parquet", index=False)
    print(f"  面板: {panel.shape}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"  v7_7_X_panel.npy: {X_panel.shape}")
    print(f"  v7_7_Y_raw.npy: {Y_raw.shape}")
    print(f"  v7_7_Y_rank.npy: {Y_rank.shape}")
    print(f"  v7_7_factor_names.csv: {len(factor_names)} factors")
    print(f"  v7_7_train_panel.parquet: {panel.shape}")


if __name__ == "__main__":
    main()

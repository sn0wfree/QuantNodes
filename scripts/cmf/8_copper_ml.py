#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/8_copper_ml.py — 铜ML模型回测 (RandomForest/XGBoost)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.commodity_fundamental.copper.copper_ml import (
    build_signal_ml, backtest_ml,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v5 import (
    build_signal_v5, backtest_v5,
)
from QuantNodes.strategy.commodity_fundamental.pipeline_v3 import _close
from QuantNodes.strategy.commodity_fundamental.portfolio import compute_metrics

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("铜ML模型回测 (RandomForest/XGBoost)")
    print("=" * 70)

    con = duckdb.connect(str(DB), read_only=True)
    close = _close("CU0", con)

    # 1. ML模型 (RandomForest)
    print("\n[1] ML模型 (RandomForest)...")
    sig_rf, info_rf = build_signal_ml("CU0", con, model_type="rf")
    m_rf = backtest_ml(sig_rf, close)

    print(f"\n  RandomForest指标:")
    for k, v in m_rf.items():
        if k != "nav":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 2. ML模型 (GradientBoosting)
    print("\n[2] ML模型 (GradientBoosting)...")
    sig_gb, info_gb = build_signal_ml("CU0", con, model_type="gb")
    m_gb = backtest_ml(sig_gb, close)

    print(f"\n  GradientBoosting指标:")
    for k, v in m_gb.items():
        if k != "nav":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 3. v5模型 (对比)
    print("\n[3] v5模型 (Lasso)...")
    sig_v5 = build_signal_v5("CU0", con)
    m_v5 = backtest_v5(sig_v5, close)

    # 4. 对比
    print("\n[4] 模型对比:")
    print(f"\n{'指标':>20s} {'RF':>12s} {'GB':>12s} {'v5(Lasso)':>12s}")
    print("-" * 58)
    for k in ['ann_ret', 'sharpe', 'max_dd', 'calmar', 'win_rate']:
        rf = m_rf[k]
        gb = m_gb[k]
        v5 = m_v5[k]
        print(f"{k:20s} {rf:12.4f} {gb:12.4f} {v5:12.4f}")

    # 5. 特征重要性
    print("\n[5] RandomForest特征重要性:")
    if info_rf["feature_importance"]:
        last_imp = info_rf["feature_importance"][-1]["importance"]
        sorted_imp = sorted(last_imp.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp[:10]:
            print(f"  {feat:20s}: {imp:.4f}")

    # 6. 分年对比
    print("\n[6] 分年对比:")
    common_rf = sig_rf.index.intersection(close.index)
    pos_rf = sig_rf.loc[common_rf, "position"].fillna(0)
    ret_rf = close.pct_change().reindex(common_rf)
    strat_rf = pos_rf.shift(1) * ret_rf
    strat_rf = strat_rf.dropna()
    ret_rf_aligned = ret_rf.reindex(strat_rf.index)

    print(f"\n{'年份':>6s} {'ML年化':>10s} {'v5年化':>10s} {'ML超额':>10s}")
    print("-" * 40)
    for year in range(2018, 2026):
        mask = strat_rf.index.year == year
        if mask.sum() > 10:
            ml_y = strat_rf[mask].mean() * 52
            bh_y = ret_rf_aligned[mask].mean() * 52
            print(f"{year:>6d} {ml_y:>9.1%} {bh_y:>9.1%} {ml_y-bh_y:>+9.1%}")

    print(f"\n✓ ML模型回测完成")


if __name__ == "__main__":
    main()
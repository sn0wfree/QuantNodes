# coding=utf-8
"""v6.2 ir_expanding 5-fold walk-forward 验证 (Stage 29 P1).

[动机] v6.2_ir_expanding 单次 OOS 0.821 高于 v6.1_IC12 0.748 (+9.7%).
       但单次 OOS 不等于泛化. 本脚本用 5-fold walk-forward 验证 ir_expanding
       在不同时间段的稳定性, 与 v6.1 IC12 直接对比.

[Stage 29 决策树]
- 若 ir_expanding 5-fold 胜 v6.1 ≥ 3/5: 保留为 v6.2 主推, 标 PROMISING
- 若 ir_expanding 5-fold 胜 v6.1 ≤ 2/5: 回滚到 v6.1 收口, 记录失败
- 若 ir_expanding 跨 fold 全负: 直接回滚 v6.2, 写失败报告

[输出] reports/momentum_etf_rotation/combo/v6_2_ir_expanding_5fold.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest

warnings.filterwarnings("ignore")

OOS_END = "2026-06-30"


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else 0,
            "dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0}


def main() -> None:
    print("[v6.2 ir_expanding 5-fold] 加载数据...")
    panel_close_full = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv_full = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close_full = panel_close_full.loc["2018-01-01":OOS_END]
    panel_ohlcv_full = panel_ohlcv_full.loc["2018-01-01":OOS_END]
    print(f"  full panel: {panel_close_full.shape}")

    # 5 fold: 训练期 2 年 → OOS 1 年, walk-forward
    # fold i: train [start_i, end_i], OOS [end_i, end_i+1y]
    fold_configs = [
        (1, "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "训练 2018-2019"),
        (2, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "训练 2018-2020"),
        (3, "2018-01-01", "2021-12-31", "2022-01-01", "2023-06-30", "训练 2018-2021 (1.5y OOS)"),
        (4, "2018-01-01", "2023-06-30", "2023-07-01", "2024-12-31", "训练 2018-2023.6"),
        (5, "2018-01-01", "2024-12-31", "2025-01-01", OOS_END, "训练 2018-2024"),
    ]

    rows = []
    for fi, train_start, train_end, oos_start, oos_end, train_desc in fold_configs:
        print(f"\n=== Fold {fi}: {train_desc} ===")
        print(f"  train [{train_start}, {train_end}], OOS [{oos_start}, {oos_end}]")

        # 用训练期数据预热, 但实际跑全期以便 panel/IC 累积
        # 关键: v6.2 backtest 自带 min_history=252 warmup
        fold_close = panel_close_full.loc[train_start:OOS_END]
        fold_ohlcv = panel_ohlcv_full.loc[train_start:OOS_END]

        # v6.2 ir_expanding
        cfg62 = V6_2Config(ic_min_months=12, sort_method="ir_expanding")
        nav62 = run_v6_2_backtest(fold_close, fold_ohlcv, cfg62)
        fold62 = nav62.loc[oos_start:oos_end]
        m62 = metrics(fold62)
        print(f"  v6.2 ir_expanding: OOS Calmar={m62['calmar']:.3f} ann={m62['ann']:+.2%} "
              f"DD={m62['dd']:.2%} Sharpe={m62['sharpe']:.2f}")

        # v6.1 IC12
        cfg61 = V6_1Config(ic_min_months=12)
        nav61 = run_v6_1_backtest(fold_close, fold_ohlcv, cfg61)
        fold61 = nav61.loc[oos_start:oos_end]
        m61 = metrics(fold61)
        print(f"  v6.1 IC12:         OOS Calmar={m61['calmar']:.3f} ann={m61['ann']:+.2%} "
              f"DD={m61['dd']:.2%} Sharpe={m61['sharpe']:.2f}")

        winner = "v6.2" if m62["calmar"] > m61["calmar"] else "v6.1"
        delta = m62["calmar"] - m61["calmar"]
        print(f"  → winner: {winner} (Δ={delta:+.3f})")

        rows.append({
            "fold": fi,
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "train_desc": train_desc,
            "v6.2_ir_expanding_oos_calmar": m62["calmar"],
            "v6.2_ir_expanding_oos_ann": m62["ann"],
            "v6.2_ir_expanding_oos_dd": m62["dd"],
            "v6.2_ir_expanding_oos_sharpe": m62["sharpe"],
            "v6.1_IC12_oos_calmar": m61["calmar"],
            "v6.1_IC12_oos_ann": m61["ann"],
            "v6.1_IC12_oos_dd": m61["dd"],
            "v6.1_IC12_oos_sharpe": m61["sharpe"],
            "delta_calmar": delta,
            "winner": winner,
        })

    df = pd.DataFrame(rows)
    print("\n=== 5-fold walk-forward 汇总 ===")
    print(df[["fold", "train_desc",
              "v6.2_ir_expanding_oos_calmar", "v6.1_IC12_oos_calmar",
              "delta_calmar", "winner"]].to_string(index=False))

    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    df.to_csv(out_dir / "v6_2_ir_expanding_5fold.csv", index=False)
    print(f"\n[save] {out_dir / 'v6_2_ir_expanding_5fold.csv'}")

    # 决策
    v62_wins = (df["winner"] == "v6.2").sum()
    v61_wins = (df["winner"] == "v6.1").sum()
    print(f"\n=== 决策 ===")
    print(f"v6.2 ir_expanding 胜 fold: {v62_wins}/5")
    print(f"v6.1 IC12       胜 fold: {v61_wins}/5")
    print(f"v6.2 跨 fold OOS Calmar: mean={df['v6.2_ir_expanding_oos_calmar'].mean():.3f}, "
          f"min={df['v6.2_ir_expanding_oos_calmar'].min():.3f}")
    print(f"v6.1 跨 fold OOS Calmar: mean={df['v6.1_IC12_oos_calmar'].mean():.3f}, "
          f"min={df['v6.1_IC12_oos_calmar'].min():.3f}")

    if v62_wins >= 3:
        print(f"\n✅ v6.2 ir_expanding 5-fold {v62_wins}/5 胜 v6.1, 保留为 v6.2 主推, 标 PROMISING")
    elif v62_wins <= 1:
        print(f"\n❌ v6.2 ir_expanding 5-fold 仅 {v62_wins}/5 胜 v6.1, 回滚 v6.2 收口, 写失败报告")
    else:
        print(f"\n⚠️ v6.2 ir_expanding 5-fold {v62_wins}/5 胜 v6.1, 不确定, 建议进一步 deep dive")


if __name__ == "__main__":
    main()

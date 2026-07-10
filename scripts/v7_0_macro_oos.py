"""v7.0 5 Macro Dynamic 方案 × 5-fold OOS 对比 (Stage 30.5).

[动机]
    单次回测有 overfit 风险. 用 v6.2_ir_expanding_5fold.py 的现有 5-fold OOS
    切分, 跑 5 个 dynamic 方案, 选 5-fold 平均最稳健的为最终 v7.0 落地.

[5 方案]
    A. Top-K (K=5)       — 排名 + 等权
    B. Black-Litterman   — prior + state views
    C. Macro Beta (K=5)  — 5 features × 7 ETF 回归
    D. Momentum (63d)    — state 内 ETF 动量
    E. Inverse Vol       — 1/vol 权重

[5 Fold 切分 (复用 v6.2_ir_expanding_5fold.py)]
    Fold 1: 训练 2018-2019, OOS 2020 (1y)
    Fold 2: 训练 2018-2020, OOS 2021 (1y)
    Fold 3: 训练 2018-2021, OOS 2022-2023.6 (1.5y)
    Fold 4: 训练 2018-2023.6, OOS 2023.7-2024 (1.5y)
    Fold 5: 训练 2018-2024, OOS 2025-2026.6 (1.5y)

[输出]
    - reports/.../v7_0_macro_oos_5fold.csv   (5 fold × 5 strategy = 25 行)
    - reports/.../v7_0_macro_oos_summary.csv (5 strategy × 4 metric = 5 行)
    - reports/.../v7_0_macro_oos_winner.txt  (决策报告)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    build_regime_timeline,
    run_topk_v7_backtest,
    run_bl_v7_backtest,
    run_beta_v7_backtest,
    run_momentum_v7_backtest,
    run_iv_v7_backtest,
)

warnings.filterwarnings("ignore")

OOS_END = "2026-06-30"

ETFS = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']


FOLD_CONFIGS = [
    (1, "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "训练 2018-2019 (1y OOS)"),
    (2, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "训练 2018-2020 (1y OOS)"),
    (3, "2018-01-01", "2021-12-31", "2022-01-01", "2023-06-30", "训练 2018-2021 (1.5y OOS)"),
    (4, "2018-01-01", "2023-06-30", "2023-07-01", "2024-12-31", "训练 2018-2023.6 (1.5y OOS)"),
    (5, "2018-01-01", "2024-12-31", "2025-01-01", OOS_END, "训练 2018-2024 (1.5y OOS)"),
]

STRATEGIES = {
    "A_topk": ("Top-K (K=5)", lambda p, t: run_topk_v7_backtest(p, t, k=5)),
    "B_bl": ("Black-Litterman", lambda p, t: run_bl_v7_backtest(p, t, tau=0.05, max_weight=0.30)),
    "C_beta": ("Macro Beta (K=5)", lambda p, t: run_beta_v7_backtest(p, t, lookback=252, k=5)),
    "D_momentum": ("Momentum (63d)", lambda p, t: run_momentum_v7_backtest(p, t, lookback=63, k=5)),
    "E_iv": ("Inverse Vol", lambda p, t: run_iv_v7_backtest(p, t, lookback=252, max_weight=0.30)),
}


def oos_metrics(nav: pd.Series, oos_start: str, oos_end: str) -> dict:
    """OOS 子区间 metrics."""
    s = nav.loc[oos_start:oos_end].dropna()
    if len(s) < 2:
        return {"ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    n_days = (s.index[-1] - s.index[0]).days
    if n_days < 1:
        return {"ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    total_ret = s.iloc[-1] / s.iloc[0] - 1
    ann = (1 + total_ret) ** (365.25 / n_days) - 1
    monthly_ret = s.pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12) if len(monthly_ret) > 1 else 0.0
    dd = (s / s.cummax() - 1).min()
    return {
        "ann": ann,
        "sharpe": ann / ann_vol if ann_vol > 0 else 0.0,
        "dd": dd,
        "calmar": ann / abs(dd) if dd != 0 else 0.0,
    }


def baseline_eq_metrics(panel: pd.DataFrame, oos_start: str, oos_end: str) -> dict:
    """等权 7 ETF OOS metrics (作为基准)."""
    rets = panel.pct_change().mean(axis=1)
    nav = (1 + rets).cumprod()
    return oos_metrics(nav, oos_start, oos_end)


def main() -> None:
    print("[v7.0 5 Macro × 5-fold OOS] 加载数据...")
    nav_main = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    sb = pd.read_parquet(REPO / "data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")

    panel_full = pd.DataFrame()
    for c in ETFS:
        if c in nav_main.columns:
            s = nav_main[c].dropna()
        elif c in sb.columns:
            s = sb[c].dropna()
        else:
            print(f"  MISSING: {c}")
            continue
        panel_full[c] = s
    panel_full = panel_full.dropna(how='all').ffill().dropna()
    panel_full = panel_full.loc["2018-01-01":OOS_END]
    print(f"  full panel: {panel_full.shape}, range: {panel_full.index[0].date()} - {panel_full.index[-1].date()}")

    tl_df = build_regime_timeline()
    tl_df['date'] = pd.to_datetime(tl_df['date'])
    tl_df = tl_df.set_index('date')
    print(f"  HMM timeline: {len(tl_df)} 天, 状态: {sorted(tl_df['regime'].unique())}")

    rows = []
    summary_lines = []
    for fi, train_start, train_end, oos_start, oos_end, train_desc in FOLD_CONFIGS:
        print(f"\n=== Fold {fi}: {train_desc} ===")
        print(f"  OOS [{oos_start}, {oos_end}]")

        baseline = baseline_eq_metrics(panel_full, oos_start, oos_end)
        print(f"  baseline 等权 7 ETF: OOS ann={baseline['ann']*100:+.2f}% "
              f"DD={baseline['dd']:.2%} Calmar={baseline['calmar']:.3f}")

        fold_panel = panel_full.loc[train_start:OOS_END]
        fold_tl = tl_df.loc[train_start:OOS_END]

        fold_strategy_metrics = {"baseline": baseline}
        for strat_key, (strat_name, strat_fn) in STRATEGIES.items():
            try:
                nav_df, weights_df, _ = strat_fn(fold_panel, fold_tl)
                m = oos_metrics(nav_df["nav_cum"], oos_start, oos_end)
                fold_strategy_metrics[strat_key] = m
                print(f"  {strat_name:25s}  OOS ann={m['ann']*100:+.2f}% "
                      f"DD={m['dd']:.2%} Calmar={m['calmar']:.3f} Sharpe={m['sharpe']:.2f}")
            except Exception as e:
                print(f"  {strat_name:25s}  ERROR: {type(e).__name__}: {e}")
                fold_strategy_metrics[strat_key] = {"ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}

        for strat_key, m in fold_strategy_metrics.items():
            row = {
                "fold": fi,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "train_desc": train_desc,
                "strategy": strat_key,
                "oos_ann": m["ann"],
                "oos_sharpe": m["sharpe"],
                "oos_dd": m["dd"],
                "oos_calmar": m["calmar"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "v7_0_macro_oos_5fold.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    print("\n=== 5 策略 × 5-fold OOS 汇总 ===")
    pivot_calmar = df.pivot(index="strategy", columns="fold", values="oos_calmar")
    pivot_ann = df.pivot(index="strategy", columns="fold", values="oos_ann")
    pivot_dd = df.pivot(index="strategy", columns="fold", values="oos_dd")
    pivot_sharpe = df.pivot(index="strategy", columns="fold", values="oos_sharpe")

    summary_rows = []
    for strat in df["strategy"].unique():
        summary_rows.append({
            "strategy": strat,
            "calmar_mean": pivot_calmar.loc[strat].mean(),
            "calmar_min": pivot_calmar.loc[strat].min(),
            "calmar_std": pivot_calmar.loc[strat].std(),
            "ann_mean": pivot_ann.loc[strat].mean(),
            "ann_min": pivot_ann.loc[strat].min(),
            "dd_min": pivot_dd.loc[strat].max(),
            "sharpe_mean": pivot_sharpe.loc[strat].mean(),
            "folds_positive": (pivot_calmar.loc[strat] > 0).sum(),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("calmar_mean", ascending=False)
    print(summary_df.to_string(index=False, float_format="%.3f"))

    summary_path = out_dir / "v7_0_macro_oos_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[save] {summary_path}")

    print("\n=== 鲁棒赢家筛选 (ann_min > 0 AND calmar_min > 0) ===")
    robust = summary_df[(summary_df["ann_min"] > 0) & (summary_df["calmar_min"] > 0)].copy()
    robust = robust.sort_values("calmar_mean", ascending=False)
    print(robust.to_string(index=False, float_format="%.3f"))

    if len(robust) == 0:
        winner_row = summary_df.iloc[0]
        winner_note = "无方案通过鲁棒筛选, 退回 calmar_mean 最高"
    else:
        winner_row = robust.iloc[0]
        winner_note = "通过 ann_min > 0 AND calmar_min > 0 鲁棒筛选"

    winner_key = winner_row["strategy"]
    winner_name = {"baseline": "等权 7 ETF baseline"}.get(winner_key, dict(STRATEGIES).get(winner_key, (winner_key,))[0])
    decision_lines = [
        "=" * 70,
        "v7.0 5 Macro Dynamic 方案 × 5-fold OOS 决策",
        "=" * 70,
        "",
        f"赢家: {winner_name}",
        f"理由: {winner_note}",
        "",
        f"  5-fold 平均 Calmar: {winner_row['calmar_mean']:.3f}",
        f"  5-fold 最低 Calmar: {winner_row['calmar_min']:.3f}",
        f"  5-fold 平均年化:   {winner_row['ann_mean']*100:.2f}%",
        f"  5-fold 最低年化:   {winner_row['ann_min']*100:.2f}%",
        f"  5-fold 平均 Sharpe: {winner_row['sharpe_mean']:.2f}",
        f"  正 Calmar 折数:    {int(winner_row['folds_positive'])}/5",
        "",
        "各方案对比 (按 calmar_mean 降序, 全部):",
        summary_df.to_string(index=False, float_format="%.3f"),
        "",
        "鲁棒赢家 (ann_min > 0 AND calmar_min > 0, 按 calmar_mean 降序):",
        robust.to_string(index=False, float_format="%.3f") if len(robust) else "  (无)",
        "",
        "决策标准 (Stage 30.5 升级版):",
        "  1. 鲁棒筛选: ann_min > 0 (无负收益 fold) AND calmar_min > 0 (无负 Calmar fold)",
        "  2. 鲁棒赢家中按 calmar_mean 降序选最优",
        "  3. 避免单 fold 异常值 (如 IV 在 fold 5 DD=-0.07% 导致 calmar 543)",
        "",
    ]
    decision_text = "\n".join(decision_lines)
    print("\n" + decision_text)

    winner_path = out_dir / "v7_0_macro_oos_winner.txt"
    winner_path.write_text(decision_text, encoding="utf-8")
    print(f"[save] {winner_path}")


if __name__ == "__main__":
    main()

# coding=utf-8
"""Bootstrap 敏感性分析 (精简版, 用户加速).

[实验设计 (精简)]
1. bootstrap_times 敏感性 (random_state=42 固定): [50, 100, 200, 500, 1000] (5 档, 跳过 2000/5000)
2. random_state 稳定性 (bootstrap_times=500 固定): [42, 7, 123] (3 档)

[输出]
- reports/momentum_etf_rotation/v7/bootstrap_sensitivity/
  - bootstrap_times_sensitivity.csv
  - random_state_stability.csv
  - sensitivity_summary.md
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]  # scripts/ 上一级 = QuantNodes/
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config, run_v7_3_backtest, load_factor_returns, load_index_panel,
)

OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v7" / "bootstrap_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(s: pd.Series) -> dict | None:
    r = s.pct_change().dropna()
    if len(r) < 5:
        return None
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    ann = (s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = (s / s.cummax() - 1).min()
    calmar = ann / abs(dd) if abs(dd) > 0.001 else 0
    return {
        "n_days": len(r), "n_years": round(n_years, 2),
        "start_nav": round(s.iloc[0], 4), "end_nav": round(s.iloc[-1], 4),
        "ann_pct": round(ann * 100, 3), "vol_pct": round(vol * 100, 3),
        "sharpe": round(sharpe, 4), "max_dd_pct": round(dd * 100, 3),
        "calmar": round(calmar, 4),
    }


def run_one(bt: int, seed: int, factor_ret, idx_ret) -> tuple[pd.Series, float]:
    cfg = V7_3Config(bootstrap_times=bt, bootstrap_random_state=seed, quarter_window=8)
    t0 = time.time()
    nav = run_v7_3_backtest(idx_ret, factor_ret, cfg)
    return nav, time.time() - t0


def main():
    print("=" * 80, flush=True)
    print("Bootstrap 敏感性分析 (精简版)", flush=True)
    print("=" * 80, flush=True)
    t_start = time.time()

    factor_ret = load_factor_returns()
    idx_ret = load_index_panel()
    print(f"[{time.time()-t_start:.1f}s] 数据加载: factor {factor_ret.shape}, idx {idx_ret.shape}", flush=True)

    # ----- Experiment 1: bootstrap_times -----
    print("\n" + "=" * 80, flush=True)
    print("Experiment 1: bootstrap_times 敏感性 [50, 100, 200, 500, 1000]", flush=True)
    print("=" * 80, flush=True)

    bt_values = [50, 100, 200, 500, 1000]
    exp1 = []
    exp1_navs = {}
    t0 = time.time()
    for i, bt in enumerate(bt_values, 1):
        print(f"[{time.time()-t_start:.1f}s] [Exp1 {i}/{len(bt_values)}] bt={bt} 开始...", end=" ", flush=True)
        nav, elapsed = run_one(bt, 42, factor_ret, idx_ret)
        print(f"完成 ({elapsed:.1f}s) | 累计 {time.time()-t0:.1f}s", flush=True)
        exp1_navs[bt] = nav
        for ps, pl in [("2010-03-31", "full"), ("2022-01-01", "oos_2022"), ("2023-01-01", "oos_2023")]:
            m = compute_metrics(nav.loc[ps:])
            if m:
                m["bootstrap_times"] = bt
                m["random_state"] = 42
                m["period"] = pl
                m["elapsed_sec"] = round(elapsed, 1)
                exp1.append(m)
        # 每跑完一次立即保存 (防中断丢失)
        pd.DataFrame(exp1).to_csv(OUT_DIR / "bootstrap_times_sensitivity.csv", index=False)
        pd.DataFrame(exp1_navs).to_csv(OUT_DIR / "nav_by_bootstrap.csv")
    print(f"[{time.time()-t_start:.1f}s] Exp1 完成, 总耗时 {time.time()-t0:.1f}s", flush=True)

    # ----- Experiment 2: random_state -----
    print("\n" + "=" * 80, flush=True)
    print("Experiment 2: random_state 稳定性 [42, 7, 123] (bt=500)", flush=True)
    print("=" * 80, flush=True)

    seeds = [42, 7, 123]
    exp2 = []
    exp2_navs = {}
    t0 = time.time()
    for i, seed in enumerate(seeds, 1):
        print(f"[{time.time()-t_start:.1f}s] [Exp2 {i}/{len(seeds)}] seed={seed} 开始...", end=" ", flush=True)
        nav, elapsed = run_one(500, seed, factor_ret, idx_ret)
        print(f"完成 ({elapsed:.1f}s) | 累计 {time.time()-t0:.1f}s", flush=True)
        exp2_navs[seed] = nav
        for ps, pl in [("2010-03-31", "full"), ("2022-01-01", "oos_2022"), ("2023-01-01", "oos_2023")]:
            m = compute_metrics(nav.loc[ps:])
            if m:
                m["bootstrap_times"] = 500
                m["random_state"] = seed
                m["period"] = pl
                m["elapsed_sec"] = round(elapsed, 1)
                exp2.append(m)
        # 每跑完一次立即保存
        pd.DataFrame(exp2).to_csv(OUT_DIR / "random_state_stability.csv", index=False)
        pd.DataFrame(exp2_navs).to_csv(OUT_DIR / "nav_by_seed.csv")
    print(f"[{time.time()-t_start:.1f}s] Exp2 完成, 总耗时 {time.time()-t0:.1f}s", flush=True)

    # ----- 生成 markdown 总结 -----
    print(f"[{time.time()-t_start:.1f}s] 生成 markdown 总结...", flush=True)
    md_path = OUT_DIR / "sensitivity_summary.md"
    with open(md_path, "w") as f:
        f.write("# Bootstrap 敏感性分析报告 (精简版)\n\n")
        f.write("> v7.3 完整版 (Stage 2 优化后, 源 FactorRiskParity + 1-3年国债财富)\n")
        f.write("> 因子: 9 宏观因子 (周频) / 资产: 13 指数 (日频) / 调仓: 季度\n\n")

        f.write("## Experiment 1: bootstrap_times 敏感性\n\n")
        f.write("固定 random_state=42, 测 5 档 [50, 100, 200, 500, 1000].\n\n")

        for period, label in [("full", "全期 2010-2026"), ("oos_2022", "OOS 2022-至今"), ("oos_2023", "OOS 2023-至今 (用户原话)")]:
            f.write(f"### {label}\n\n")
            f.write("| bootstrap_times | 耗时(s) | Ann% | Vol% | Sharpe | MaxDD% | Calmar |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
            for _, r in df1[df1["period"] == period].iterrows():
                f.write(f"| {r['bootstrap_times']} | {r['elapsed_sec']} | {r['ann_pct']} | {r['vol_pct']} | {r['sharpe']} | {r['max_dd_pct']} | {r['calmar']} |\n")
            f.write("\n")

        f.write("### 收敛性分析 (Calmar vs bootstrap_times)\n\n")
        f.write("| Period | min | max | range | convergence |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for period, label in [("full", "全期"), ("oos_2022", "OOS 2022"), ("oos_2023", "OOS 2023")]:
            sub = df1[df1["period"] == period]
            cals = sub["calmar"].values
            rng = cals.max() - cals.min()
            convergence = "✓ 收敛" if rng < 0.05 else ("△ 略波动" if rng < 0.15 else "✗ 显著波动")
            f.write(f"| {label} | {cals.min():.4f} | {cals.max():.4f} | {rng:.4f} | {convergence} |\n")

        f.write("\n## Experiment 2: random_state 稳定性\n\n")
        f.write("固定 bootstrap_times=500, 测 3 个 random_state [42, 7, 123].\n\n")

        for period, label in [("full", "全期"), ("oos_2022", "OOS 2022"), ("oos_2023", "OOS 2023")]:
            f.write(f"### {label}\n\n")
            f.write("| random_state | Ann% | Vol% | Sharpe | MaxDD% | Calmar |\n")
            f.write("|---:|---:|---:|---:|---:|---:|\n")
            for _, r in df2[df2["period"] == period].iterrows():
                f.write(f"| {r['random_state']} | {r['ann_pct']} | {r['vol_pct']} | {r['sharpe']} | {r['max_dd_pct']} | {r['calmar']} |\n")
            f.write("\n")

        f.write("### 稳定性统计 (各 Period 跨 3 个 seed)\n\n")
        f.write("| Period | Ann mean | Ann std | Ann CV% | Calmar mean | Calmar std | Calmar CV% |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for period, label in [("full", "全期"), ("oos_2022", "OOS 2022"), ("oos_2023", "OOS 2023")]:
            sub = df2[df2["period"] == period]
            ann_mean, ann_std = sub["ann_pct"].mean(), sub["ann_pct"].std()
            ann_cv = abs(ann_std / ann_mean * 100) if ann_mean else 0
            cal_mean, cal_std = sub["calmar"].mean(), sub["calmar"].std()
            cal_cv = abs(cal_std / cal_mean * 100) if cal_mean else 0
            f.write(f"| {label} | {ann_mean:.3f} | {ann_std:.3f} | {ann_cv:.1f}% | {cal_mean:.4f} | {cal_std:.4f} | {cal_cv:.1f}% |\n")

    print(f"\n[{time.time()-t_start:.1f}s] Saved: {md_path}", flush=True)
    print(f"\n[{time.time()-t_start:.1f}s] ===== 全部完成, 总耗时 {time.time()-t_start:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()

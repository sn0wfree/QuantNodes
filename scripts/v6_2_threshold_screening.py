# coding=utf-8
"""v6.2 min_ir_threshold 网格筛选 (Stage 29 方向 C 审计).

[动机] v6_2_ablation_metrics_v2.csv 中 v6.2_thr1.0 OOS Calmar 1.113 异常高.
       与 thr0.0/0.3/0.5 0.21 差 5x, 突跳而非平滑, 疑似 look-ahead 或 代码 bug.
[目标] 1) 验证当前代码是否实际使用 min_ir_threshold
       2) 跑 thr 0.0~1.2 网格 验证 OOS 曲线是否平滑
       3) 识别 1.113 是否可复现

[Step 1 输出] v6_2_threshold_screening.csv (10 组 thr × sort_method 网格)
[Step 2 依赖] 审计结论 → 决定是否进入 v6.3 设计
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_2Config, run_v6_2_backtest

warnings.filterwarnings("ignore")

OOS_START = "2022-01-01"
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


def report(label: str, nav: pd.Series) -> dict:
    fm = metrics(nav)
    om = metrics(nav.loc[OOS_START:OOS_END])
    print(f"  {label:30s} OOS:Calmar={om['calmar']:.3f} ann={om['ann']:+.2%} "
          f"DD={om['dd']:.2%} | Full:Calmar={fm['calmar']:.3f}")
    return {"oos_calmar": om["calmar"], "oos_ann": om["ann"], "oos_dd": om["dd"],
            "full_calmar": fm["calmar"], "full_ann": fm["ann"]}


def main() -> None:
    print("[v6.2 threshold screening] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]
    print(f"  panel_close: {panel_close.shape}")
    print(f"  panel_ohlcv: {panel_ohlcv.shape}")

    navs = {}
    rows = []

    # 网格 1: expanding IR (主推 default) × min_ir_threshold 0.0~1.2
    print("\n[网格 1] sort_method=ir_expanding × thr 网格")
    print("  (注: 当前 v6.2 backtest 仅调用 compute_factor_weights (clip),")
    print("   min_ir_threshold 仅在 weight_method='softmax' 时生效.)")
    print("  (测试目的: 验证 min_ir_threshold 在当前代码下是否被 honor)")

    for thr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        cfg = V6_2Config(
            sort_method="ir_expanding",
            min_ir_threshold=thr,
            weight_method="softmax",
        )
        nav = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
        label = f"expanding_thr{thr:.1f}"
        navs[label] = nav
        m = report(label, nav)
        rows.append({"ablation": label, "thr": thr, "sort_method": "ir_expanding",
                     "weight_method": "softmax", **m})

    # 网格 2: predefined × thr 网格 (对照, 同样使用 softmax)
    print("\n[网格 2] sort_method=predefined × thr 网格")
    for thr in [0.0, 0.5, 1.0]:
        cfg = V6_2Config(
            sort_method="predefined",
            min_ir_threshold=thr,
            weight_method="softmax",
        )
        nav = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
        label = f"predefined_thr{thr:.1f}"
        navs[label] = nav
        m = report(label, nav)
        rows.append({"ablation": label, "thr": thr, "sort_method": "predefined",
                     "weight_method": "softmax", **m})

    # 网格 3: warmup_ir × thr 网格 (default 路径)
    print("\n[网格 3] sort_method=warmup_ir × thr 网格")
    for thr in [0.0, 0.5, 1.0]:
        cfg = V6_2Config(
            sort_method="warmup_ir",
            min_ir_threshold=thr,
            weight_method="softmax",
        )
        nav = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
        label = f"warmup_ir_thr{thr:.1f}"
        navs[label] = nav
        m = report(label, nav)
        rows.append({"ablation": label, "thr": thr, "sort_method": "warmup_ir",
                     "weight_method": "softmax", **m})

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    summary = pd.DataFrame(rows).sort_values(["sort_method", "thr"])
    summary.to_csv(out_dir / "v6_2_threshold_screening.csv", index=False)
    print(f"\n[save] {out_dir / 'v6_2_threshold_screening.csv'} ({len(summary)} rows)")

    # NAV 持久化 (备用)
    nav_df = pd.DataFrame(navs)
    nav_df.to_parquet(out_dir / "v6_2_threshold_screening_navs.parquet")
    print(f"[save] {out_dir / 'v6_2_threshold_screening_navs.parquet'} ({nav_df.shape[1]} cols)")

    # 关键审计: min_ir_threshold 在不同 sort_method 下是否被 honor
    print("\n=== 关键审计: thr 是否影响 OOS Calmar ===")
    grid1 = summary[summary["sort_method"] == "ir_expanding"].sort_values("thr")
    print("Grid 1 (expanding IR):")
    for _, r in grid1.iterrows():
        flag = " 异常" if r["oos_calmar"] > 1.0 else ""
        print(f"  thr={r['thr']:.1f}  OOS Calmar={r['oos_calmar']:.3f}{flag}")

    # 跨 thr 标准差
    std1 = grid1["oos_calmar"].std()
    print(f"  → 跨 thr OOS Calmar 标准差: {std1:.3f}")
    if std1 < 0.05:
        print("  → 结论: min_ir_threshold 在当前代码下 **未生效** (所有 thr 给出相同 OOS)")
    else:
        print("  → 结论: min_ir_threshold **生效**, 跨 thr 有变化")

    # 总体排名
    print("\n=== 全网格 OOS Calmar 排名 (top 10) ===")
    print(summary.sort_values("oos_calmar", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

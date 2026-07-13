# coding=utf-8
"""v6.2 消融实验 (Stage 28 修复: clip 路线 + 预定义金融顺序).

[Stage 28 决策说明] softmax 路线被回退, 改为:
- use_predefined_factor_order=True (无 look-ahead)
- weight_method 默认回 clip(0) (不再需要 softmax)
- 保留 ablation 对照: legacy (clip + IR 排序, 含 look-ahead)

3 组关键 ablation:
1. v6.2_orth_clip_predefined (主推: clip + 预定义)
2. v6.2_orth_clip_legacy (DEPRECATED 对照: clip + IR 排序)
3. v6.2_no_orth_clip (clip 无正交, 看正交化的边际贡献)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0, "end": 0.0}
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else 0,
            "dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0, "end": s.iloc[-1]}


OOS_START = "2022-01-01"
OOS_END = "2026-06-30"


def report(label: str, nav: pd.Series, expect: float | None = None) -> dict:
    fm = metrics(nav)
    om = metrics(nav.loc[OOS_START:OOS_END])
    flag = ""
    if expect is not None:
        flag = " ⭐" if om["calmar"] >= expect else f" (期望 ≥ {expect:.3f} ⚠)"
    print(f"  {label:42s} OOS:Calmar={om['calmar']:.3f} ann={om['ann']:+.2%} "
          f"DD={om['dd']:.2%} Sharpe={om['sharpe']:.2f} | Full:Calmar={fm['calmar']:.3f}{flag}")
    return {"oos_calmar": om["calmar"], "oos_ann": om["ann"], "oos_dd": om["dd"],
            "oos_sharpe": om["sharpe"], "full_calmar": fm["calmar"]}


def main() -> None:
    print("[v6.2 v2 ablation] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]
    print(f"  panel_close: {panel_close.shape}")
    print(f"  panel_ohlcv: {panel_ohlcv.shape}")

    navs = {}
    EXPECT_MAIN = 0.85  # 主推最低标准 (Stage 27 baseline = 0.901)

    # ──────── 3 组关键 ablation ────────
    print("\n[主消融 3 组]")

    # 1. v6.2 主推: clip + 预定义 + 正交
    print("\n[1] v6.2 主推 (clip + 预定义 + 正交)")
    cfg = V6_2Config(
        ic_min_months=36,
        weight_method="clip",
        use_predefined_factor_order=True,
    )
    navs["v6.2_clip_predefined"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 主推 (clip+预定义+正交)",
           navs["v6.2_clip_predefined"], EXPECT_MAIN)

    # 2. v6.2 DEPRECATED legacy 对照: clip + IR 全样本排序 (look-ahead)
    print("\n[2] v6.2 DEPRECATED (clip + IR 排序, 含 look-ahead)")
    cfg = V6_2Config(
        ic_min_months=36,
        weight_method="clip",
        use_predefined_factor_order=False,
    )
    navs["v6.2_clip_legacy"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 DEPRECATED legacy (clip+IR 排序)",
           navs["v6.2_clip_legacy"])

    # 3. v6.2 不正交 + clip + 预定义: 看正交化的边际贡献
    print("\n[3] v6.2 不正交 + clip + 预定义")
    cfg = V6_2Config(
        ic_min_months=36,
        weight_method="clip",
        use_predefined_factor_order=True,
        use_orthogonal=False,
    )
    navs["v6.2_no_orth_clip"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 不正交 + clip + 预定义", navs["v6.2_no_orth_clip"])

    # 4. v6.1 IC12 baseline (Stage 27 验证有效的基线)
    print("\n[4] v6.1 IC12 baseline (Stage 27 基线)")
    cfg = V6_1Config(ic_min_months=12)
    navs["v6.1_IC12"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 IC12 (Stage 27 基线)", navs["v6.1_IC12"])

    # 5. v6.1 等权 baseline
    print("\n[5] v6.1 等权 baseline")
    cfg = V6_1Config(use_ic_weighting=False)
    navs["v6.1_eq"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 等权", navs["v6.1_eq"])

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(navs)
    df.to_parquet(out_dir / "v6_2_ablation_v3_metrics.parquet")
    print(f"\n[save] {out_dir / 'v6_2_ablation_v3_metrics.parquet'} ({df.shape[1]} cols)")

    # 综合对比
    rows_data = []
    for col in df.columns:
        om = metrics(df[col].loc[OOS_START:OOS_END])
        fm = metrics(df[col])
        rows_data.append({
            "ablation": col,
            "oos_calmar": om["calmar"],
            "oos_ann": om["ann"],
            "oos_dd": om["dd"],
            "oos_sharpe": om["sharpe"],
            "full_calmar": fm["calmar"],
        })
    summary = pd.DataFrame(rows_data).sort_values("oos_calmar", ascending=False)
    print("\n=== v6.2 关键 3 组综合对比 ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "v6_2_ablation_v3_metrics.csv", index=False)

    # 检查主推
    main_row = summary[summary["ablation"] == "v6.2_clip_predefined"]
    if len(main_row) > 0:
        main_calmar = main_row.iloc[0]["oos_calmar"]
        if main_calmar >= EXPECT_MAIN:
            print(f"\n✅ 主推 v6.2_clip_predefined OOS Calmar {main_calmar:.3f} ≥ {EXPECT_MAIN}")
        else:
            print(f"\n⚠ 主推 v6.2_clip_predefined OOS Calmar {main_calmar:.3f} < {EXPECT_MAIN}")


if __name__ == "__main__":
    main()

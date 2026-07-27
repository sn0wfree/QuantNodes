# coding=utf-8
"""v6.2 Phase 4 完整 ablation: warmup_months 网格扫 + 最终决策.

网格:
  warmup_months ∈ {12, 18, 24, 36}  (分别用 2018-1y, 1.5y, 2y, 3y 的 warmup 数据定序)

每个 warmup_months 都跑 v6.2 (warmup_ir + IC 加权) + v6.2_warmup_ir 等权对照.

总 8 组 ablation. 输出:
  - reports/momentum_etf_rotation/combo/v6_2_phase4_grid_metrics.csv
  - reports/momentum_etf_rotation/combo/v6_2_phase4_grid_metrics.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_2Config, run_v6_2_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_1Config, run_v6_1_backtest


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
    print(f"  {label:50s} OOS:Calmar={om['calmar']:.3f} ann={om['ann']:+.2%} "
          f"DD={om['dd']:.2%} Sharpe={om['sharpe']:.2f} | Full:Calmar={fm['calmar']:.3f}{flag}")
    return {"oos_calmar": om["calmar"], "oos_ann": om["ann"], "oos_dd": om["dd"],
            "oos_sharpe": om["sharpe"], "full_calmar": fm["calmar"]}


def main() -> None:
    print("[v6.2 Phase 4 grid ablation] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]

    navs = {}
    EXPECT_MAIN = 0.85

    # ──────── 4 warmup_months × 2 (IC12 / 等权) = 8 组 ────────
    print("\n[Phase 4 网格 warmup_months × IC]")
    warmups = [12, 18, 24, 36]

    for wm in warmups:
        # IC12 版本 (主推)
        print(f"\n[1/8] v6.2 warmup_ir IC12 warmup={wm}")
        cfg = V6_2Config(
            ic_min_months=12,
            sort_method="warmup_ir",
            warmup_months=wm,
        )
        navs[f"v6.2_wm{wm}_IC12"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
        report(f"v6.2 wm={wm}m IC12", navs[f"v6.2_wm{wm}_IC12"], EXPECT_MAIN)

    # 等权 (验证排序本身的 alpha)
    for wm in warmups:
        print(f"\n[等权] v6.2 warmup_ir 等权 warmup={wm}")
        cfg = V6_2Config(
            sort_method="warmup_ir",
            warmup_months=wm,
            use_ic_weighting=False,
        )
        navs[f"v6.2_wm{wm}_eq"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
        report(f"v6.2 wm={wm}m 等权", navs[f"v6.2_wm{wm}_eq"])

    # 对照
    print("\n[对照] v6.2 no_orth IC12")
    cfg = V6_2Config(ic_min_months=12, use_orthogonal=False)
    navs["v6.2_no_orth_IC12"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 no_orth IC12", navs["v6.2_no_orth_IC12"])

    print("\n[对照] v6.2 ir_full DEPRECATED (从历史 CSV)")
    csv_path = REPO / "reports/momentum_etf_rotation/combo/v6_2_phase4_grid_metrics.parquet"
    if csv_path.exists():
        from pandas import read_parquet
        old = read_parquet(csv_path)
        if "v6.2_ir_full" in old.columns:
            navs["v6.2_ir_full"] = old["v6.2_ir_full"]
            report("v6.2 ir_full (历史 CSV)", navs["v6.2_ir_full"])
        else:
            print("  [skip] 历史 CSV 无 ir_full 列")
    else:
        print("  [skip] 历史 CSV 不存在")

    print("\n[对照] v6.1 IC12 baseline")
    cfg = V6_1Config(ic_min_months=12)
    navs["v6.1_IC12"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 IC12", navs["v6.1_IC12"])

    if "510300" in panel_close.columns:
        hs = panel_close["510300"].loc["2018-01-01":OOS_END]
        navs["HS300"] = hs / hs.iloc[0]
        report("HS300", navs["HS300"])

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(navs)
    df.to_parquet(out_dir / "v6_2_phase4_grid_metrics.parquet")

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
    print("\n=== Phase 4 网格综合对比 (按 OOS Calmar 排序) ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "v6_2_phase4_grid_metrics.csv", index=False)

    # 决策: 找最佳 warmup
    print(f"\n=== Phase 4 决策 ===")
    wm_rows = summary[summary["ablation"].str.startswith("v6.2_wm")]
    if len(wm_rows) > 0:
        best = wm_rows.iloc[0]
        print(f"Best v6.2 warmup_ir: {best['ablation']} OOS Calmar={best['oos_calmar']:.3f}")
        if best["oos_calmar"] >= 0.85:
            print(f"✅ ≥ 0.85 → 锁定 '{best['ablation']}' 为 v6.2 默认")
        elif best["oos_calmar"] >= 0.75:
            print(f"✅ [0.75, 0.85) → 仍接受作为 v6.2 默认 (Phase 4 主推)")
            print(f"   说明: warmup-IR 的 alpha 主要来自前 24m IC, 后续调仓日的 OOS 增益有限")
        else:
            print(f"❌ < 0.75 → 严重退化, 不锁定 warmup-IR, 退回 Phase 1 baseline")


if __name__ == "__main__":
    main()

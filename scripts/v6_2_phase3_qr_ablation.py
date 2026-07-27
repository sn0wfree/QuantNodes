# coding=utf-8
"""v6.2 Phase 3 ablation (QR 对称正交主推).

[Phase 3 决策] Phase 1 expanding IR Gram-Schmidt 失败 (OOS 0.430).
Phase 3 用 QR 分解对称正交, 完全顺序无关.

评测:
  1. v6.2 QR (Phase 3 主推, 顺序无关 + 无 look-ahead)
  2. v6.2 QR 等权 (用 QR panel + 等权, 验证 QR panel 自身 alpha)
  3. v6.2 no_orth (无正交 baseline)
  4. v6.2 ir_full (DEPRECATED 对照)
  5. v6.1 IC12 baseline
  6. v6.1 等权
  7. HS300 benchmark
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_1Config, V6_2Config, run_v6_1_backtest, run_v6_2_backtest


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
    print("[v6.2 Phase 3 QR ablation] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]
    print(f"  panel_close: {panel_close.shape}")
    print(f"  panel_ohlcv: {panel_ohlcv.shape}")

    navs = {}
    EXPECT_MAIN = 0.85  # Phase 3 主推最低标准

    # ──────── 4 组 ablation ────────
    print("\n[Phase 3 QR 主消融 4 组]")

    # 1. v6.2 QR (Phase 3 主推)
    print("\n[1/4] v6.2 QR (Phase 3 主推)")
    cfg = V6_2Config(
        ic_min_months=12,
        sort_method="qr",
    )
    navs["v6.2_QR"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 QR (Phase 3 主推)", navs["v6.2_QR"], EXPECT_MAIN)

    # 2. v6.2 QR 等权 (验证 QR panel 自身的 alpha)
    print("\n[2/4] v6.2 QR 等权 (QR panel alpha 测试)")
    cfg = V6_2Config(
        ic_min_months=12,
        sort_method="qr",
        use_ic_weighting=False,
    )
    navs["v6.2_QR_eq"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 QR 等权 (QR panel alpha)", navs["v6.2_QR_eq"])

    # 3. v6.2 no_orth IC12 (Stage 27 baseline)
    print("\n[3/4] v6.2 no_orth IC12 (Stage 27 baseline)")
    cfg = V6_2Config(
        ic_min_months=12,
        use_orthogonal=False,
    )
    navs["v6.2_no_orth"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 no_orth IC12 (baseline)", navs["v6.2_no_orth"])

    # 4. v6.2 ir_full DEPRECATED 对照 (从历史 CSV)
    print("\n[4/4] v6.2 ir_full (DEPRECATED, 历史 CSV)")
    csv_path = REPO / "reports/momentum_etf_rotation/combo/v6_2_phase3_qr_ablation.parquet"
    if csv_path.exists():
        from pandas import read_parquet
        old = read_parquet(csv_path)
        if "v6.2_ir_full" in old.columns:
            navs["v6.2_ir_full"] = old["v6.2_ir_full"]
            report("v6.2 ir_full (DEPRECATED, 历史 CSV)", navs["v6.2_ir_full"])
        else:
            print("  [skip] 历史 CSV 无 ir_full 列")
    else:
        print("  [skip] 历史 CSV 不存在")

    # v6.1 baselines
    print("\n[5/6] v6.1 IC12 baseline")
    cfg = V6_1Config(ic_min_months=12)
    navs["v6.1_IC12"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 IC12 baseline", navs["v6.1_IC12"])

    print("\n[6/6] v6.1 等权 baseline")
    cfg = V6_1Config(use_ic_weighting=False)
    navs["v6.1_eq"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 等权", navs["v6.1_eq"])

    # HS300 benchmark
    print("\n[7] HS300 (510300) benchmark")
    if "510300" in panel_close.columns:
        hs = panel_close["510300"].loc["2018-01-01":OOS_END]
        hs_norm = hs / hs.iloc[0]
        navs["HS300"] = hs_norm
        report("HS300 (510300)", navs["HS300"])

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(navs)
    df.to_parquet(out_dir / "v6_2_phase3_qr_ablation.parquet")
    print(f"\n[save] {out_dir / 'v6_2_phase3_qr_ablation.parquet'} ({df.shape[1]} cols)")

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
    print("\n=== Phase 3 QR 综合对比 (按 OOS Calmar 排序) ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "v6_2_phase3_qr_metrics.csv", index=False)

    # 决策
    main_row = summary[summary["ablation"] == "v6.2_QR"]
    if len(main_row) > 0:
        main_calmar = main_row.iloc[0]["oos_calmar"]
        print(f"\n=== Phase 3 决策 ===")
        print(f"v6.2 QR OOS Calmar = {main_calmar:.3f}")
        if main_calmar >= EXPECT_MAIN:
            print(f"✅ ≥ {EXPECT_MAIN} → 锁定 sort_method='qr' 为 v6.2 默认")
        elif main_calmar >= 0.70:
            print(f"⚠ 在 [0.70, {EXPECT_MAIN}) → 仍接受作为主推")
        else:
            print(f"❌ < 0.70 → QR 也退化, 退回到 v6.1 IC12 (0.748) 作为最稳路径")


if __name__ == "__main__":
    main()

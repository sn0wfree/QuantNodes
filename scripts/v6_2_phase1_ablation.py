# coding=utf-8
"""v6.2 Phase 1 ablation 测试 (expanding IR Gram-Schmidt 主推).

[Phase 1 决策] v6.2 路径选 expanding IR 排序 (无 look-ahead) 作为默认主推.
评测:
  1. v6.2 expanding IR (主推, 无 look-ahead)
  2. v6.2 predefined (Stage 28 试过的金融预定义, OOS 0.473)
  3. v6.2 ir_full (Stage 27 冠军 DEPRECATED, 含 look-ahead, 历史 OOS 0.901)
  4. v6.2 no_orth (无正交, 等于 v6.1 IC12 baseline, OOS 0.748)

如果 v6.2 expanding IR OOS Calmar ≥ 0.85 → 锁定为 v6.2 默认
否则 → fallback 到 Phase 3: QR 对称正交
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
    print(f"  {label:50s} OOS:Calmar={om['calmar']:.3f} ann={om['ann']:+.2%} "
          f"DD={om['dd']:.2%} Sharpe={om['sharpe']:.2f} | Full:Calmar={fm['calmar']:.3f}{flag}")
    return {"oos_calmar": om["calmar"], "oos_ann": om["ann"], "oos_dd": om["dd"],
            "oos_sharpe": om["sharpe"], "full_calmar": fm["calmar"]}


def main() -> None:
    print("[v6.2 Phase 1 ablation] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]
    print(f"  panel_close: {panel_close.shape}")
    print(f"  panel_ohlcv: {panel_ohlcv.shape}")

    navs = {}
    EXPECT_MAIN = 0.85  # Phase 1 主推最低标准

    # ──────── 4 组 ablation ────────
    print("\n[Phase 1 主消融 4 组]")

    # 1. v6.2 主推: expanding IR (无 look-ahead)
    print("\n[1/4] v6.2 expanding IR (Phase 1 主推, 无 look-ahead)")
    cfg = V6_2Config(
        ic_min_months=12,            # IC 加权 12 月 (Stage 27 最佳)
        sort_method="ir_expanding",
        ir_order_min_periods_lookback=36,  # IR 排序回看 36 月
    )
    navs["v6.2_ir_expanding"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 expanding IR (Phase 1 主推)",
           navs["v6.2_ir_expanding"], EXPECT_MAIN)

    # 2. v6.2 预定义金融顺序 (Stage 28 已试过)
    print("\n[2/4] v6.2 predefined (Stage 28 金融预定义, 对照)")
    cfg = V6_2Config(
        ic_min_months=12,
        sort_method="predefined",
    )
    navs["v6.2_predefined"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 predefined (金融预定义)",
           navs["v6.2_predefined"])

    # 3. v6.2 DEPRECATED ir_full (Stage 27 冠军, 含 look-ahead, 对照)
    print("\n[3/4] v6.2 ir_full (DEPRECATED, 含 look-ahead, ablation 对照)")
    print("  [Stage 29] ir_full 已从生产路径移出, 直接从历史 CSV 读取")
    csv_path = REPO / "reports/momentum_etf_rotation/combo/v6_2_phase1_ablation.parquet"
    if csv_path.exists():
        from pandas import read_parquet
        old = read_parquet(csv_path)
        if "v6.2_ir_full" in old.columns:
            navs["v6.2_ir_full"] = old["v6.2_ir_full"]
            report("v6.2 ir_full (DEPRECATED, 历史 CSV)",
                   navs["v6.2_ir_full"])
        else:
            print("  [skip] 历史 CSV 无 ir_full 列")
    else:
        print("  [skip] 历史 CSV 不存在")

    # 4. v6.2 不正交 (baseline, 等同 v6.1 IC12)
    print("\n[4/4] v6.2 no_orth (不正交, baseline)")
    cfg = V6_2Config(
        ic_min_months=12,
        use_orthogonal=False,
    )
    navs["v6.2_no_orth"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2 不正交 (baseline)",
           navs["v6.2_no_orth"])

    # 5. v6.1 IC12 baseline (Stage 27 老稳定)
    print("\n[5] v6.1 IC12 baseline (Stage 27 验证)")
    cfg = V6_1Config(ic_min_months=12)
    navs["v6.1_IC12"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 IC12 (Stage 27 baseline)", navs["v6.1_IC12"])

    # 6. v6.1 等权 baseline
    print("\n[6] v6.1 等权 baseline")
    cfg = V6_1Config(use_ic_weighting=False)
    navs["v6.1_eq"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1 等权", navs["v6.1_eq"])

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(navs)
    df.to_parquet(out_dir / "v6_2_phase1_ablation.parquet")
    print(f"\n[save] {out_dir / 'v6_2_phase1_ablation.parquet'} ({df.shape[1]} cols)")

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
    print("\n=== Phase 1 综合对比 (按 OOS Calmar 排序) ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "v6_2_phase1_metrics.csv", index=False)

    # 检查主推
    main_row = summary[summary["ablation"] == "v6.2_ir_expanding"]
    if len(main_row) > 0:
        main_calmar = main_row.iloc[0]["oos_calmar"]
        print(f"\n=== 决策 ===")
        print(f"主推 v6.2_ir_expanding OOS Calmar = {main_calmar:.3f}")
        if main_calmar >= EXPECT_MAIN:
            print(f"✅ ≥ {EXPECT_MAIN} → 锁定 v6.2 expanding IR 为默认")
            print(f"   后续 Phase 3 (QR) 转为可选 fallback, 不在主推路径")
        elif main_calmar >= 0.70:
            print(f"⚠ {main_calmar:.3f} 在 [{0.70}, {EXPECT_MAIN}) → 仍接受但不够稳定")
            print(f"   进入 Phase 3 (QR 对称正交), 看能否突破 {EXPECT_MAIN}")
        else:
            print(f"❌ < 0.70 → 严重退化, 直接 fallback 到 Phase 3 (QR 对称正交)")


if __name__ == "__main__":
    main()

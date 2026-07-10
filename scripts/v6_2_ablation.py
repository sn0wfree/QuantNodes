# coding=utf-8
"""v6.2 消融实验 (Stage 27 v6.2): 对比正交化与基线 v6.1.

5 组消融:
1. v6.1 baseline 等权 (无 IC 加权, 无正交)
2. v6.1 IC12 (推荐配置, 不正交)
3. v6.2 正交 + IC12 (主推)
4. v6.2 正交 + IC24
5. v6.2 正交 + IC36

评估指标: OOS Calmar / OOS DD / OOS Sharpe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0, "end": 0.0}
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    sharpe = ann / vol if vol > 0 else 0
    calmar = ann / abs(dd) if dd != 0 else 0
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "dd": dd, "calmar": calmar, "end": s.iloc[-1]}


IS_END = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END = "2026-06-30"


def report(label: str, nav: pd.Series) -> dict:
    fm = metrics(nav)
    om = metrics(nav.loc[OOS_START:OOS_END])
    print(f"  {label:40s} OOS:Calmar={om['calmar']:.3f} ann={om['ann']:+.2%} "
          f"DD={om['dd']:.2%} Sharpe={om['sharpe']:.2f} | Full:Calmar={fm['calmar']:.3f}")
    return {"oos_calmar": om["calmar"], "oos_ann": om["ann"], "oos_dd": om["dd"],
            "oos_sharpe": om["sharpe"], "full_calmar": fm["calmar"], "full_ann": fm["ann"]}


def main() -> None:
    print("[v6.2] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]
    print(f"  panel_close: {panel_close.shape}")
    print(f"  panel_ohlcv: {panel_ohlcv.shape}")

    navs = {}

    # 1. v6.1 等权 baseline
    print("\n[1] v6.1_baseline_eq (等权)")
    cfg = V6_1Config(use_ic_weighting=False)
    navs["v6.1_baseline_eq"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1_baseline_eq (等权)", navs["v6.1_baseline_eq"])

    # 2. v6.1 IC12 (推荐, 不正交)
    print("\n[2] v6.1_IC12 (IC12, 不正交)")
    cfg = V6_1Config(ic_min_months=12)
    navs["v6.1_IC12_no_orth"] = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.1_IC12 (不正交)", navs["v6.1_IC12_no_orth"])

    # 3. v6.2 正交 + IC12
    print("\n[3] v6.2_orth_IC12 (推荐配置)")
    cfg = V6_2Config(ic_min_months=12, use_orthogonal=True)
    navs["v6.2_orth_IC12"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2_orth_IC12 (主推)", navs["v6.2_orth_IC12"])

    # 4. v6.2 正交 + IC24
    print("\n[4] v6.2_orth_IC24")
    cfg = V6_2Config(ic_min_months=24, use_orthogonal=True)
    navs["v6.2_orth_IC24"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2_orth_IC24", navs["v6.2_orth_IC24"])

    # 5. v6.2 正交 + IC36
    print("\n[5] v6.2_orth_IC36")
    cfg = V6_2Config(ic_min_months=36, use_orthogonal=True)
    navs["v6.2_orth_IC36"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2_orth_IC36", navs["v6.2_orth_IC36"])

    # 6. v6.2 不正交 + IC12 (看纯 IC 加权 + 不正交)
    print("\n[6] v6.2_no_orth_IC12")
    cfg = V6_2Config(ic_min_months=12, use_orthogonal=False)
    navs["v6.2_no_orth_IC12"] = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    report("v6.2_no_orth_IC12", navs["v6.2_no_orth_IC12"])

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(navs)
    df.to_parquet(out_dir / "v6_2_ablation_navs.parquet")
    print(f"\n[save] {out_dir / 'v6_2_ablation_navs.parquet'} ({df.shape[1]} cols, {df.shape[0]} rows)")

    print("\n=== v6.2 消融综合对比 ===")
    rows = []
    for col in df.columns:
        om = metrics(df[col].loc[OOS_START:OOS_END])
        fm = metrics(df[col])
        rows.append({
            "ablation": col,
            "oos_calmar": om["calmar"],
            "oos_ann": om["ann"],
            "oos_dd": om["dd"],
            "oos_sharpe": om["sharpe"],
            "full_calmar": fm["calmar"],
        })
    summary = pd.DataFrame(rows).sort_values("oos_calmar", ascending=False)
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "v6_2_ablation_metrics.csv", index=False)
    print(f"\n[save] {out_dir / 'v6_2_ablation_metrics.csv'}")

    best = summary.iloc[0]
    print(f"\n⭐ Best: {best['ablation']}")
    print(f"   OOS Calmar {best['oos_calmar']:.3f}, ann {best['oos_ann']:+.2%}, DD {best['oos_dd']:.2%}, Sharpe {best['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()

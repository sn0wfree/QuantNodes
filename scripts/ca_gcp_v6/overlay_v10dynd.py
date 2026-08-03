"""CA-GCP v6 风险预算 overlay v10-DynD (Phase 1.7).

目的: 验证 v6 risk_budget 能否改善 v10-DynD OOS 业绩.

逻辑:
  baseline = v10-DynD 信号加权 NAV
  overlay  = v10-DynD × (1 - risk_budget)

若 v6 风险预测可靠:
  - 大盘下跌前期 risk_budget → 0.2 → 仓位降到 80%
  - 平稳期 risk_budget → 0 → 全仓 v10-DynD
  - 整体应改善 Calmar (MaxDD 改善) 且 Sharpe 持平或上升
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics  # noqa: E402

V10_DIR = ROOT / "reports" / "momentum_etf_rotation" / "v10"
V6_DIR = ROOT / "reports" / "momentum_etf_rotation" / "ca_gcp_ew"
COMBO_DIR = ROOT / "reports" / "momentum_etf_rotation" / "combo"
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-06-30")


def to_business_day_index(nav: pd.Series) -> pd.Series:
    """强制 business-day-only."""
    s = nav.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    biz = pd.bdate_range(start=s.index.min(), end=s.index.max())
    s = s.reindex(biz).dropna()
    return s


def load_dynd_nav() -> pd.Series:
    """加载 v10-DynD 信号加权 NAV."""
    path = V10_DIR / "dynamic_nav_D_signal_weighted.parquet"
    df = pd.read_parquet(path)
    s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    s = s / s.iloc[0]
    s.name = "v10-DynD"
    return s


def load_v6_risk_budget() -> pd.Series:
    """加载 v6 风险预算."""
    path = V6_DIR / "ca_gcp_v6_risk_budget.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df["risk_budget"]


def apply_overlay(
    nav_baseline: pd.Series,
    risk_budget: pd.Series,
    rebalance_freq: int = 5,
) -> pd.Series:
    """对 baseline NAV 应用 risk_budget overlay.

    overlay 仓位 = 1 - risk_budget (前向填充)
    调仓频率 = 每 rebalance_freq 天
    """
    common = nav_baseline.index.intersection(risk_budget.index)
    nav_aligned = nav_baseline.loc[common]
    rb_aligned = risk_budget.reindex(common).ffill().fillna(0.0)

    exposure = 1 - rb_aligned.values
    # Rebalance: 每 5 天调整一次 exposure
    exposure = pd.Series(exposure, index=common)
    rebal_mask = exposure.index[::rebalance_freq]
    exposure_smooth = exposure.copy()
    exposure_smooth.loc[~exposure_smooth.index.isin(rebal_mask)] = np.nan
    exposure_smooth = exposure_smooth.ffill().fillna(1.0)
    exposure_smooth = exposure_smooth.clip(0.0, 1.0)

    # 计算 overlay NAV
    nav_overlay = pd.Series(1.0, index=common)
    baseline_rets = nav_aligned.pct_change().fillna(0.0)
    overlay_rets = baseline_rets * exposure_smooth

    for i in range(1, len(common)):
        nav_overlay.iloc[i] = nav_overlay.iloc[i - 1] * (1 + overlay_rets.iloc[i])

    return nav_overlay


def evaluate_overlay(
    nav_baseline: pd.Series,
    nav_overlay: pd.Series,
    label: str = "v10-DynD + v6 risk_budget",
) -> dict:
    """评估 overlay vs baseline."""
    common = nav_baseline.index.intersection(nav_overlay.index)
    nav_b = nav_baseline.loc[common]
    nav_o = nav_overlay.loc[common]
    oos_b = nav_b.loc[OOS_START:]
    oos_o = nav_o.loc[OOS_START:]

    m_b = compute_metrics(oos_b)
    m_o = compute_metrics(oos_o)
    return {
        "label": label,
        "baseline": {
            "Sharpe": m_b.get("Sharpe", 0),
            "Calmar": m_b.get("Calmar", 0),
            "AnnRet": m_b.get("AnnRet", 0),
            "MaxDD": m_b.get("MaxDD", 0),
            "final": float(nav_b.iloc[-1]),
        },
        "overlay": {
            "Sharpe": m_o.get("Sharpe", 0),
            "Calmar": m_o.get("Calmar", 0),
            "AnnRet": m_o.get("AnnRet", 0),
            "MaxDD": m_o.get("MaxDD", 0),
            "final": float(nav_o.iloc[-1]),
        },
        "delta": {
            "Sharpe": m_o.get("Sharpe", 0) - m_b.get("Sharpe", 0),
            "Calmar": m_o.get("Calmar", 0) - m_b.get("Calmar", 0),
            "AnnRet": m_o.get("AnnRet", 0) - m_b.get("AnnRet", 0),
            "MaxDD": m_o.get("MaxDD", 0) - m_b.get("MaxDD", 0),
        },
    }


def main() -> None:
    print("=" * 70)
    print("CA-GCP v6 风险预算 overlay v10-DynD (Phase 1.7)")
    print("=" * 70)

    nav_baseline = to_business_day_index(load_dynd_nav())
    risk_budget = to_business_day_index(load_v6_risk_budget())

    print(f"\n  baseline: {nav_baseline.index[0].date()} ~ {nav_baseline.index[-1].date()}, "
          f"{len(nav_baseline)} 天")
    print(f"  risk_budget: {risk_budget.index[0].date()} ~ {risk_budget.index[-1].date()}, "
          f"{len(risk_budget)} 天")

    nav_overlay = apply_overlay(nav_baseline, risk_budget)
    result = evaluate_overlay(nav_baseline, nav_overlay)

    print(f"\n{'='*70}")
    print(f"OOS (2022-01-01 ~ 2026-06-30) 业绩对比")
    print("=" * 70)
    print(f"\n{'':20s} {'Baseline':>12s} {'Overlay':>12s} {'Delta':>10s}")
    print(f"{'Sharpe':20s} {result['baseline']['Sharpe']:>12.3f} "
          f"{result['overlay']['Sharpe']:>12.3f} {result['delta']['Sharpe']:>+10.3f}")
    print(f"{'Calmar':20s} {result['baseline']['Calmar']:>12.3f} "
          f"{result['overlay']['Calmar']:>12.3f} {result['delta']['Calmar']:>+10.3f}")
    print(f"{'AnnRet':20s} {result['baseline']['AnnRet']*100:>11.2f}% "
          f"{result['overlay']['AnnRet']*100:>11.2f}% {result['delta']['AnnRet']*100:>+9.2f}%")
    print(f"{'MaxDD':20s} {result['baseline']['MaxDD']*100:>11.2f}% "
          f"{result['overlay']['MaxDD']*100:>11.2f}% {result['delta']['MaxDD']*100:>+9.2f}%")

    # 决定: 是否值得采纳 overlay
    sharp_improve = result['delta']['Sharpe'] > 0
    calmar_improve = result['delta']['Calmar'] > 0
    dd_better = result['delta']['MaxDD'] > 0  # 越接近 0 越好 (MaxDD 是负数)

    print(f"\n  Sharpe 改善: {'✓' if sharp_improve else '✗'} ({result['delta']['Sharpe']:+.3f})")
    print(f"  Calmar 改善: {'✓' if calmar_improve else '✗'} ({result['delta']['Calmar']:+.3f})")
    print(f"  MaxDD 改善:  {'✓' if dd_better else '✗'} ({result['delta']['MaxDD']*100:+.2f}%)")

    # 保存 overlay NAV
    out_path = COMBO_DIR / "v10_dynd_v6_overlay.parquet"
    nav_overlay.to_frame("nav").to_parquet(out_path)
    print(f"\n[保存] {out_path}")

    # 报告
    verdict = "采纳" if (sharp_improve or calmar_improve) and dd_better else "不采纳"
    report = [
        "# CA-GCP v6 Overlay v10-DynD 报告",
        "",
        "## 1. OOS 业绩对比 (2022-01-01 ~ 2026-06-30)",
        "",
        "| 指标 | Baseline (v10-DynD) | Overlay (+ v6 risk_budget) | Delta |",
        "|---|---|---|---|",
        f"| Sharpe | {result['baseline']['Sharpe']:.3f} | {result['overlay']['Sharpe']:.3f} | {result['delta']['Sharpe']:+.3f} |",
        f"| Calmar | {result['baseline']['Calmar']:.3f} | {result['overlay']['Calmar']:.3f} | {result['delta']['Calmar']:+.3f} |",
        f"| AnnRet | {result['baseline']['AnnRet']*100:.2f}% | {result['overlay']['AnnRet']*100:.2f}% | {result['delta']['AnnRet']*100:+.2f}% |",
        f"| MaxDD | {result['baseline']['MaxDD']*100:.2f}% | {result['overlay']['MaxDD']*100:.2f}% | {result['delta']['MaxDD']*100:+.2f}% |",
        "",
        "## 2. 决定",
        "",
        f"**{verdict}**",
        "",
        f"- Sharpe 改善: {result['delta']['Sharpe']:+.3f}",
        f"- Calmar 改善: {result['delta']['Calmar']:+.3f}",
        f"- MaxDD 改善:  {result['delta']['MaxDD']*100:+.2f}%",
        "",
        "## 3. 限制",
        "",
        "- v6 AUC 接近 0.5 (随机), 因此 risk_budget 信号弱",
        "- Phase 1.6 walk-forward 0% is_downside recall, 模型过于保守",
        "- Overlay 改善可能来自纯降仓 (暴露降低) 而非 v6 信号",
    ]
    (V6_DIR / "ca_gcp_v6_overlay_report.md").write_text("\n".join(report))
    print(f"[报告] {V6_DIR / 'ca_gcp_v6_overlay_report.md'}")


if __name__ == "__main__":
    main()

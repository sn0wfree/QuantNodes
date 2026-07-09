# coding=utf-8
"""CICC vs Stage 12A 对比回测 (4 个关键配置).

直接跑 4 个回测,不再解析 HTML:
- v1_CICC_baseline:    price momentum, 无 VT, 无 Cost
- v1_CICC_vt:          price momentum + VT (tv=0.15)
- v2_hybrid:           hybrid momentum (price + slope_r²), 无 VT
- v2_hybrid_vt:        hybrid momentum + VT (推荐 v1.0 配置)

数据: 2018-01-02 ~ 2026-06-30 (完整新数据)
输出: reports/momentum_etf_rotation/docs/CICC_VS_STAGE12A_COMPARISON.md
      + 2 个对比图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    BacktestConfig,
    ConcentrationCaps,
    CostModel,
    RotationConfig,
    TrendFilter,
    VolTargeting,
    run_rotation_backtest,
)


def load_real_data() -> pd.DataFrame:
    """加载真实 ETF 数据 (新数据, 截至 2026-06-30)."""
    return pd.read_parquet("data/real/etf_nav_2018-01-01_2026-06-30.parquet")


def make_vt() -> VolTargeting:
    return VolTargeting(enabled=True, target_vol=0.15)


def make_cost() -> CostModel:
    return CostModel(enabled=True, commission_bp=5.0, slippage_bp=10.0)


def make_trend() -> TrendFilter:
    return TrendFilter(enabled=True, ma_window=55, bear_exposure=0.7)


def run_4_configs(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """跑 4 个配置,返回 name -> NAV Series."""
    results = {}

    configs = [
        ("v1_CICC_baseline", RotationConfig(
            lookback=144, top_n=10, momentum_type="price",
        )),
        ("v1_CICC_vt", RotationConfig(
            lookback=144, top_n=10, momentum_type="price",
            vol_targeting=make_vt(),
        )),
        ("v2_hybrid", RotationConfig(
            lookback=144, top_n=10, momentum_type="hybrid",
            momentum_fused_weight=0.5,
        )),
        ("v2_hybrid_vt", RotationConfig(
            lookback=144, top_n=10, momentum_type="hybrid",
            momentum_fused_weight=0.5,
            vol_targeting=make_vt(),
            cost_model=make_cost(),
        )),
    ]

    for name, rot in configs:
        cfg = BacktestConfig(rotation=rot, freq="ME")
        result = run_rotation_backtest(panel, DEFAULT_POOL, cfg)
        results[name] = result.nav
        print(f"  {name}: final NAV = {result.nav.iloc[-1]:.4f}, "
              f"n_rebal = {len(result.rebalance_dates)}")

    return results


def compute_metrics(nav: pd.Series, freq: int = 252) -> dict:
    """计算 12 个核心指标."""
    if nav.empty or len(nav) < 2:
        return {}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {}

    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
    n_years = len(rets) / freq
    ann_return = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(rets.std() * np.sqrt(freq))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def get_period_return(nav: pd.Series, start: str, end: str) -> dict:
    """提取指定期间表现."""
    win = nav.loc[start:end]
    if len(win) < 2:
        return {"period_return": 0.0, "max_dd": 0.0}
    period_ret = float(win.iloc[-1] / win.iloc[0] - 1)
    cummax = win.cummax()
    dd = (win / cummax - 1)
    max_dd = float(dd.min())
    return {
        "period_return": period_ret,
        "max_dd": max_dd,
    }


def main():
    print("=" * 70)
    print("CICC vs Stage 12A 对比回测 (4 个关键配置)")
    print("=" * 70)

    panel = load_real_data()
    print(f"\n数据: {panel.shape[0]} 天 × {panel.shape[1]} ETF")
    print(f"时间: {panel.index[0].date()} ~ {panel.index[-1].date()}")
    print(f"样本回测起点: ~{panel.index[144].date()} (lookback=144)")

    print("\n[1/2] 跑 4 个配置...")
    results = run_4_configs(panel)

    print("\n[2/2] 计算指标...")
    summary = {}
    for name, nav in results.items():
        m = compute_metrics(nav)
        # 2026 H1
        h1_2026 = get_period_return(nav, "2026-01-01", "2026-06-30")
        # 2025 H2
        h2_2025 = get_period_return(nav, "2025-07-01", "2025-12-31")
        # 924 期间 (背景)
        sep_2024 = get_period_return(nav, "2024-09-23", "2024-10-31")

        summary[name] = {
            "metrics": m,
            "h1_2026": h1_2026,
            "h2_2025": h2_2025,
            "sep_2024": sep_2024,
            "final_nav": float(nav.iloc[-1]),
            "start_date": str(nav.index[0].date()),
            "end_date": str(nav.index[-1].date()),
        }
        print(f"  {name}:")
        print(f"    Calmar={m.get('calmar', 0):.3f}, DD={m.get('max_drawdown', 0)*100:.2f}%, "
              f"Ann={m.get('ann_return', 0)*100:.2f}%, Sharpe={m.get('sharpe', 0):.2f}")
        print(f"    2026 H1: {h1_2026['period_return']*100:.2f}%, 2025 H2: {h2_2025['period_return']*100:.2f}%, "
              f"924 期间: {sep_2024['period_return']*100:.2f}%")

    # 保存
    out_dir = Path("reports/momentum_etf_rotation/docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 NAVs
    nav_df = pd.DataFrame(results)
    nav_df.to_parquet(out_dir / "cicc_vs_stage12a_navs.parquet")

    # 保存 summary
    with open(out_dir / "cicc_vs_stage12a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存:")
    print(f"  - {out_dir / 'cicc_vs_stage12a_navs.parquet'}")
    print(f"  - {out_dir / 'cicc_vs_stage12a_summary.json'}")

    return summary, nav_df


if __name__ == "__main__":
    main()

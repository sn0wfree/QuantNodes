# coding=utf-8
"""Stage 16A: 多策略组合回测验证 (真实数据).

对比 v3 多策略 vs v2 单策略:
- 全周期: 2018-01-02 ~ 2026-06-30
- 关键指标: Calmar, DD, Sharpe, Ann Return
- 924 专项: 2024-09-24 ~ 2024-10-31

输出: reports/momentum_etf_rotation/v3/stage16a_validation.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from QuantNodes.strategy.momentum_etf_rotation.common import DEFAULT_POOL
from QuantNodes.strategy.momentum_etf_rotation.v3 import (
    MultiStrategyConfig,
    run_multi_strategy_backtest,
)


def load_real_data() -> pd.DataFrame:
    """加载真实 ETF 数据."""
    return pd.read_parquet("data/real/etf_nav_2018-01-01_2026-06-30.parquet")


def get_924_window_nav(nav: pd.Series) -> dict:
    """提取 924 期间关键指标."""
    win = nav.loc["2024-09-23":"2024-10-31"]
    if len(win) < 2:
        return {"period_return": 0.0, "peak_day": "", "peak_return": 0.0}
    period_return = float(win.iloc[-1] / win.iloc[0] - 1)
    daily_rets = win.pct_change().dropna()
    peak_idx = daily_rets.idxmax()
    peak_return = float(daily_rets.max())
    return {
        "period_return": period_return,
        "peak_day": str(peak_idx.date()) if pd.notna(peak_idx) else "",
        "peak_return": peak_return,
    }


def run_v2_baseline(panel: pd.DataFrame) -> dict:
    """v2 baseline (从 v2 复用 select_and_weight_v2)."""
    from QuantNodes.strategy.momentum_etf_rotation import (
        BacktestConfig, RotationConfig, run_rotation_backtest,
    )
    rot = RotationConfig(
        lookback=144, top_n=10, momentum_type="hybrid",
        momentum_fused_weight=0.5,
    )
    cfg = BacktestConfig(rotation=rot, freq="ME")
    result = run_rotation_backtest(panel, DEFAULT_POOL, cfg)
    return {
        "nav": result.nav,
        "metrics": result.metrics,
        "924": get_924_window_nav(result.nav),
    }


def run_v3_equal(panel: pd.DataFrame) -> dict:
    """v3 等权子策略."""
    cfg = MultiStrategyConfig(weight_method="equal")
    result = run_multi_strategy_backtest(panel, DEFAULT_POOL, cfg)
    return {
        "nav": result.nav,
        "metrics": result.metrics,
        "924": get_924_window_nav(result.nav),
    }


def run_v3_signal(panel: pd.DataFrame) -> dict:
    """v3 信号加权."""
    cfg = MultiStrategyConfig(weight_method="signal")
    result = run_multi_strategy_backtest(panel, DEFAULT_POOL, cfg)
    return {
        "nav": result.nav,
        "metrics": result.metrics,
        "924": get_924_window_nav(result.nav),
    }


def main():
    print("=" * 70)
    print("Stage 16A 验证: 多策略组合 (v3) vs 单策略 (v2)")
    print("=" * 70)

    panel = load_real_data()
    print(f"\n数据: {panel.shape[0]} 天 × {panel.shape[1]} ETF")
    print(f"时间: {panel.index[0].date()} ~ {panel.index[-1].date()}")

    results = {}

    print("\n[1/3] 运行 v2 baseline (Stage 12A)...")
    v2 = run_v2_baseline(panel)
    results["v2"] = v2
    m = v2["metrics"]
    print(f"  v2: Calmar={m.get('calmar', 0):.3f}, DD={m.get('max_drawdown', 0)*100:.2f}%, "
          f"Ann={m.get('ann_return', 0)*100:.2f}%, Sharpe={m.get('sharpe', 0):.2f}")
    print(f"  924 期间: 收益={v2['924']['period_return']*100:.2f}%, 峰值日={v2['924']['peak_day']}")

    print("\n[2/3] 运行 v3 (等权)...")
    v3_eq = run_v3_equal(panel)
    results["v3_equal"] = v3_eq
    m = v3_eq["metrics"]
    print(f"  v3_equal: Calmar={m.get('calmar', 0):.3f}, DD={m.get('max_drawdown', 0)*100:.2f}%, "
          f"Ann={m.get('ann_return', 0)*100:.2f}%, Sharpe={m.get('sharpe', 0):.2f}")
    print(f"  924 期间: 收益={v3_eq['924']['period_return']*100:.2f}%")

    print("\n[3/3] 运行 v3 (信号加权)...")
    v3_sig = run_v3_signal(panel)
    results["v3_signal"] = v3_sig
    m = v3_sig["metrics"]
    print(f"  v3_signal: Calmar={m.get('calmar', 0):.3f}, DD={m.get('max_drawdown', 0)*100:.2f}%, "
          f"Ann={m.get('ann_return', 0)*100:.2f}%, Sharpe={m.get('sharpe', 0):.2f}")
    print(f"  924 期间: 收益={v3_sig['924']['period_return']*100:.2f}%")

    # 对比
    print("\n" + "=" * 70)
    print("对比表")
    print("=" * 70)
    print(f"{'配置':<15} {'Calmar':>8} {'DD':>8} {'Ann':>8} {'Sharpe':>8} {'924':>8}")
    print("-" * 70)
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<15} {m.get('calmar', 0):>8.3f} {m.get('max_drawdown', 0)*100:>7.2f}% "
              f"{m.get('ann_return', 0)*100:>7.2f}% {m.get('sharpe', 0):>8.2f} "
              f"{r['924']['period_return']*100:>7.2f}%")

    # 保存
    out_dir = Path("reports/momentum_etf_rotation/v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 NAV 序列为 parquet (后续画图)
    nav_df = pd.DataFrame({
        name: r["nav"] for name, r in results.items()
    })
    nav_df.to_parquet(out_dir / "stage16a_navs.parquet")

    # 保存 summary 为 JSON
    summary = {
        name: {
            "metrics": r["metrics"],
            "924": r["924"],
        }
        for name, r in results.items()
    }
    with open(out_dir / "stage16a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存:")
    print(f"  - {out_dir / 'stage16a_navs.parquet'}")
    print(f"  - {out_dir / 'stage16a_summary.json'}")


if __name__ == "__main__":
    main()

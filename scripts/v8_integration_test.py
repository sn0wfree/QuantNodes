#!/usr/bin/env python3
"""v8 Jump Model 集成到 v7.14 TV-PR 框架测试脚本.

测试三种集成方案:
  A. 市场状态叠加 (regime_overlay)
  B. 仓位调节 (position_sizing)
  C. 混合信号 (hybrid)

OOS 测试: 2022-02-17 ~ 2026-06-30 (4.2 年)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v8.integration import (
    regime_overlay_weights,
    position_sizing_weights,
    hybrid_signal_weights,
    backtest_v8_integration,
)
from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import jump_model_rolling
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio_components,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import expanding_window_tvpr
from QuantNodes.strategy.momentum_etf_rotation.v7.adapters import load_v7_14_data_uniform


# ============================================================
# 参数配置
# ============================================================
PARAMS = {
    "jump_penalty": 50.0,
    "train_window": 1000,
    "retrain_every": 30,
    "min_duration": 60,
    "reduce_ratio": 0.5,
    "bear_threshold": 0.3,
    "regime_weight": 0.5,
}

# 测试区间
OOS_START = "2022-02-17"
OOS_END = "2026-06-30"


# ============================================================
# 工具函数
# ============================================================
def calc_metrics(nav: pd.Series, name: str) -> dict:
    """计算性能指标."""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}

    # 年化收益
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1

    # 年化波动率
    daily_returns = nav.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252)

    # 夏普比率
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()

    # 卡玛比率
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "name": name,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "years": years,
    }


def run_v714_baseline() -> pd.Series:
    """运行 v7.14 基准回测."""
    print("  运行 v7.14 基准...")

    # 加载数据
    X, Y, codes = load_v7_14_data_uniform()

    # 配置
    cfg = V7_6Config(
        top_n=10,
        vol_window=26,
        max_weight=0.25,
        lambda_tv=0.15,
        lambda_l1=0.05,
        step=13,
        method="expanding",
        cost_enabled=False,
    )

    # TV-PR beta 估计
    beta = expanding_window_tvpr(
        Y, X,
        cfg.lambda_tv, cfg.lambda_l1,
        min_history=cfg.min_history,
        step=cfg.step,
    )

    # 构造组合
    shares, prices, weekly_weights = construct_portfolio_components(
        Y, X, beta, cfg,
    )

    # 加载日频收益
    HF_DIR = REPO / "data" / "high_freq_macro"
    daily_returns = pd.read_parquet(HF_DIR / "v7_6_daily_etf_returns.parquet")

    # 用与集成方案相同的方法计算 NAV
    from QuantNodes.strategy.momentum_etf_rotation.v8.integration import _compute_daily_nav_from_weights
    nav = _compute_daily_nav_from_weights(weekly_weights, daily_returns)

    return nav


def run_integration_test(method: str) -> pd.Series:
    """运行集成方案回测."""
    print(f"  运行 {method} 集成方案...")

    nav, weekly_weights, adjusted_weights = backtest_v8_integration(
        version="v7.14",
        integration_method=method,
        **PARAMS,
    )

    return nav


# ============================================================
# 主测试流程
# ============================================================
def main():
    """主测试流程."""
    print("=" * 60)
    print("v8 Jump Model 集成到 v7.14 TV-PR 框架测试")
    print("=" * 60)
    print()
    print("测试参数:")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")
    print()
    print("测试区间:")
    print(f"  OOS: {OOS_START} ~ {OOS_END}")
    print()

    # 1. 运行 v7.14 基准
    print("1. 运行 v7.14 基准")
    nav_baseline = run_v714_baseline()

    # 2. 运行三种集成方案
    print("\n2. 运行集成方案")
    nav_regime = run_integration_test("regime_overlay")
    nav_position = run_integration_test("position_sizing")
    nav_hybrid = run_integration_test("hybrid")

    # 3. 截取 OOS 区间
    print("\n3. 截取 OOS 区间")
    nav_baseline_oos = nav_baseline.loc[OOS_START:OOS_END]
    nav_regime_oos = nav_regime.loc[OOS_START:OOS_END]
    nav_position_oos = nav_position.loc[OOS_START:OOS_END]
    nav_hybrid_oos = nav_hybrid.loc[OOS_START:OOS_END]

    # 归一化 NAV 到 OOS 起点 = 1.0
    nav_baseline_oos = nav_baseline_oos / nav_baseline_oos.iloc[0]
    nav_regime_oos = nav_regime_oos / nav_regime_oos.iloc[0]
    nav_position_oos = nav_position_oos / nav_position_oos.iloc[0]
    nav_hybrid_oos = nav_hybrid_oos / nav_hybrid_oos.iloc[0]

    # 4. 计算性能指标
    print("\n4. 计算性能指标")
    metrics = []
    metrics.append(calc_metrics(nav_baseline_oos, "v7.14 基准"))
    metrics.append(calc_metrics(nav_regime_oos, "方案 A: 市场状态叠加"))
    metrics.append(calc_metrics(nav_position_oos, "方案 B: 仓位调节"))
    metrics.append(calc_metrics(nav_hybrid_oos, "方案 C: 混合信号"))

    # 5. 生成报告
    print("\n5. 生成报告")

    # 控制台输出
    print("\n" + "=" * 60)
    print("测试结果对比")
    print("=" * 60)
    print()
    print(f"{'策略':<25} {'AnnRet':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10} {'Calmar':>10}")
    print("-" * 75)
    for m in metrics:
        print(f"{m['name']:<25} {m['ann_ret']*100:>9.2f}% {m['ann_vol']*100:>9.2f}% {m['sharpe']:>10.3f} {m['max_dd']*100:>9.2f}% {m['calmar']:>10.3f}")

    # 保存 CSV
    csv_path = REPO / "reports" / "momentum_etf_rotation" / "v8_integration_test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")

    # 保存 Markdown 报告
    md_path = REPO / "reports" / "momentum_etf_rotation" / "v8_integration_test.md"
    with open(md_path, "w") as f:
        f.write("# v8 Jump Model 集成到 v7.14 TV-PR 框架测试报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## 1. 测试参数\n\n")
        f.write("| 参数 | 值 |\n|------|----|\n")
        for k, v in PARAMS.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n")

        f.write("## 2. 测试区间\n\n")
        f.write(f"- OOS: {OOS_START} ~ {OOS_END}\n\n")

        f.write("## 3. 测试结果\n\n")
        f.write("| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar |\n")
        f.write("|------|--------|-----|--------|-------|--------|\n")
        for m in metrics:
            f.write(f"| {m['name']} | {m['ann_ret']*100:.2f}% | {m['ann_vol']*100:.2f}% | {m['sharpe']:.3f} | {m['max_dd']*100:.2f}% | {m['calmar']:.3f} |\n")
        f.write("\n")

        f.write("## 4. 结论\n\n")
        # 找出最优方案
        best = max(metrics, key=lambda x: x['sharpe'])
        f.write(f"- **最优方案**: {best['name']} (Sharpe = {best['sharpe']:.3f})\n")
        f.write(f"- **vs v7.14 基准**: Sharpe 提升 {(best['sharpe'] - metrics[0]['sharpe']) / metrics[0]['sharpe'] * 100:.1f}%\n")
        f.write(f"- **MaxDD 改善**: {abs(best['max_dd']):.2f}% vs {abs(metrics[0]['max_dd']):.2f}%\n")

    print(f"  MD: {md_path}")

    # 保存 NAV 序列
    nav_path = REPO / "reports" / "momentum_etf_rotation" / "v8_integration_nav.csv"
    nav_df = pd.DataFrame({
        "date": nav_baseline_oos.index,
        "v7_14_baseline": nav_baseline_oos.values,
        "regime_overlay": nav_regime_oos.values,
        "position_sizing": nav_position_oos.values,
        "hybrid": nav_hybrid_oos.values,
    })
    nav_df.to_csv(nav_path, index=False)
    print(f"  NAV: {nav_path}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# coding=utf-8
"""v7.10 起点依赖 CV% 测试 (Stage 32 验收).

运行 v7.10 TV-PR (标准化+CV) 在 3 个不同起点, 计算 OOS Calmar 的变异系数 (CV%).
用于验证 v7.10 是否存在类似 v6.2 的起点依赖问题 (v6.2 CV% = 56.9% FAIL).

⚠️ 2026-07-28 修复: 每个起点独立加载原始数据 + 重新标准化,
  不再截断全量标准化后的数据 (expanding 标准化在截断起点时会偷看未来).

判定标准:
  CV% < 25% → PASS (v7.10 维持 RECOMMENDED)
  CV% 25-50% → PROMISING (降级, 需调参)
  CV% > 50% → DEPRECATED (复刻 v6.2 命运)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到 path
ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "QuantNodes").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_9_data,
    standardize_v7_10,
    load_daily_etf_returns,
    load_weekly_monday_open_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    run_v7_6_backtest,
    construct_portfolio,
    calculate_daily_nav,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics


# ============================================================
# 配置
# ============================================================
# 3 个起点 (年份)
START_DATES = [
    pd.Timestamp("2018-01-07"),  # 全量 (baseline)
    pd.Timestamp("2020-01-06"),  # 跳过 2 年
    pd.Timestamp("2022-01-03"),  # 跳过 4 年
]

# v7.10 最优参数 (Stage 31)
V710_CFG = V7_6Config(
    name="v7_10_std_newλ",
    lambda_tv=0.06,
    lambda_l1=0.105,
)


def run_single_start(start_date: pd.Timestamp, cfg: V7_6Config) -> dict:
    """单起点回测 — 加载原始数据, 截断后独立标准化."""
    print(f"\n{'='*60}")
    print(f"起点: {start_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    # 加载原始未标准化的 v7.9 数据
    X_raw, Y_raw, codes = load_v7_9_data()
    factor_names = (Path(__file__).resolve().parent.parent
                    / "data" / "high_freq_macro"
                    / "v7_9_factor_names.csv").read_text().strip().split("\n")[1:]

    # 按起点截断原始数据
    mask = np.array(Y_raw.index >= start_date)
    Y = Y_raw.loc[mask]
    idx = np.where(mask)[0]
    X_truncated = X_raw[idx]

    if len(Y) < 52:
        print(f"  ⚠️ 数据不足 52 周 (仅 {len(Y)} 周), 跳过")
        return {"name": start_date.strftime('%Y-%m-%d'), "error": "数据不足"}

    # 对截断后的数据独立标准化 (expanding window, 无未来函数)
    X = standardize_v7_10(X_truncated, factor_names)
    print(f"  数据: {len(Y)} 周, {X.shape[1]} 资产, {X.shape[2]} 因子")
    print(f"  标准化: 独立 expanding (仅用 [0:t] 数据)")

    # TV-PR 估计
    beta_path = tvpr_estimator(
        Y, X,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        method=cfg.method,
        min_history=cfg.min_history,
        rho=cfg.rho,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )
    print(f"  Beta 估计完成: {beta_path.shape}")

    # 组合构造
    nav = construct_portfolio(Y, X, beta_path, cfg)

    # 使用 common/metrics.py 计算指标
    m = compute_metrics(nav, freq='W')

    result = {
        "name": start_date.strftime('%Y-%m-%d'),
        "ann_return": m["AnnRet"],
        "ann_vol": m["Vol"],
        "sharpe": m["Sharpe"],
        "max_dd": m["MaxDD"],
        "calmar": m["Calmar"],
        "n_weeks": len(nav),
    }

    print(f"  结果 (common/metrics.py):")
    print(f"    年化收益: {result['ann_return']*100:+.2f}%")
    print(f"    Sharpe:   {result['sharpe']:.3f}")
    print(f"    最大回撤: {result['max_dd']*100:.2f}%")
    print(f"    Calmar:   {result['calmar']:.3f}")
    print(f"    周数:     {result['n_weeks']}")

    return result


def main():
    """运行 3 起点 CV% 测试."""
    print("=" * 60)
    print("v7.10 起点依赖 CV% 测试 (Stage 32)")
    print("=" * 60)
    print(f"参数: λ_tv={V710_CFG.lambda_tv}, λ_l1={V710_CFG.lambda_l1}")
    print(f"起点: {[d.strftime('%Y-%m-%d') for d in START_DATES]}")
    print(f"⚠️ 每个起点独立加载原始数据 + 独立 expanding 标准化")

    results = []
    for start in START_DATES:
        metrics = run_single_start(start, V710_CFG)
        results.append(metrics)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)

    calmars = [m["calmar"] for m in results if "error" not in m]
    sharpes = [m["sharpe"] for m in results if "error" not in m]
    dds = [m["max_dd"] for m in results if "error" not in m]

    if len(calmars) < 2:
        print("⚠️ 有效起点不足 2 个, 无法计算 CV%")
        return

    calmar_mean = np.mean(calmars)
    calmar_std = np.std(calmars)
    calmar_cv = calmar_std / abs(calmar_mean) if abs(calmar_mean) > 1e-9 else float('inf')

    sharpe_mean = np.mean(sharpes)
    sharpe_std = np.std(sharpes)
    sharpe_cv = sharpe_std / abs(sharpe_mean) if abs(sharpe_mean) > 1e-9 else float('inf')

    print(f"\nOOS Calmar 跨起点:")
    for m in results:
        if "error" not in m:
            print(f"  {m['name']}: Calmar={m['calmar']:.3f}, Sharpe={m['sharpe']:.3f}, DD={m['max_dd']*100:.2f}%")

    print(f"\nCalmar 统计:")
    print(f"  均值:  {calmar_mean:.3f}")
    print(f"  标准差: {calmar_std:.3f}")
    print(f"  CV%:   {calmar_cv*100:.1f}%")

    print(f"\nSharpe 统计:")
    print(f"  均值:  {sharpe_mean:.3f}")
    print(f"  标准差: {sharpe_std:.3f}")
    print(f"  CV%:   {sharpe_cv*100:.1f}%")

    # 判定
    print(f"\n{'='*60}")
    print("判定")
    print(f"{'='*60}")
    if calmar_mean < 0:
        verdict = "❌ DEPRECATED"
        detail = "所有起点 Calmar 为负, 策略全面亏损"
    elif calmar_cv < 0.25:
        verdict = "✅ PASS"
        detail = "v7.10 维持 RECOMMENDED"
    elif calmar_cv < 0.50:
        verdict = "⚠️ PROMISING"
        detail = "需调参, 可能降级"
    else:
        verdict = "❌ DEPRECATED"
        detail = "起点依赖严重, 复刻 v6.2 命运"

    print(f"  Calmar CV%: {calmar_cv*100:.1f}% → {verdict}")
    print(f"  {detail}")

    # 生成报告
    report_path = ROOT / "reports" / "momentum_etf_rotation" / "v7_10_cv_test.md"
    with open(report_path, "w") as f:
        f.write("# v7.10 起点依赖 CV% 测试报告 (Stage 32)\n\n")
        f.write(f"> **日期**: 2026-07-28 (修复 expanding 标准化)\n")
        f.write(f"> **参数**: λ_tv={V710_CFG.lambda_tv}, λ_l1={V710_CFG.lambda_l1}\n")
        f.write(f"> **判定**: {verdict} (Calmar CV% = {calmar_cv*100:.1f}%)\n")
        f.write(f"> **修复**: 每个起点独立加载原始数据 + 独立 expanding 标准化\n\n")

        f.write("## 1. 测试设置\n\n")
        f.write("- 3 个起点: 2018-01 / 2020-01 / 2022-01\n")
        f.write("- 固定 v7.10 最优参数 (Stage 31)\n")
        f.write("- CV% = std(OOS_Calmar) / mean(OOS_Calmar)\n")
        f.write("- ⚠️ 每个起点独立标准化 (不再截断全量标准化后的数据)\n\n")

        f.write("## 2. 结果\n\n")
        f.write("| 起点 | 周数 | 年化收益 | Sharpe | 最大回撤 | Calmar |\n")
        f.write("|------|------|----------|--------|----------|--------|\n")
        for m in results:
            if "error" not in m:
                f.write(f"| {m['name']} | {m['n_weeks']} | {m['ann_return']*100:+.2f}% | {m['sharpe']:.3f} | {m['max_dd']*100:.2f}% | {m['calmar']:.3f} |\n")

        f.write(f"\n## 3. 统计\n\n")
        f.write(f"| 指标 | 均值 | 标准差 | CV% |\n")
        f.write(f"|------|------|--------|------|\n")
        f.write(f"| Calmar | {calmar_mean:.3f} | {calmar_std:.3f} | **{calmar_cv*100:.1f}%** |\n")
        f.write(f"| Sharpe | {sharpe_mean:.3f} | {sharpe_std:.3f} | {sharpe_cv*100:.1f}% |\n")

        f.write(f"\n## 4. 判定\n\n")
        f.write(f"**{verdict}** — Calmar CV% = {calmar_cv*100:.1f}%\n\n")
        if calmar_cv < 0.25:
            f.write("v7.10 起点稳定性良好, 维持 RECOMMENDED 状态.\n")
        elif calmar_cv < 0.50:
            f.write("v7.10 存在一定起点依赖, 需进一步调参.\n")
        else:
            f.write("v7.10 起点依赖严重, 建议降级为 DEPRECATED.\n")

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()

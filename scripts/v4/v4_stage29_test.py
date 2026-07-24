# coding=utf-8
"""scripts/v4/v4_stage29_test.py — Stage 29 增强测试 (相关性约束 + 多窗口动量).

对比:
  Stage 28: 4 因子加权 (单窗口动量)
  Stage 29: 4 因子加权 + 多窗口动量
  Stage 29+: 4 因子加权 + 多窗口动量 + 相关性约束

数据: 43 ETF (2018-2026, 8.3 年)
top_k: 4 (黄金点)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import (
    select_smart_beta_proxy,
    _composite_score,
    SECTOR_CODES,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics


def load_data():
    etf = pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_10_Y_weekly.parquet")
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]
    etf_price = (1 + etf_clean).cumprod() * 100
    return etf_clean, etf_price.pct_change().fillna(0)


def evaluate(
    returns: pd.DataFrame,
    weights: dict[str, float],
    lookback: int = 60,
    top_k: int = 4,
    codes: tuple[str, ...] = SECTOR_CODES,
    rebal_freq: int = 4,
    momentum_windows: tuple[int, ...] | None = None,
    momentum_window_weights: tuple[float, ...] | None = None,
    corr_constraint: bool = False,
    corr_threshold: float = 0.7,
    corr_window: int = 60,
) -> dict:
    valid = [c for c in codes if c in returns.columns]
    nav = pd.Series(1.0, index=returns.index)
    selected_history = []

    for i in range(lookback, len(returns), rebal_freq):
        sub = returns[valid].iloc[i - lookback:i]
        if sub.empty:
            continue

        try:
            selected = select_smart_beta_proxy(
                returns.iloc[:i],
                lookback=lookback,
                top_k=top_k,
                codes=codes,
                weights=weights,
                zscore_norm=True,
                winsorize_sigma=3.0,
                momentum_windows=momentum_windows,
                momentum_window_weights=momentum_window_weights,
                corr_constraint=corr_constraint,
                corr_threshold=corr_threshold,
                corr_window=corr_window,
            )
        except Exception:
            continue

        selected_history.append(selected)

        next_i = min(i + rebal_freq, len(returns))
        for j in range(i, next_i):
            if j == 0 or not selected:
                continue
            v = [c for c in selected if c in returns.columns]
            if not v:
                continue
            nav.iloc[j] = nav.iloc[j - 1] * (1 + returns[v].iloc[j].mean())

    nav_ret = nav.pct_change().fillna(0)
    metrics = compute_metrics(nav_ret, freq="W")

    stability = 0
    if len(selected_history) > 1:
        stability = sum(
            1 for s in selected_history[1:]
            if set(s) == set(selected_history[0])
        ) / (len(selected_history) - 1)

    metrics["stability"] = stability
    metrics["total_rebalances"] = len(selected_history)
    return metrics


def main():
    print("=" * 80)
    print("Stage 29 增强测试 (相关性约束 + 多窗口动量)")
    print("=" * 80)

    _, returns = load_data()
    print(f"\n数据: {returns.shape[0]} 周, {returns.shape[1]} ETF")
    print(f"时间: {returns.index.min()} ~ {returns.index.max()}")

    # 最优权重
    BEST_W = {"value": 0.20, "quality": 0.30, "low_vol": 0.20, "momentum": 0.30}

    configs = [
        # Stage 28: 4 因子加权 (单窗口动量)
        {
            "name": "Stage 28: 单窗口动量",
            "momentum_windows": None,
            "momentum_window_weights": None,
            "corr_constraint": False,
        },
        # Stage 29: 多窗口动量
        {
            "name": "Stage 29a: 多窗口 (5, 20, 60)",
            "momentum_windows": (5, 20, 60),
            "momentum_window_weights": (0.3, 0.4, 0.3),
            "corr_constraint": False,
        },
        {
            "name": "Stage 29b: 多窗口 (20, 60, 120)",
            "momentum_windows": (20, 60, 120),
            "momentum_window_weights": (0.3, 0.4, 0.3),
            "corr_constraint": False,
        },
        {
            "name": "Stage 29c: 多窗口 (5, 20, 60, 120)",
            "momentum_windows": (5, 20, 60, 120),
            "momentum_window_weights": (0.2, 0.3, 0.3, 0.2),
            "corr_constraint": False,
        },
        # Stage 29+: 多窗口 + 相关性约束
        {
            "name": "Stage 29+: 多窗口+相关性0.7",
            "momentum_windows": (5, 20, 60),
            "momentum_window_weights": (0.3, 0.4, 0.3),
            "corr_constraint": True,
            "corr_threshold": 0.7,
        },
        {
            "name": "Stage 29++: 多窗口+相关性0.5",
            "momentum_windows": (5, 20, 60),
            "momentum_window_weights": (0.3, 0.4, 0.3),
            "corr_constraint": True,
            "corr_threshold": 0.5,
        },
    ]

    print(f"\n{'=' * 80}")
    print(f"性能对比 (top_k=4, 4 因子最优权重)")
    print("=" * 80)

    results = []
    for cfg in configs:
        print(f"\n跑 {cfg['name']}...")
        m = evaluate(
            returns, BEST_W, lookback=60, top_k=4,
            codes=SECTOR_CODES,
            momentum_windows=cfg["momentum_windows"],
            momentum_window_weights=cfg["momentum_window_weights"],
            corr_constraint=cfg["corr_constraint"],
            corr_threshold=cfg.get("corr_threshold", 0.7),
        )
        results.append({"配置": cfg["name"], **m})

    df = pd.DataFrame(results)
    df = df.sort_values("Sharpe", ascending=False)

    display_cols = [
        "配置", "Sharpe", "Calmar", "AnnRet", "MaxDD",
        "WinRate", "TotalReturn", "stability", "total_rebalances",
    ]
    available = [c for c in display_cols if c in df.columns]
    print()
    print(df[available].to_string(index=False))

    # 保存
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "v4_stage29_comparison.csv", index=False)

    # 生成报告
    best = df.iloc[0]
    report_lines = [
        "# Stage 29 增强测试 (相关性约束 + 多窗口动量)",
        "",
        f"> 数据: {returns.index.min()} ~ {returns.index.max()} ({returns.shape[0]} 周)",
        f"> ETF: {returns.shape[1]} 个",
        f"> top_k: 4 (黄金点)",
        "",
        "## 测试配置",
        "",
        "| 序号 | 配置 | 多窗口动量 | 相关性约束 |",
        "|------|------|------------|------------|",
    ]

    for i, cfg in enumerate(configs, 1):
        mw = cfg["momentum_windows"] or "单窗口"
        cc = f"✓ 阈值={cfg.get('corr_threshold', 0.7)}" if cfg["corr_constraint"] else "✗"
        report_lines.append(f"| {i} | {cfg['name']} | {mw} | {cc} |")

    report_lines.extend([
        "",
        "## 业绩对比",
        "",
        "| 排序 | 配置 | Sharpe | Calmar | 年化 | MaxDD | 胜率 | 总收益 | 稳定性 | 调仓次数 |",
        "|------|------|--------|--------|------|-------|------|--------|--------|----------|",
    ])

    for i, (_, row) in enumerate(df.iterrows(), 1):
        report_lines.append(
            f"| {i} | {row['配置']} | {row['Sharpe']:.3f} | {row['Calmar']:.3f} | "
            f"{row['AnnRet']:.2%} | {row['MaxDD']:.2%} | {row['WinRate']:.2%} | "
            f"{row['TotalReturn']:.2%} | {row.get('stability', 0):.2%} | "
            f"{row.get('total_rebalances', 0)} |"
        )

    report_lines.extend([
        "",
        "## 关键发现",
        "",
    ])

    stage28 = df[df["配置"].str.contains("Stage 28")].iloc[0]
    stage29a = df[df["配置"].str.contains("Stage 29a")].iloc[0]
    stage29_plus = df[df["配置"].str.contains("Stage 29\\+")].iloc[0]

    report_lines.extend([
        f"1. **多窗口动量**: Stage 28 (单窗口) Sharpe {stage28['Sharpe']:.3f} → "
        f"Stage 29a (5/20/60) Sharpe {stage29a['Sharpe']:.3f}",
        f"2. **加相关性约束**: Stage 29+ (阈值 0.7) Sharpe {stage29_plus['Sharpe']:.3f}",
        f"3. **最优配置**: {best['配置']}",
        f"   - Sharpe: {best['Sharpe']:.3f}",
        f"   - Calmar: {best['Calmar']:.3f}",
        f"   - 年化: {best['AnnRet']:.2%}",
        f"   - MaxDD: {best['MaxDD']:.2%}",
        f"   - 总收益: {best['TotalReturn']:.2%}",
        "",
        "## 推荐配置",
        "",
        "```python",
        "smart_beta_config = SmartBetaConfig(",
        "    top_n=4,                              # top_k=4 黄金点",
        "    proxy_value_weight=0.20,",
        "    proxy_quality_weight=0.30,",
        "    proxy_low_vol_weight=0.20,",
        "    proxy_momentum_weight=0.30,",
        "    # Stage 29: 多窗口动量",
        "    proxy_momentum_windows=(5, 20, 60),",
        "    proxy_momentum_window_weights=(0.3, 0.4, 0.3),",
        "    # Stage 29: 相关性约束",
        "    proxy_corr_constraint=True,",
        "    proxy_corr_threshold=0.7,",
        ")",
        "```",
        "",
        "## 文件清单",
        "",
        "- `v4_stage29_comparison.csv`: 完整数据",
        "- `v4_stage29_report.md`: 本报告",
    ])

    report_path = output_dir / "v4_stage29_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    print(f"\n{'=' * 80}")
    print("完成!")


if __name__ == "__main__":
    main()
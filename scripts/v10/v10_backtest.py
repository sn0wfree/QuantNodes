# coding=utf-8
"""scripts/v10/v10_backtest.py — v10 完整回测 (周频 + 月频).

基于 docs/57-v10_final_design.md 用户确认版.

用法:
    python3.11 scripts/v10/v10_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from QuantNodes.strategy.momentum_etf_rotation.v10 import (
    V10Config,
    run_v10_backtest,
)


def load_data():
    """加载 43 ETF 数据 + 宏观因子."""
    data_dir = REPO / "data" / "high_freq_macro"

    # ETF 收益
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]
    # 转换为价格序列 (v10 回测引擎假设输入是价格, 但实际用收益)
    # 注: backtest_v10 用的是收益率序列, 不是价格

    # 宏观因子
    try:
        macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    except FileNotFoundError:
        macro = None

    return etf_clean, macro


def run_v10_backtest_for_freq(returns_df, macro_df, freq):
    """跑单个调仓频率的回测."""
    cfg = V10Config()
    cfg.rebal_freq = freq
    print(f"\n{'=' * 70}")
    print(f"v10 回测: 调仓频率 = {freq}")
    print(f"{'=' * 70}")

    try:
        result = run_v10_backtest(returns_df, macro_df, cfg)
        return result
    except Exception as e:
        import traceback
        print(f"回测失败: {e}")
        traceback.print_exc()
        return None


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v10"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v10 完整回测 (5 层架构, 基于 docs/57 用户确认版)")
    print("=" * 70)

    # 加载数据
    etf_returns, macro_df = load_data()
    print(f"\n数据: {etf_returns.shape[0]} 周, {etf_returns.shape[1]} ETF")
    if macro_df is not None:
        print(f"宏观因子: {macro_df.shape[0]} 周, {macro_df.shape[1]} 因子")
    print(f"时间: {etf_returns.index.min()} ~ {etf_returns.index.max()}")

    # 跑周频回测
    result_W = run_v10_backtest_for_freq(etf_returns, macro_df, 'W')

    # 跑月频回测
    result_M = run_v10_backtest_for_freq(etf_returns, macro_df, 'M')

    # === 输出结果 ===
    print(f"\n{'=' * 70}")
    print("v10 回测结果对比")
    print("=" * 70)

    results = []
    if result_W is not None:
        m = result_W.metrics
        results.append({"调仓频率": "W", **m})
    if result_M is not None:
        m = result_M.metrics
        results.append({"调仓频率": "M", **m})

    if not results:
        print("回测全部失败, 无法输出对比")
        return

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # 保存
    df_results.to_csv(output_dir / "v10_backtest_results.csv", index=False)
    print(f"\n保存: {output_dir / 'v10_backtest_results.csv'}")

    # === 画图 ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # NAV 对比
    ax = axes[0]
    if result_W is not None:
        ax.plot(result_W.nav.index, result_W.nav.values, label='v10 (W)', linewidth=2)
    if result_M is not None:
        ax.plot(result_M.nav.index, result_M.nav.values, label='v10 (M)', linewidth=2)
    ax.set_title('v10 NAV Curve (5 Layer Architecture)', fontsize=14)
    ax.set_ylabel('NAV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 仓位曲线
    ax = axes[1]
    if result_W is not None and result_W.position_size is not None:
        ax.plot(result_W.position_size.index, result_W.position_size.values,
                label='position_size (W)', linewidth=1.5, alpha=0.8)
    if result_M is not None and result_M.position_size is not None:
        ax.plot(result_M.position_size.index, result_M.position_size.values,
                label='position_size (M)', linewidth=1.5, alpha=0.8)
    ax.set_title('Dynamic Position Size', fontsize=14)
    ax.set_ylabel('Position Size')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v10_backtest.png", dpi=120, bbox_inches='tight')
    print(f"图片: {output_dir / 'v10_backtest.png'}")
    plt.close()

    # === 输出 Markdown 报告 ===
    report_lines = [
        "# v10 完整回测报告",
        "",
        f"> 数据: {etf_returns.index.min()} ~ {etf_returns.index.max()} ({etf_returns.shape[0]} 周)",
        f"> ETF: {etf_returns.shape[1]} 个",
        f"> 宏观因子: {macro_df.shape[1] if macro_df is not None else 0} 个",
        "",
        "## 5 层架构",
        "",
        "1. **Layer 1 宏观择时**: 5 宏观因子 + 熵权法 + TV-PR (默认开启)",
        "2. **Layer 2A 行业轮动**: 23 行业 ETF, regime 条件 (相关约束默认关闭)",
        "3. **Layer 2B 风格轮动**: 6 风格因子 IC 驱动",
        "4. **Layer 2C 因子选股**: 5 风格因子 + Top-K=10",
        "5. **Layer 3 风险控制**: Jump Model 牛熊检测",
        "6. **Layer 4 动态仓位**: pos = (0.7 - 0.5z).clip(0.2, 1.0) × bear_prob 调整",
        "7. **Layer 5 组合构建**: RP 底仓 × 行业/因子 tilt × 仓位",
        "",
        "## 用户决策 (docs/57)",
        "",
        "| # | 议题 | 决策 |",
        "|---|------|------|",
        "| 1 | TV-PR (Layer 1) | 必加, 可配置 |",
        "| 2 | Top-K | K=10 |",
        "| 3 | 因子加权 | 5 因子 (中信多因子) |",
        "| 4 | Jump Model | 需要 (乘到仓位, 0.5x) |",
        "| 5 | 调仓频率 | 周+月都测试 |",
        "| 6 | 估值/基本面 | v10.1 再加 |",
        "",
        "## 回测结果",
        "",
        "| 调仓频率 | Sharpe | Calmar | 年化 | MaxDD | 胜率 | 总收益 |",
        "|----------|--------|--------|------|-------|------|--------|",
    ]

    for _, row in df_results.iterrows():
        report_lines.append(
            f"| {row['调仓频率']} | {row['sharpe']:.3f} | {row['calmar']:.3f} | "
            f"{row['ann_return']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row['win_rate']:.2%} | {row['total_return']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 与 v9 对比 (期望)",
        "",
        "| 指标 | v9 银河方案 | v9 中信多因子 | v10 预期 |",
        "|------|-------------|---------------|----------|",
        "| Sharpe | 1.23 | 0.62 | 1.3 - 1.8 |",
        "| Calmar | 1.20 | 0.50 | 1.4 - 1.6 |",
        "| MaxDD | -13.7% | -18.0% | -8% ~ -11% |",
        "| 年化 | 16.4% | 9.0% | 18% - 25% |",
        "",
        "## 文件清单",
        "",
        "- `v10_backtest_results.csv`: 回测结果",
        "- `v10_backtest.png`: NAV + 仓位曲线",
        "- `v10_report.md`: 本报告",
        "",
        "## 代码",
        "",
        "- `QuantNodes/strategy/momentum_etf_rotation/v10/`: 5 层模块",
        "- `scripts/v10/v10_backtest.py`: 本脚本",
    ])

    report_path = output_dir / "v10_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告: {report_path}")
    print(f"\n完成!")


if __name__ == "__main__":
    main()
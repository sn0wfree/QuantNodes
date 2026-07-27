# coding=utf-8
"""scripts/v11/v11_backtest.py — v11 完整回测 (真实数据 + ACT-1/2/3).

基于 scripts/v10/v10_backtest.py 修改, 添加 OHLCV 数据加载.

用法:
    python3 scripts/v11/v11_backtest.py
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

from QuantNodes.strategy.momentum_etf_rotation.v10.backtest_v10 import run_v10_backtest
from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Config, run_v11_backtest


def load_data():
    """加载真实数据: ETF 收益 + 宏观因子 + OHLCV."""
    data_dir = REPO / "data" / "high_freq_macro"
    real_dir = REPO / "data" / "real"

    # 1. ETF 收益 (周频)
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]

    # 2. 宏观因子 (周频)
    try:
        macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    except FileNotFoundError:
        macro = None

    # 3. OHLCV 数据 (日频 → 周频)
    ohlcv = pd.read_parquet(real_dir / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    
    # 只保留 ETF 收益中存在的代码
    common_codes = list(set(etf_clean.columns) & set(ohlcv.columns.get_level_values(0)))
    ohlcv_common = ohlcv[common_codes]
    
    # 日频 → 周频 (取每周最后一个交易日)
    ohlcv_weekly = ohlcv_common.resample('W').last()
    
    # 对齐时间
    common_dates = etf_clean.index.intersection(ohlcv_weekly.index)
    etf_clean = etf_clean.loc[common_dates]
    ohlcv_weekly = ohlcv_weekly.loc[common_dates]
    
    if macro is not None:
        macro = macro.loc[macro.index.intersection(common_dates)]

    return etf_clean, macro, ohlcv_weekly


def run_backtest_for_freq(returns_df, macro_df, ohlcv_df, freq, version='v11'):
    """跑单个调仓频率的回测."""
    print(f"\n{'=' * 70}")
    print(f"{version} 回测: 调仓频率 = {freq}")
    print(f"{'=' * 70}")

    try:
        if version == 'v10':
            cfg = V10Config()
            cfg.rebal_freq = freq
            result = run_v10_backtest(returns_df, macro_df, cfg)
        else:
            cfg = V11Config()
            cfg.rebal_freq = freq
            result = run_v11_backtest(returns_df, macro_df, ohlcv_df, cfg)
        return result
    except Exception as e:
        import traceback
        print(f"回测失败: {e}")
        traceback.print_exc()
        return None


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v11"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v11 完整回测 (v10 + ACT-1/2/3, 真实数据)")
    print("=" * 70)

    # 加载数据
    etf_returns, macro_df, ohlcv_df = load_data()
    print(f"\n数据: {etf_returns.shape[0]} 周, {etf_returns.shape[1]} ETF")
    if macro_df is not None:
        print(f"宏观因子: {macro_df.shape[0]} 周, {macro_df.shape[1]} 因子")
    print(f"OHLCV: {ohlcv_df.shape[0]} 周, {ohlcv_df.shape[1] // 5} 代码 × 5 字段")
    print(f"时间: {etf_returns.index.min()} ~ {etf_returns.index.max()}")

    # === v10 回测 (周频) ===
    result_v10_W = run_backtest_for_freq(etf_returns, macro_df, ohlcv_df, 'W', 'v10')

    # === v11 回测 (周频) ===
    result_v11_W = run_backtest_for_freq(etf_returns, macro_df, ohlcv_df, 'W', 'v11')

    # === v11 回测 (月频) ===
    result_v11_M = run_backtest_for_freq(etf_returns, macro_df, ohlcv_df, 'M', 'v11')

    # === 输出结果 ===
    print(f"\n{'=' * 70}")
    print("回测结果对比")
    print("=" * 70)

    results = []
    if result_v10_W is not None:
        m = result_v10_W.metrics
        results.append({"版本": "v10", "调仓频率": "W", **m})
    if result_v11_W is not None:
        m = result_v11_W.metrics
        results.append({"版本": "v11", "调仓频率": "W", **m})
    if result_v11_M is not None:
        m = result_v11_M.metrics
        results.append({"版本": "v11", "调仓频率": "M", **m})

    if not results:
        print("回测全部失败")
        return

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # 保存
    df_results.to_csv(output_dir / "v11_backtest_results.csv", index=False)
    print(f"\n保存: {output_dir / 'v11_backtest_results.csv'}")

    # === 画图 ===
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # NAV 对比
    ax = axes[0]
    if result_v10_W is not None:
        ax.plot(result_v10_W.nav.index, result_v10_W.nav.values, 
                label='v10 (W)', linewidth=2, alpha=0.8)
    if result_v11_W is not None:
        ax.plot(result_v11_W.nav.index, result_v11_W.nav.values, 
                label='v11 (W)', linewidth=2, alpha=0.8)
    if result_v11_M is not None:
        ax.plot(result_v11_M.nav.index, result_v11_M.nav.values, 
                label='v11 (M)', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_title('NAV Curve: v10 vs v11', fontsize=14)
    ax.set_ylabel('NAV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 仓位曲线
    ax = axes[1]
    if result_v10_W is not None and result_v10_W.position_size is not None:
        ax.plot(result_v10_W.position_size.index, result_v10_W.position_size.values,
                label='v10 position (W)', linewidth=1.5, alpha=0.8)
    if result_v11_W is not None and result_v11_W.position_size is not None:
        ax.plot(result_v11_W.position_size.index, result_v11_W.position_size.values,
                label='v11 position (W)', linewidth=1.5, alpha=0.8)
    ax.set_title('Dynamic Position Size', fontsize=14)
    ax.set_ylabel('Position Size')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DD 控制器乘数 (v11 only)
    ax = axes[2]
    if result_v11_W is not None and result_v11_W.dd_multipliers is not None:
        ax.plot(result_v11_W.dd_multipliers.index, result_v11_W.dd_multipliers.values,
                label='v11 DD multiplier (W)', linewidth=1.5, alpha=0.8, color='red')
    ax.set_title('Drawdown Controller Multiplier (v11)', fontsize=14)
    ax.set_ylabel('Multiplier')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v11_backtest.png", dpi=120, bbox_inches='tight')
    print(f"图片: {output_dir / 'v11_backtest.png'}")
    plt.close()

    # === Kelly 审计报告 ===
    if result_v11_W is not None and result_v11_W.kelly_result:
        print(f"\n{'=' * 70}")
        print("v11 Kelly 审计 (ACT-2)")
        print("=" * 70)
        for k, v in result_v11_W.kelly_result.items():
            print(f"  {k}: {v}")

    # === 输出 Markdown 报告 ===
    report_lines = [
        "# v11 完整回测报告",
        "",
        f"> 数据: {etf_returns.index.min()} ~ {etf_returns.index.max()} ({etf_returns.shape[0]} 周)",
        f"> ETF: {etf_returns.shape[1]} 个",
        f"> 宏观因子: {macro_df.shape[1] if macro_df is not None else 0} 个",
        f"> OHLCV: {ohlcv_df.shape[1] // 5} 个代码",
        "",
        "## v11 升级点 (基于 10_TURTLE_TRADING_MATHEMATICS.md)",
        "",
        "1. **ACT-1: Yang-Zhang 波动率** — 消除漂移污染, 5x 响应速度",
        "2. **ACT-2: Kelly 审计** — 自动输出 sizing 位置",
        "3. **ACT-3: 回撤控制器** — Grossman-Zhou (1993) 连续控制",
        "",
        "## 回测结果",
        "",
        "| 版本 | 调仓频率 | Sharpe | Calmar | 年化 | MaxDD | 胜率 | 总收益 |",
        "|------|----------|--------|--------|------|-------|------|--------|",
    ]

    for _, row in df_results.iterrows():
        report_lines.append(
            f"| {row['版本']} | {row['调仓频率']} | {row['sharpe']:.3f} | {row['calmar']:.3f} | "
            f"{row['ann_return']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row['win_rate']:.2%} | {row['total_return']:.2%} |"
        )

    if result_v11_W is not None and result_v11_W.kelly_result:
        kr = result_v11_W.kelly_result
        report_lines.extend([
            "",
            "## Kelly 审计 (v11)",
            "",
            f"- Sharpe: {kr.get('sharpe', 0):.3f}",
            f"- CAGR: {kr.get('cagr', 0):.2%}",
            f"- Kelly Fraction: {kr.get('kelly_fraction', 0):.1%}",
            f"- Status: {kr.get('status', 'UNKNOWN')}",
        ])

    report_lines.extend([
        "",
        "## 文件清单",
        "",
        "- `v11_backtest_results.csv`: 回测结果",
        "- `v11_backtest.png`: NAV + 仓位 + DD 控制器曲线",
        "- `v11_report.md`: 本报告",
    ])

    report_path = output_dir / "v11_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告: {report_path}")
    print(f"\n完成!")


if __name__ == "__main__":
    main()

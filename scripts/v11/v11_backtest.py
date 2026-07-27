# coding=utf-8
"""scripts/v11/v11_backtest.py — v11 完整回测 (真实数据 + ACT-1/2/3).

吸收自:
  - scripts/v10/v10_backtest.py (5 层 + 周/月调仓对比) [已迁移, 删除]
  - scripts/v10/v10_compare.py (v11 vs v9 多策略对比) [已迁移, 删除]
  - scripts/v10/v10_tvpr_sensitivity.py (TV-PR 权重敏感性) [后续 step]

基于 scripts/v10/v10_backtest.py 修改, 添加 OHLCV 数据加载 + ACT-1/2/3.

用法:
    python3 scripts/v11/v11_backtest.py                     # 默认: W 调仓完整回测
    python3 scripts/v11/v11_backtest.py --mode compare      # v10 W vs v11 W vs v11 M
    python3 scripts/v11/v11_backtest.py --freq D            # 单次回测 (日频)
"""
from __future__ import annotations

import argparse
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
        if version == 'v10_5layer':
            print("v10 5 层架构已迁移到 v11, 请用 --version v11 跑 v10 配置.")
            return None
        cfg = V11Config()
        cfg.rebal_freq = freq
        result = run_v11_backtest(returns_df, macro_df, ohlcv_df, cfg)
        return result
    except Exception as e:
        import traceback
        print(f"回测失败: {e}")
        traceback.print_exc()
        return None


def main(mode='compare'):
    """v11 回测主入口.

    Args:
        mode: 'compare' = v11 W vs v11 M 对比 (默认)
              'single' = 单调仓频率回测
    """
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v11"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v11 完整回测 (5 层架构 + ACT-1/2/3, 真实数据)")
    print("=" * 70)

    # 加载数据
    etf_returns, macro_df, ohlcv_df = load_data()
    print(f"\n数据: {etf_returns.shape[0]} 周, {etf_returns.shape[1]} ETF")
    if macro_df is not None:
        print(f"宏观因子: {macro_df.shape[0]} 周, {macro_df.shape[1]} 因子")
    print(f"OHLCV: {ohlcv_df.shape[0]} 周, {ohlcv_df.shape[1] // 5} 代码 × 5 字段")
    print(f"时间: {etf_returns.index.min()} ~ {etf_returns.index.max()}")

    if mode == 'single':
        return run_single_mode(etf_returns, macro_df, ohlcv_df, output_dir)
    elif mode == 'compare':
        return run_compare_mode(etf_returns, macro_df, ohlcv_df, output_dir)
    else:
        print(f"未知 mode: {mode}")
        return


def run_single_mode(etf_returns, macro_df, ohlcv_df, output_dir):
    """单次回测 (默认周频)."""
    result_v11_W = run_backtest_for_freq(etf_returns, macro_df, ohlcv_df, 'W', 'v11')
    if result_v11_W is None:
        print("回测失败")
        return

    results = [{"版本": "v11", "调仓频率": "W", **result_v11_W.metrics}]
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    df_results.to_csv(output_dir / "v11_backtest_results.csv", index=False)
    print(f"\n保存: {output_dir / 'v11_backtest_results.csv'}")
    _plot_single(result_v11_W, output_dir)


def run_compare_mode(etf_returns, macro_df, ohlcv_df, output_dir):
    """对比: v11 W vs v11 M (吸收自 v10_backtest.py)."""
    # === v11 回测 (周频) ===
    result_v11_W = run_backtest_for_freq(etf_returns, macro_df, ohlcv_df, 'W', 'v11')

    # === v11 回测 (月频) ===
    result_v11_M = run_backtest_for_freq(etf_returns, macro_df, ohlcv_df, 'M', 'v11')

    # === 输出结果 ===
    print(f"\n{'=' * 70}")
    print("回测结果对比")
    print("=" * 70)

    results = []
    if result_v11_W is not None:
        results.append({"版本": "v11", "调仓频率": "W", **result_v11_W.metrics})
    if result_v11_M is not None:
        results.append({"版本": "v11", "调仓频率": "M", **result_v11_M.metrics})

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
    if result_v11_W is not None:
        ax.plot(result_v11_W.nav.index, result_v11_W.nav.values,
                label='v11 (W)', linewidth=2, alpha=0.8)
    if result_v11_M is not None:
        ax.plot(result_v11_M.nav.index, result_v11_M.nav.values,
                label='v11 (M)', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_title('v11 NAV Curve: W vs M', fontsize=14)
    ax.set_ylabel('NAV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 仓位曲线
    ax = axes[1]
    if result_v11_W is not None and result_v11_W.position_size is not None:
        ax.plot(result_v11_W.position_size.index, result_v11_W.position_size.values,
                label='v11 position (W)', linewidth=1.5, alpha=0.8)
    if result_v11_M is not None and result_v11_M.position_size is not None:
        ax.plot(result_v11_M.position_size.index, result_v11_M.position_size.values,
                label='v11 position (M)', linewidth=1.5, alpha=0.8, linestyle='--')
    ax.set_title('Dynamic Position Size', fontsize=14)
    ax.set_ylabel('Position Size')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DD 控制器乘数 (v11 only, if wired in)
    ax = axes[2]
    dd_m = getattr(result_v11_W, 'dd_multipliers', None) if result_v11_W is not None else None
    if dd_m is not None:
        ax.plot(dd_m.index, dd_m.values, label='v11 DD multiplier (W)',
                linewidth=1.5, alpha=0.8, color='red')
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
    kelly = getattr(result_v11_W, 'kelly_result', None) if result_v11_W is not None else None
    if kelly:
        print(f"\n{'=' * 70}")
        print("v11 Kelly 审计 (ACT-2)")
        print("=" * 70)
        for k, v in kelly.items():
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

    if kelly:
        report_lines.extend([
            "",
            "## Kelly 审计 (v11)",
            "",
            f"- Sharpe: {kelly.get('sharpe', 0):.3f}",
            f"- CAGR: {kelly.get('cagr', 0):.2%}",
            f"- Kelly Fraction: {kelly.get('kelly_fraction', 0):.1%}",
            f"- Status: {kelly.get('status', 'UNKNOWN')}",
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


def _plot_single(result, output_dir):
    """单调仓频率模式的画图."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    if result.nav is not None:
        ax.plot(result.nav.index, result.nav.values, linewidth=2, alpha=0.8)
        ax.set_title(f'v11 NAV ({result.metrics.get("sharpe", 0):.3f} Sharpe)', fontsize=14)
        ax.set_ylabel('NAV')
        ax.grid(True, alpha=0.3)

    ax = axes[1]
    if result.position_size is not None:
        ax.plot(result.position_size.index, result.position_size.values,
                linewidth=1.5, alpha=0.8)
        ax.set_title('Dynamic Position Size', fontsize=14)
        ax.set_ylabel('Position Size')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v11_backtest.png", dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='v11 backtest (5-layer + ACT-1/2/3)')
    parser.add_argument('--mode', choices=['compare', 'single'], default='compare',
                        help='compare: W vs M; single: single freq (default: compare)')
    parser.add_argument('--freq', default='W', help='rebalance freq for single mode (default: W)')
    args = parser.parse_args()
    main(mode=args.mode)

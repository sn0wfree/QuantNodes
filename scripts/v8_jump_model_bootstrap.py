#!/usr/bin/env python3
# coding=utf-8
"""v8 Jump Model Bootstrap 实验 — 测试 n_restarts 稳定性.

目标: 找到最优的随机初始化次数，确保结果可复现

用法:
    python3.11 scripts/v8_jump_model_bootstrap.py

输出:
    reports/momentum_etf_rotation/v8_bootstrap_results.csv
    reports/momentum_etf_rotation/v8_bootstrap_cv.png
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 1000


# ============================================================
# 数据加载
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    """加载日频 ETF 收益."""
    path = REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet"
    return pd.read_parquet(path)


# ============================================================
# Bootstrap 实验
# ============================================================
def run_bootstrap_experiment(
    returns: pd.Series,
    n_restarts_list: list[int],
    n_experiments: int = 10,
    random_seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Bootstrap 实验: 测试不同 n_restarts 的稳定性.

    Parameters:
        returns: 日频收益序列
        n_restarts_list: 要测试的 n_restarts 值列表
        n_experiments: 每个 n_restarts 跑多少次实验
        random_seeds: 随机种子列表 (如果为 None，自动生成)

    Returns:
        DataFrame, columns=['n_restarts', 'seed', 'bear_ratio', 'oos_sharpe', 'oos_maxdd']
    """
    from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import jump_model_periodic_retrain

    if random_seeds is None:
        random_seeds = list(range(n_experiments))

    results = []

    for n_restarts in n_restarts_list:
        logging.info(f"\n  测试 n_restarts={n_restarts}...")
        for seed in random_seeds:
            t0 = time.time()
            states = jump_model_periodic_retrain(
                returns,
                asset_type='equity',
                train_window=TRAIN_WINDOW,
                retrain_every=60,
                n_restarts=n_restarts,
                show_progress=False,
                random_state=seed,
            )
            elapsed = time.time() - t0

            # OOS 评估
            oos_states = states.iloc[TRAIN_WINDOW:]
            returns_oos = returns.iloc[TRAIN_WINDOW:]

            # 对齐 index
            common_idx = returns_oos.index.intersection(oos_states.index)
            returns_oos = returns_oos.loc[common_idx]
            oos_states = oos_states.loc[common_idx]

            bear_ratio = (oos_states == 1).mean() * 100

            # 计算 Sharpe
            signal = oos_states * 0.5
            adjusted_returns = returns_oos * (1 - signal)
            nav = (1 + adjusted_returns).cumprod()
            n_years = len(nav) / 252
            total_ret = nav.iloc[-1] / nav.iloc[0] - 1
            ann_ret = (1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1
            vol = adjusted_returns.std() * np.sqrt(252)
            sharpe = ann_ret / vol if vol > 0 else 0

            # 计算 MaxDD
            cummax = nav.cummax()
            dd = (nav / cummax - 1)
            max_dd = float(dd.min())

            results.append({
                'n_restarts': n_restarts,
                'seed': seed,
                'bear_ratio': bear_ratio,
                'oos_sharpe': sharpe,
                'oos_ann_ret': ann_ret,
                'oos_max_dd': max_dd,
                'time': elapsed,
            })

    return pd.DataFrame(results)


# ============================================================
# 可视化
# ============================================================
def plot_cv_vs_restarts(df: pd.DataFrame, output_path: Path):
    """可视化 CV 随 n_restarts 的变化."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # 计算每个 n_restarts 的统计量
        stats = df.groupby('n_restarts').agg({
            'bear_ratio': ['mean', 'std'],
            'oos_sharpe': ['mean', 'std'],
            'oos_ann_ret': ['mean', 'std'],
            'time': 'mean',
        })

        # 计算 CV (变异系数)
        stats['bear_ratio_cv'] = stats[('bear_ratio', 'std')] / stats[('bear_ratio', 'mean')]
        stats['sharpe_cv'] = stats[('oos_sharpe', 'std')] / stats[('oos_sharpe', 'mean')].clip(lower=0.01)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Bear% CV vs n_restarts
        ax = axes[0, 0]
        ax.plot(stats.index, stats['bear_ratio_cv'], 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=0.05, color='r', linestyle='--', label='CV=5% 阈值')
        ax.set_xlabel('n_restarts')
        ax.set_ylabel('Bear% CV (变异系数)')
        ax.set_title('Bear% 变异系数 vs n_restarts')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Sharpe CV vs n_restarts
        ax = axes[0, 1]
        ax.plot(stats.index, stats['sharpe_cv'], 'go-', linewidth=2, markersize=8)
        ax.axhline(y=0.10, color='r', linestyle='--', label='CV=10% 阈值')
        ax.set_xlabel('n_restarts')
        ax.set_ylabel('Sharpe CV (变异系数)')
        ax.set_title('Sharpe 变异系数 vs n_restarts')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Mean Bear% vs n_restarts
        ax = axes[1, 0]
        ax.errorbar(stats.index, stats[('bear_ratio', 'mean')],
                    yerr=stats[('bear_ratio', 'std')], fmt='bo-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('n_restarts')
        ax.set_ylabel('Bear% (均值 ± 标准差)')
        ax.set_title('Bear% 均值 vs n_restarts')
        ax.grid(True, alpha=0.3)

        # 4. 计算时间 vs n_restarts
        ax = axes[1, 1]
        ax.plot(stats.index, stats[('time', 'mean')], 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('n_restarts')
        ax.set_ylabel('计算时间 (秒)')
        ax.set_title('计算时间 vs n_restarts')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"  图表已保存: {output_path}")

    except ImportError:
        logging.warning("  matplotlib 未安装，跳过可视化")


# ============================================================
# 主测试
# ============================================================
def main():
    logging.info("=" * 60)
    logging.info("v8 Jump Model Bootstrap 实验")
    logging.info("=" * 60)

    # 加载数据
    logging.info("加载数据...")
    daily_returns = load_daily_returns()
    logging.info(f"  日频收益: {daily_returns.shape}")

    # 选择代表性资产测试
    test_assets = {
        '510300': '沪深300',
        '511260': '国债',
    }

    # n_restarts 列表
    n_restarts_list = [1, 2, 3, 5, 10, 20]

    # 随机种子
    random_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    all_results = []

    for code, name in test_assets.items():
        if code not in daily_returns.columns:
            continue

        returns = daily_returns[code].dropna()
        if len(returns) < TRAIN_WINDOW + 100:
            logging.warning(f"  {name} ({code}): 数据不足 ({len(returns)} 天)")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"资产: {name} ({code})")
        logging.info(f"{'='*60}")

        # 运行 Bootstrap 实验
        results = run_bootstrap_experiment(
            returns,
            n_restarts_list=n_restarts_list,
            n_experiments=len(random_seeds),
            random_seeds=random_seeds,
        )
        results['asset'] = name
        results['code'] = code
        all_results.append(results)

        # 统计分析
        stats = results.groupby('n_restarts').agg({
            'bear_ratio': ['mean', 'std'],
            'oos_sharpe': ['mean', 'std'],
            'time': 'mean',
        })

        # 计算 CV
        stats['bear_ratio_cv'] = stats[('bear_ratio', 'std')] / stats[('bear_ratio', 'mean')]

        logging.info(f"\n  n_restarts | Bear% CV | Sharpe Mean | 时间(s)")
        logging.info(f"  -----------|----------|-------------|--------")
        for n_restarts in n_restarts_list:
            row = stats.loc[n_restarts]
            cv = float(row['bear_ratio_cv'])
            sharpe_mean = float(row[('oos_sharpe', 'mean')])
            time_mean = float(row[('time', 'mean')])
            logging.info(f"  {n_restarts:11d} | {cv:8.4f} | {sharpe_mean:11.3f} | {time_mean:6.1f}")

    # 保存结果
    logging.info("\n" + "=" * 60)
    logging.info("保存结果")
    logging.info("=" * 60)

    df_all = pd.concat(all_results, ignore_index=True)
    out_csv = OUTPUT_DIR / "v8_bootstrap_results.csv"
    df_all.to_csv(out_csv, index=False)
    logging.info(f"  CSV: {out_csv}")

    # 可视化
    plot_cv_vs_restarts(df_all, OUTPUT_DIR / "v8_bootstrap_cv.png")

    # 生成 Markdown 报告
    lines = [
        "# v8 Jump Model Bootstrap 实验报告",
        "",
        f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**训练窗口**: {TRAIN_WINDOW} 天",
        f"**n_restarts 列表**: {n_restarts_list}",
        f"**随机种子**: {random_seeds}",
        "",
        "## 1. 实验目的",
        "",
        "测试不同 n_restarts 值对结果稳定性的影响，找到最优的随机初始化次数。",
        "",
        "## 2. 实验结果",
        "",
    ]

    for code, name in test_assets.items():
        if code not in daily_returns.columns:
            continue

        df_asset = df_all[df_all['code'] == code]
        stats = df_asset.groupby('n_restarts').agg({
            'bear_ratio': ['mean', 'std'],
            'oos_sharpe': ['mean', 'std'],
            'time': 'mean',
        })
        stats['bear_ratio_cv'] = stats[('bear_ratio', 'std')] / stats[('bear_ratio', 'mean')]

        lines.extend([
            f"### {name} ({code})",
            "",
            "| n_restarts | Bear% Mean | Bear% Std | Bear% CV | Sharpe Mean | 时间(s) |",
            "|------------|------------|-----------|----------|-------------|---------|",
        ])

        for n_restarts in n_restarts_list:
            row = stats.loc[n_restarts]
            bear_mean = float(row[('bear_ratio', 'mean')])
            bear_std = float(row[('bear_ratio', 'std')])
            bear_cv = float(row['bear_ratio_cv'])
            sharpe_mean = float(row[('oos_sharpe', 'mean')])
            time_mean = float(row[('time', 'mean')])
            lines.append(
                f"| {n_restarts} | {bear_mean:.1f}% | "
                f"{bear_std:.1f}% | {bear_cv:.4f} | "
                f"{sharpe_mean:.3f} | {time_mean:.1f} |"
            )

        lines.append("")

    lines.extend([
        "## 3. 结论",
        "",
        "(待填写)",
    ])

    out_md = OUTPUT_DIR / "v8_bootstrap_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"  MD: {out_md}")

    logging.info("\n" + "=" * 60)
    logging.info("实验完成!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

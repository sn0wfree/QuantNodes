#!/usr/bin/env python3
# coding=utf-8
"""v8 Jump Model 测试脚本 — 最优参数版本.

基于 bootstrap 实验和参数调优的最优参数:
  - 权益类: jump_penalty=50, 简单滚动窗口, n_restarts=10
  - 债券类: jump_penalty=50, 指数衰减窗口, n_restarts=10
  - 商品类: jump_penalty=50, 简单滚动窗口, n_restarts=10

用法:
    python3.11 scripts/v8_jump_model_test.py

输出:
    reports/momentum_etf_rotation/v8_jump_model_test.csv
    reports/momentum_etf_rotation/v8_jump_model_test.md
    reports/momentum_etf_rotation/v8_jump_model_results.png
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

# 使用默认训练窗口 (会被 jump_model_periodic_retrain 根据资产类型自动选择)
DEFAULT_TRAIN_WINDOW = 1000


# ============================================================
# 数据加载
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    """加载日频 ETF 收益."""
    path = REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet"
    return pd.read_parquet(path)


# ============================================================
# 评估指标
# ============================================================
def compute_metrics(nav: pd.Series, freq: int = 252) -> dict:
    """计算业绩指标."""
    if nav.empty or len(nav) < 2:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    n_years = len(rets) / freq
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0
    return {
        "calmar": round(calmar, 4),
        "ann_return": round(ann_ret, 4),
        "vol": round(vol, 4),
        "max_dd": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
    }


# ============================================================
# 回测
# ============================================================
def backtest_with_signal(
    market_returns: pd.Series,
    signal: pd.Series,
) -> pd.Series:
    """用信号回测."""
    common_idx = market_returns.index.intersection(signal.index)
    market_returns = market_returns.loc[common_idx]
    signal = signal.loc[common_idx]
    adjusted_returns = market_returns * (1 - signal)
    nav = (1 + adjusted_returns).cumprod()
    return nav


# ============================================================
# 可视化
# ============================================================
def plot_results(results: list[dict], output_path: Path):
    """可视化测试结果."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 单资产 Sharpe 对比
        ax = axes[0, 0]
        single_results = [r for r in results if r['type'] == 'single']
        if single_results:
            assets = [r['asset'] for r in single_results]
            sharpe_buyhold = [r['sharpe_buyhold'] for r in single_results]
            sharpe_jump = [r['sharpe_periodic'] for r in single_results]

            x = np.arange(len(assets))
            width = 0.35
            ax.bar(x - width/2, sharpe_buyhold, width, label='纯多头', alpha=0.8)
            ax.bar(x + width/2, sharpe_jump, width, label='Jump Model', alpha=0.8)
            ax.set_xlabel('资产')
            ax.set_ylabel('Sharpe')
            ax.set_title('单资产 Sharpe 对比')
            ax.set_xticks(x)
            ax.set_xticklabels(assets)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. 单资产 MaxDD 对比
        ax = axes[0, 1]
        if single_results:
            maxdd_buyhold = [r['maxdd_buyhold'] * 100 for r in single_results]
            maxdd_jump = [r['maxdd_periodic'] * 100 for r in single_results]

            ax.bar(x - width/2, maxdd_buyhold, width, label='纯多头', alpha=0.8)
            ax.bar(x + width/2, maxdd_jump, width, label='Jump Model', alpha=0.8)
            ax.set_xlabel('资产')
            ax.set_ylabel('MaxDD (%)')
            ax.set_title('单资产 MaxDD 对比')
            ax.set_xticks(x)
            ax.set_xticklabels(assets)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. 多资产复合信号 Sharpe/Calmar
        ax = axes[1, 0]
        composite_results = [r for r in results if r['type'] == 'composite']
        if composite_results:
            r = composite_results[0]
            strategies = ['纯多头', '复合信号', '增强信号']
            sharpes = [r['sharpe_buyhold'], r.get('sharpe_composite_periodic', 0), r.get('sharpe_enhanced_periodic', 0)]
            calmars = [r['calmar_buyhold'], r.get('calmar_composite', 0), r.get('calmar_enhanced', 0)]

            x = np.arange(len(strategies))
            ax.bar(x - width/2, sharpes, width, label='Sharpe', alpha=0.8)
            ax.bar(x + width/2, calmars, width, label='Calmar', alpha=0.8)
            ax.set_xlabel('策略')
            ax.set_ylabel('指标值')
            ax.set_title('多资产复合信号 Sharpe/Calmar 对比')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. 触发率
        ax = axes[1, 1]
        if composite_results:
            r = composite_results[0]
            triggers = [r.get('composite_trigger_periodic', 0), r.get('enhanced_trigger_periodic', 0)]
            labels = ['复合信号', '增强信号']

            ax.bar(labels, triggers, alpha=0.8, color=['blue', 'orange'])
            ax.set_ylabel('触发率 (%)')
            ax.set_title('信号触发率')
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
    logging.info("v8 Jump Model 测试 — 最优参数版本")
    logging.info("=" * 60)

    # 加载数据
    logging.info("加载数据...")
    daily_returns = load_daily_returns()
    logging.info(f"  日频收益: {daily_returns.shape}")

    # 选择代表性资产测试
    test_assets = {
        '510300': '沪深300',
        '510500': '中证500',
        '159915': '创业板',
        '511260': '国债',
        '518880': '黄金',
    }

    results = []

    # 单资产测试
    logging.info("\n" + "=" * 60)
    logging.info("单资产测试 — 最优参数")
    logging.info("=" * 60)

    for code, name in test_assets.items():
        if code not in daily_returns.columns:
            continue

        returns = daily_returns[code].dropna()
        if len(returns) < DEFAULT_TRAIN_WINDOW + 100:
            logging.warning(f"  {name} ({code}): 数据不足 ({len(returns)} 天)")
            continue

        logging.info(f"\n  {name} ({code}):")

        # 获取资产类型
        from QuantNodes.strategy.momentum_etf_rotation.v8.signal_composer import ASSET_CLASSES
        from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import TRAIN_WINDOW_MAP, RETRAIN_EVERY_MAP
        asset_type = ASSET_CLASSES.get(code, 'equity')
        train_window = TRAIN_WINDOW_MAP.get(asset_type, 1000)
        retrain_every = RETRAIN_EVERY_MAP.get(asset_type, 30)

        # Jump Model (最优参数)
        logging.info(f"    Jump Model (asset_type={asset_type}, train={train_window}, retrain={retrain_every})...")
        t0 = time.time()
        from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import jump_model_periodic_retrain
        states = jump_model_periodic_retrain(
            returns,
            asset_type=asset_type,
            n_restarts=10,
            show_progress=True,
        )
        elapsed = time.time() - t0

        # 使用 index 对齐
        common_idx = returns.index.intersection(states.index)
        returns_aligned = returns.loc[common_idx]
        states_aligned = states.loc[common_idx]

        # 只从 train_window 之后评估
        oos_start_date = returns.index[train_window]
        returns_oos = returns_aligned.loc[oos_start_date:]
        states_oos = states_aligned.loc[oos_start_date:]

        nav_buyhold = (1 + returns_oos).cumprod()
        signal = states_oos * 0.5
        nav_jump = backtest_with_signal(returns_oos, signal)

        metrics_buyhold = compute_metrics(nav_buyhold)
        metrics_jump = compute_metrics(nav_jump)
        bear_ratio = (states_oos == 1).mean() * 100

        logging.info(f"      Bear%: {bear_ratio:.1f}%")
        logging.info(f"      纯多头: AnnRet={metrics_buyhold['ann_return']*100:.2f}%, Vol={metrics_buyhold['vol']*100:.2f}%, Sharpe={metrics_buyhold['sharpe']:.3f}, MaxDD={metrics_buyhold['max_dd']*100:.2f}%, Calmar={metrics_buyhold['calmar']:.3f}")
        logging.info(f"      Jump:   AnnRet={metrics_jump['ann_return']*100:.2f}%, Vol={metrics_jump['vol']*100:.2f}%, Sharpe={metrics_jump['sharpe']:.3f}, MaxDD={metrics_jump['max_dd']*100:.2f}%, Calmar={metrics_jump['calmar']:.3f}")
        logging.info(f"      耗时: {elapsed:.1f}s")

        results.append({
            'asset': name,
            'code': code,
            'type': 'single',
            'asset_type': asset_type,
            'annret_buyhold': metrics_buyhold['ann_return'],
            'vol_buyhold': metrics_buyhold['vol'],
            'sharpe_buyhold': metrics_buyhold['sharpe'],
            'maxdd_buyhold': metrics_buyhold['max_dd'],
            'calmar_buyhold': metrics_buyhold['calmar'],
            'annret_periodic': metrics_jump['ann_return'],
            'vol_periodic': metrics_jump['vol'],
            'sharpe_periodic': metrics_jump['sharpe'],
            'maxdd_periodic': metrics_jump['max_dd'],
            'calmar_periodic': metrics_jump['calmar'],
            'bear_periodic': bear_ratio,
            'time': elapsed,
        })

    # 多资产复合信号测试
    logging.info("\n" + "=" * 60)
    logging.info("多资产复合信号测试 — 最优参数")
    logging.info("=" * 60)

    asset_pool = ['510300', '510500', '159915', '511260', '518880']
    available = [c for c in asset_pool if c in daily_returns.columns]

    if len(available) >= 3:
        logging.info(f"\n  资产池: {available}")

        from QuantNodes.strategy.momentum_etf_rotation.v8.signal_composer import (
            compute_composite_signal, compute_enhanced_signal, apply_min_duration, ASSET_CLASSES
        )

        logging.info("  Jump Model (最优参数)...")
        asset_signals = {}
        for code in available:
            returns = daily_returns[code].dropna()
            if len(returns) >= DEFAULT_TRAIN_WINDOW + 100:
                asset_type = ASSET_CLASSES.get(code, 'equity')
                states = jump_model_periodic_retrain(
                    returns,
                    asset_type=asset_type,
                    n_restarts=10,
                    show_progress=False,
                )
                asset_signals[code] = states

        composite = compute_composite_signal(asset_signals)
        enhanced = compute_enhanced_signal(
            composite,
            asset_signals.get('511260', pd.Series(0, index=composite.index))
        )
        composite = apply_min_duration(composite, min_duration=60)
        enhanced = apply_min_duration(enhanced, min_duration=60)

        # 回测 (使用 index 对齐)
        market_returns = daily_returns[available].mean(axis=1)
        common_idx = market_returns.index.intersection(composite.index)
        market_returns = market_returns.loc[common_idx]

        # 只从 train_window 之后评估 (使用最大训练窗口)
        max_train_window = max(TRAIN_WINDOW_MAP.values())
        oos_start_date = daily_returns.index[max_train_window]
        market_returns_oos = market_returns.loc[oos_start_date:]

        nav_buyhold = (1 + market_returns_oos).cumprod()

        composite_oos = composite.loc[oos_start_date:]
        enhanced_oos = enhanced.loc[oos_start_date:]

        # 对齐 index
        common_oos_idx = market_returns_oos.index.intersection(composite_oos.index)
        market_returns_oos = market_returns_oos.loc[common_oos_idx]
        composite_oos = composite_oos.loc[common_oos_idx]
        enhanced_oos = enhanced_oos.loc[common_oos_idx]

        nav_composite = backtest_with_signal(market_returns_oos, composite_oos)
        nav_enhanced = backtest_with_signal(market_returns_oos, enhanced_oos)

        metrics_buyhold = compute_metrics(nav_buyhold)
        metrics_composite = compute_metrics(nav_composite)
        metrics_enhanced = compute_metrics(nav_enhanced)

        logging.info(f"\n  等权组合 (OOS: {market_returns_oos.index[0].strftime('%Y-%m-%d')} ~ {market_returns_oos.index[-1].strftime('%Y-%m-%d')}, {len(market_returns_oos)} 天, {len(market_returns_oos)/252:.1f} 年):")
        logging.info(f"    纯多头:     AnnRet={metrics_buyhold['ann_return']*100:.2f}%, Vol={metrics_buyhold['vol']*100:.2f}%, Sharpe={metrics_buyhold['sharpe']:.3f}, MaxDD={metrics_buyhold['max_dd']*100:.2f}%, Calmar={metrics_buyhold['calmar']:.3f}")
        logging.info(f"    复合信号:   AnnRet={metrics_composite['ann_return']*100:.2f}%, Vol={metrics_composite['vol']*100:.2f}%, Sharpe={metrics_composite['sharpe']:.3f}, MaxDD={metrics_composite['max_dd']*100:.2f}%, Calmar={metrics_composite['calmar']:.3f}")
        logging.info(f"    增强信号:   AnnRet={metrics_enhanced['ann_return']*100:.2f}%, Vol={metrics_enhanced['vol']*100:.2f}%, Sharpe={metrics_enhanced['sharpe']:.3f}, MaxDD={metrics_enhanced['max_dd']*100:.2f}%, Calmar={metrics_enhanced['calmar']:.3f}")

        results.append({
            'asset': '等权组合',
            'code': ','.join(available),
            'type': 'composite',
            'oos_start': market_returns_oos.index[0].strftime('%Y-%m-%d'),
            'oos_end': market_returns_oos.index[-1].strftime('%Y-%m-%d'),
            'oos_days': len(market_returns_oos),
            'oos_years': len(market_returns_oos) / 252,
            'annret_buyhold': metrics_buyhold['ann_return'],
            'vol_buyhold': metrics_buyhold['vol'],
            'sharpe_buyhold': metrics_buyhold['sharpe'],
            'maxdd_buyhold': metrics_buyhold['max_dd'],
            'calmar_buyhold': metrics_buyhold['calmar'],
            'annret_composite': metrics_composite['ann_return'],
            'vol_composite': metrics_composite['vol'],
            'sharpe_composite_periodic': metrics_composite['sharpe'],
            'maxdd_composite_periodic': metrics_composite['max_dd'],
            'calmar_composite': metrics_composite['calmar'],
            'annret_enhanced': metrics_enhanced['ann_return'],
            'vol_enhanced': metrics_enhanced['vol'],
            'sharpe_enhanced_periodic': metrics_enhanced['sharpe'],
            'maxdd_enhanced_periodic': metrics_enhanced['max_dd'],
            'calmar_enhanced': metrics_enhanced['calmar'],
            'composite_trigger_periodic': (composite_oos > 0).mean() * 100,
            'enhanced_trigger_periodic': (enhanced_oos > 0).mean() * 100,
        })

    # 保存结果
    logging.info("\n" + "=" * 60)
    logging.info("保存结果")
    logging.info("=" * 60)

    df_results = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "v8_jump_model_test.csv"
    df_results.to_csv(out_csv, index=False)
    logging.info(f"  CSV: {out_csv}")

    # 可视化
    plot_results(results, OUTPUT_DIR / "v8_jump_model_results.png")

    # 生成 Markdown 报告
    from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import TRAIN_WINDOW_MAP, RETRAIN_EVERY_MAP
    
    lines = [
        "# v8 Jump Model 测试报告 — 最优参数版本",
        "",
        f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**训练窗口**: {TRAIN_WINDOW_MAP}",
        f"**重估频率**: {RETRAIN_EVERY_MAP}",
        f"**n_restarts**: 10 (基于 bootstrap 实验)",
        "",
        "## 1. 最优参数",
        "",
        "| 资产类型 | jump_penalty | 特征窗口 | train_window | retrain_every | n_restarts |",
        "|----------|--------------|----------|--------------|---------------|------------|",
        "| 权益类 | 50 | 简单滚动 | 1000 | 30 | 10 |",
        "| 债券类 | 50 | 指数衰减 | 1000 | 30 | 10 |",
        "| 商品类 | 50 | 简单滚动 | 1000 | 30 | 10 |",
        "",
        "## 2. 单资产测试 (OOS)",
        "",
        "| 资产 | 类型 | 纯多头 AnnRet | Jump AnnRet | 纯多头 Vol | Jump Vol | 纯多头 Sharpe | Jump Sharpe | 纯多头 MaxDD | Jump MaxDD | 纯多头 Calmar | Jump Calmar | Bear% |",
        "|------|------|--------------|-------------|-----------|-----------|--------------|--------------|--------------|-------------|--------------|--------------|-------|",
    ]

    for r in results:
        if r['type'] == 'single':
            lines.append(
                f"| {r['asset']} | {r['asset_type']} | {r['annret_buyhold']*100:.2f}% | {r['annret_periodic']*100:.2f}% | "
                f"{r['vol_buyhold']*100:.2f}% | {r['vol_periodic']*100:.2f}% | "
                f"{r['sharpe_buyhold']:.3f} | {r['sharpe_periodic']:.3f} | "
                f"{r['maxdd_buyhold']*100:.2f}% | {r['maxdd_periodic']*100:.2f}% | "
                f"{r['calmar_buyhold']:.3f} | {r['calmar_periodic']:.3f} | {r['bear_periodic']:.1f}% |"
            )

    lines.extend([
        "",
        "## 3. 多资产复合信号测试 (OOS)",
        "",
    ])

    for r in results:
        if r['type'] == 'composite':
            lines.append(f"**OOS 区间**: {r.get('oos_start', 'N/A')} ~ {r.get('oos_end', 'N/A')} ({r.get('oos_days', 0)} 天, {r.get('oos_years', 0):.1f} 年)")
            break

    lines.extend([
        "",
        "| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar | 触发率 |",
        "|------|--------|-----|--------|-------|--------|--------|",
    ])

    for r in results:
        if r['type'] == 'composite':
            lines.append(
                f"| 纯多头 | {r['annret_buyhold']*100:.2f}% | {r['vol_buyhold']*100:.2f}% | {r['sharpe_buyhold']:.3f} | {r['maxdd_buyhold']*100:.2f}% | {r['calmar_buyhold']:.3f} | - |"
            )
            lines.append(
                f"| 复合信号 | {r.get('annret_composite', 0)*100:.2f}% | {r.get('vol_composite', 0)*100:.2f}% | {r.get('sharpe_composite_periodic', 0):.3f} | {r.get('maxdd_composite_periodic', 0)*100:.2f}% | {r.get('calmar_composite', 0):.3f} | {r.get('composite_trigger_periodic', 0):.1f}% |"
            )
            lines.append(
                f"| 增强信号 | {r.get('annret_enhanced', 0)*100:.2f}% | {r.get('vol_enhanced', 0)*100:.2f}% | {r.get('sharpe_enhanced_periodic', 0):.3f} | {r.get('maxdd_enhanced_periodic', 0)*100:.2f}% | {r.get('calmar_enhanced', 0):.3f} | {r.get('enhanced_trigger_periodic', 0):.1f}% |"
            )

    lines.extend([
        "",
        "## 4. Bootstrap 实验结论",
        "",
        "| n_restarts | 沪深300 Bear% CV | 债券 Bear% CV | 推荐 |",
        "|------------|------------------|---------------|------|",
        "| 1 | 12.11% | 28.88% | ❌ |",
        "| 3 | 10.09% | 13.47% | ❌ |",
        "| 5 | 7.81% | 15.15% | ⚠️ |",
        "| **10** | **5.19%** | **7.77%** | **✅** |",
        "| 20 | 3.22% | 3.87% | ✅ (但计算慢) |",
        "",
        "## 5. 跳跃惩罚实验结论",
        "",
        "| 资产 | jump_penalty=25 | jump_penalty=50 | 推荐 |",
        "|------|-----------------|-----------------|------|",
        "| 债券 | Sharpe=1.793 | Sharpe=1.905 | 50 |",
        "",
        "## 6. 特征窗口实验结论",
        "",
        "| 资产 | 简单滚动 | 指数衰减 | 推荐 |",
        "|------|----------|----------|------|",
        "| 沪深300 | Sharpe=1.055 | Sharpe=0.940 | 简单滚动 |",
        "| 债券 | Sharpe=1.793 | Sharpe=3.083 | 指数衰减 |",
    ])

    out_md = OUTPUT_DIR / "v8_jump_model_test.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"  MD: {out_md}")

    logging.info("\n" + "=" * 60)
    logging.info("测试完成!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

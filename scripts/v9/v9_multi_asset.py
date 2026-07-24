# coding=utf-8
"""v9 多资产配置轮动 — 用宏观周期做资产配置.

13 个指数 (2008-2026, 日频):
  股票: 沪深300, 中证500, 中证1000, 恒生指数
  债券: 中债10年, 中债3-5年, 中债国开, 中债企业债, 中债1-3年
  商品: 南华工业品, 南华农产品, 布伦特原油, 沪金

用法:
    python3.11 scripts/v9/v9_multi_asset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from statsmodels.tsa.filters.hp_filter import hpfilter

from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import run_backtest, compute_metrics


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    indices = pd.read_parquet(data_dir / "v9_indices_daily.parquet")
    return macro, indices


def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-10)


def compute_v9_score(macro):
    """v9 组合评分."""
    gz = zscore(macro['宏观增长因子'])
    cz = zscore(macro['宏观通胀因子_生活端'])
    vix = macro['vix'].fillna(method='ffill')
    credit = macro['信用利差因子'].fillna(method='ffill')
    term = macro['期限利差因子_债'].fillna(method='ffill')

    vix_rank = vix.rolling(20).rank(pct=True)
    credit_chg = credit.diff(13)
    term_z = (term - term.mean()) / term.std()

    score = pd.Series(50.0, index=macro.index)
    score += (gz.diff(13) > 0).astype(float) * 15 - 7.5
    score -= (cz.diff(13) > 0).astype(float) * 15 - 7.5
    score += (vix_rank < 0.4).astype(float) * 15 - 7.5
    score -= (credit_chg > 0).astype(float) * 10 - 5
    score += (term_z > 0).astype(float) * 10 - 5

    # 价格趋势
    return score.clip(0, 100)


def get_regime(score):
    """评分 → 宏观阶段."""
    if score > 60:
        return 'Recovery'
    elif score > 50:
        return 'Overheat'
    elif score > 40:
        return 'Recession'
    else:
        return 'Stagflation'


def get_allocation(regime):
    """宏观阶段 → 资产配置权重.

    资产类别:
      股票: 沪深300, 中证500, 中证1000, 恒生指数
      债券: 中债10年, 中债3-5年, 中债国开, 中债企业债, 中债1-3年
      商品: 南华工业品, 南华农产品, 布伦特原油, 沪金
    """
    alloc = {
        'Recovery': {
            'stock': 0.50,  # 经济扩张+通胀回落 → 超配股票
            'bond': 0.20,
            'commodity': 0.20,
            'cash': 0.10,
        },
        'Overheat': {
            'stock': 0.30,  # 经济过热+通胀上行 → 减股配商品
            'bond': 0.10,
            'commodity': 0.40,
            'cash': 0.20,
        },
        'Recession': {
            'stock': 0.15,  # 经济下行+通胀回落 → 超配债券
            'bond': 0.55,
            'commodity': 0.10,
            'cash': 0.20,
        },
        'Stagflation': {
            'stock': 0.10,  # 经济下行+通胀上行 → 超配现金/黄金
            'bond': 0.30,
            'commodity': 0.20,
            'cash': 0.40,
        },
    }
    return alloc.get(regime, alloc['Recession'])


def assign_asset_weights(indices, regime_alloc):
    """将资产类别权重分配到具体指数."""
    asset_map = {
        'stock': ['沪深300指数', '中证500指数', '中证1000', '恒生指数'],
        'bond': ['中债10年期国债指数', '中债3-5年期国债指数', '中债国开行债券总指数',
                 '中债企业债总指数', '中债1-3年国债财富指数'],
        'commodity': ['南华工业品指数', '南华农产品指数', '期货结算价(连续):布伦特原油', '收盘价:沪金指数'],
    }

    weights = pd.Series(0.0, index=indices.columns)
    for asset_class, w in regime_alloc.items():
        if asset_class == 'cash':
            continue
        cols = [c for c in asset_map.get(asset_class, []) if c in indices.columns]
        if cols:
            per_col = w / len(cols)
            weights[cols] = per_col

    return weights


def compute_daily_nav(indices, weekly_weights, cost_bps=5.0):
    """日频回测 (indices 已是日收益率)."""
    daily_ret = indices.copy()

    nav = pd.Series(1.0, index=daily_ret.index)
    prev_weights = pd.Series(0.0, index=daily_ret.columns)

    for i in range(len(daily_ret)):
        date = daily_ret.index[i]
        nearest_week = weekly_weights.index[weekly_weights.index <= date]
        if len(nearest_week) > 0:
            w = weekly_weights.loc[nearest_week[-1]]
        else:
            w = weekly_weights.iloc[0]

        r = daily_ret.iloc[i]
        port_ret = (w * r).sum()

        turnover = (w - prev_weights).abs().sum()
        cost = turnover * cost_bps / 10000.0

        if i == 0:
            nav.iloc[i] = 1.0
        else:
            nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)
        prev_weights = w.copy()

    return nav


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v9 多资产配置轮动 (2008-2026)")
    print("=" * 70)

    macro, indices = load_data()
    print(f"\n数据:")
    print(f"  宏观: {macro.shape}, {macro.index.min().strftime('%Y-%m')} ~ {macro.index.max().strftime('%Y-%m')}")
    print(f"  指数: {indices.shape}, {indices.index.min().strftime('%Y-%m')} ~ {indices.index.max().strftime('%Y-%m')}")

    # 计算 v9 评分
    print(f"\n[Step 1] v9 组合评分")
    score = compute_v9_score(macro)
    print(f"  评分范围: {score.min():.0f} ~ {score.max():.0f}")

    # 映射到日频
    score_daily = score.reindex(indices.index, method='ffill').fillna(50)

    # 生成配置
    print(f"\n[Step 2] 资产配置")
    weekly_dates = pd.date_range(start=indices.index.min(), end=indices.index.max(), freq='W')
    weekly_weights = pd.DataFrame(0.0, index=weekly_dates, columns=indices.columns)

    regime_counts = {'Recovery': 0, 'Overheat': 0, 'Recession': 0, 'Stagflation': 0}
    for wd in weekly_dates:
        nearest = score.index[score.index <= wd]
        if len(nearest) > 0:
            s = score.iloc[-1] if len(nearest) == 0 else score.loc[nearest[-1]]
        else:
            s = 50.0

        regime = get_regime(s)
        regime_counts[regime] += 1
        alloc = get_allocation(regime)
        weights = assign_asset_weights(indices, alloc)
        weekly_weights.loc[wd] = weights

    # 用 forward fill 补全
    weekly_weights = weekly_weights.replace(0, np.nan).ffill().fillna(0)

    print(f"  阶段分布: {regime_counts}")

    # 回测
    print(f"\n[Step 3] 回测")

    # 策略 1: v9 配置轮动
    nav_v9 = compute_daily_nav(indices, weekly_weights, cost_bps=5.0)
    ret_v9 = nav_v9.pct_change().dropna()
    metrics_v9 = compute_metrics(ret_v9, freq='D')
    metrics_v9['strategy'] = 'v9多资产配置'

    # 策略 2: 等权基准
    eq_weights = pd.DataFrame(1.0 / len(indices.columns), index=weekly_weights.index, columns=indices.columns)
    nav_eq = compute_daily_nav(indices, eq_weights, cost_bps=5.0)
    ret_eq = nav_eq.pct_change().dropna()
    metrics_eq = compute_metrics(ret_eq, freq='D')
    metrics_eq['strategy'] = '等权基准'

    # 策略 3: 股债60/40
    bond_cols = [c for c in indices.columns if '债' in c]
    stock_cols = [c for c in indices.columns if any(x in c for x in ['沪深', '中证', '恒生'])]
    w_6040 = pd.DataFrame(0.0, index=weekly_weights.index, columns=indices.columns)
    if stock_cols:
        w_6040[stock_cols] = 0.6 / len(stock_cols)
    if bond_cols:
        w_6040[bond_cols] = 0.4 / len(bond_cols)
    w_6040 = w_6040.replace(0, np.nan).ffill().fillna(0)
    nav_6040 = compute_daily_nav(indices, w_6040, cost_bps=5.0)
    ret_6040 = nav_6040.pct_change().dropna()
    metrics_6040 = compute_metrics(ret_6040, freq='D')
    metrics_6040['strategy'] = '60/40股债'

    # 输出
    print(f"\n  指标对比:")
    df = pd.DataFrame([metrics_eq, metrics_6040, metrics_v9])
    print(df[['strategy', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet', 'Vol', 'WinRate']].to_string(index=False))

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    ax0 = axes[0]
    nav_eq.plot(ax=ax0, label='等权基准', color='#94a3b8', linewidth=1.5, linestyle='--')
    nav_6040.plot(ax=ax0, label='60/40股债', color='#f59e0b', linewidth=1.5, linestyle='-.')
    nav_v9.plot(ax=ax0, label='v9多资产配置', color='#3b82f6', linewidth=2)
    ax0.set_title('v9 多资产配置轮动 (2008-2026)', fontsize=14, fontweight='bold')
    ax0.set_ylabel('NAV')
    ax0.legend(fontsize=11)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(score_daily.index, score_daily.values, color='#3b82f6', linewidth=1)
    ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=40, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('v9评分')
    ax1.set_title('v9 综合评分', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    # 绘制各资产类别权重
    stock_w = weekly_weights[stock_cols].sum(axis=1) if stock_cols else 0
    bond_w = weekly_weights[bond_cols].sum(axis=1) if bond_cols else 0
    comm_w = weekly_weights[[c for c in indices.columns if any(x in c for x in ['南华', '原油', '沪金'])]].sum(axis=1)

    ax2.stackplot(weekly_weights.index, stock_w, bond_w, comm_w,
                  labels=['股票', '债券', '商品'],
                  colors=['#3b82f6', '#10b981', '#f59e0b'], alpha=0.7)
    ax2.set_ylabel('配置比例')
    ax2.set_title('v9 资产配置', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "multi_asset_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 保存
    df.to_csv(output_dir / "multi_asset_results.csv", index=False)
    print(f"\n  {output_dir / 'multi_asset_backtest.png'}")
    print(f"  {output_dir / 'multi_asset_results.csv'}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()
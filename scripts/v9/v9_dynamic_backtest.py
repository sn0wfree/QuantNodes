# coding=utf-8
"""v9 完整回测 — 6 因子动态风险平价.

用法:
    python3.11 scripts/v9/v9_dynamic_backtest.py
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

from QuantNodes.strategy.momentum_etf_rotation.v9.cycle_extractor import (
    extract_cycle, extract_multi_scale_cycles, asset_cycle_extract,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.macro_phase import (
    detect_macro_regime, REGIME_ALLOCATION, REGIME_NAMES_CN,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.trend_factor import (
    compute_trend_factor_fast,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_dynamic_risk_parity, compute_risk_parity_base,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    compute_metrics,
)


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    indices = pd.read_parquet(data_dir / "v9_indices_daily.parquet")
    return macro, indices


def compute_v9_score(macro):
    """v9 组合评分 (用于三维分类)."""
    def zscore(s):
        return (s - s.mean()) / (s.std() + 1e-10)

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
    return score.clip(0, 100)


def run_backtest_daily(weights_df, returns_df, cost_bps=5.0):
    """日频回测."""
    daily_ret = returns_df.copy()
    nav = pd.Series(1.0, index=daily_ret.index)
    prev_w = pd.Series(0.0, index=daily_ret.columns)

    for i in range(len(daily_ret)):
        date = daily_ret.index[i]
        nearest = weights_df.index[weights_df.index <= date]
        w = weights_df.loc[nearest[-1]] if len(nearest) > 0 else weights_df.iloc[0]

        r = daily_ret.iloc[i]
        port_ret = (w * r).sum()
        turnover = (w - prev_w).abs().sum()
        cost = turnover * cost_bps / 10000.0

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost) if i > 0 else 1.0
        prev_w = w.copy()

    return nav


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v9 6 因子动态风险平价回测")
    print("=" * 70)

    macro, indices = load_data()
    print(f"\n数据:")
    print(f"  宏观: {macro.shape}")
    print(f"  指数: {indices.shape}")

    # === Step 1: 三维宏观状态分类 ===
    print(f"\n[Step 1] 三维宏观状态分类")
    regimes, allocations = detect_macro_regime(macro)

    regime_counts = regimes['regime'].value_counts()
    print(f"  状态分布:")
    for regime, count in regime_counts.items():
        name_cn = REGIME_NAMES_CN.get(regime, regime)
        print(f"    {name_cn:8s}: {count:4d}周 ({count/len(regimes):.1%})")

    # === Step 2: 资产周期提取 ===
    print(f"\n[Step 2] 资产周期提取")
    asset_cycles = asset_cycle_extract(indices)
    print(f"  提取完成: {len(asset_cycles)} 个资产")

    # === Step 3: 基础风险平价 ===
    print(f"\n[Step 3] 基础风险平价")
    # 对齐到日频
    returns_daily = indices.copy()
    base_weights = compute_risk_parity_base(returns_daily, lookback=260)
    print(f"  基础权重范围: {base_weights.mean().min():.3f} ~ {base_weights.mean().max():.3f}")

    # === Step 4: 6 因子动态调整 ===
    print(f"\n[Step 4] 6 因子动态调整")
    # 提取资产周期相位
    asset_idx = asset_cycles[list(asset_cycles.keys())[0]]['cycles']['kitchin']['phase'].index
    phases_df = pd.DataFrame(index=asset_idx, columns=indices.columns, dtype=float)
    for col in indices.columns:
        if col in asset_cycles:
            phases_df[col] = asset_cycles[col]['cycles']['kitchin']['phase'].reindex(asset_idx)
    phases_df = phases_df.fillna(0).astype(float)

    # 调整因子
    from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
        compute_velocity_adjustment, compute_acceleration_adjustment,
        compute_cycle叠加_adjustment, compute_correlation_adjustment,
    )

    V = compute_velocity_adjustment(phases_df)
    A = compute_acceleration_adjustment(phases_df)

    phases_medium = pd.DataFrame(index=asset_idx, columns=indices.columns, dtype=float)
    phases_long = pd.DataFrame(index=asset_idx, columns=indices.columns, dtype=float)
    for col in indices.columns:
        if col in asset_cycles:
            phases_medium[col] = asset_cycles[col]['cycles']['juglar']['phase'].reindex(asset_idx)
            phases_long[col] = asset_cycles[col]['cycles']['long_term']['phase'].reindex(asset_idx)
    phases_medium = phases_medium.fillna(0).astype(float)
    phases_long = phases_long.fillna(0).astype(float)

    C = compute_cycle叠加_adjustment(phases_df, phases_medium, phases_long)
    R = compute_correlation_adjustment(returns_daily, lookback=260)
    # 对齐到资产周期索引, 扩展为 DataFrame (与 base_aligned 同形)
    R_aligned = R.reindex(asset_idx, method='ffill').fillna(1.0)
    R_df = pd.DataFrame(
        np.tile(R_aligned.values.reshape(-1, 1), (1, len(indices.columns))),
        index=asset_idx, columns=indices.columns
    )

    print(f"  V 范围: {V.mean().min():.3f} ~ {V.mean().max():.3f}")
    print(f"  A 范围: {A.mean().min():.3f} ~ {A.mean().max():.3f}")
    print(f"  C 范围: {C.mean().min():.3f} ~ {C.mean().max():.3f}")
    print(f"  R 范围: {R_df.mean().min():.3f} ~ {R_df.mean().max():.3f}")

    # 基础权重对齐到资产周期索引
    base_aligned = base_weights.reindex(asset_idx, method='ffill').fillna(0)

    # 最终权重 = 基础 × V × A × C × R (不含 T, T 在回测中单独计算)
    raw_weights = base_aligned * V * A * C * R_df
    # 归一化
    weekly_weights = raw_weights.div(raw_weights.sum(axis=1), axis=0).fillna(0)

    # === Step 5: 趋势因子 (预先计算) ===
    print(f"\n[Step 5] 趋势因子")
    T = pd.DataFrame(1.0, index=returns_daily.index, columns=returns_daily.columns)
    for col in returns_daily.columns:
        series = returns_daily[col].dropna()
        nav = (1 + series).cumprod()
        # 短期动量 (13周)
        mom_short = nav.pct_change(13*5)
        sig_short = np.sign(mom_short)
        # 中期 MA 交叉 (26周)
        ma_fast = nav.rolling(13*5).mean()
        ma_slow = nav.rolling(26*5).mean()
        sig_medium = np.sign(ma_fast - ma_slow)
        # 长期 (52周)
        ma_long = nav.rolling(52*5).mean()
        sig_long = np.sign(nav - ma_long)
        # 融合
        trend_score = 0.4 * sig_short + 0.35 * sig_medium + 0.25 * sig_long
        T[col] = (1.0 + trend_score * 0.3).clip(0.7, 1.3)
    T = T.fillna(1.0)
    print(f"  T 范围: {T.mean().min():.3f} ~ {T.mean().max():.3f}")

    # === Step 6: 回测对比 ===
    print(f"\n[Step 6] 回测")
    returns_daily = indices.copy()

    # 策略 1: 等权基准
    eq_w = pd.DataFrame(1.0/len(indices.columns), index=returns_daily.index, columns=indices.columns)
    nav_eq = run_backtest_daily(eq_w, returns_daily)
    met_eq = compute_metrics(nav_eq.pct_change().dropna(), freq='D')
    met_eq['strategy'] = '等权基准'

    # 策略 2: 基础风险平价
    base_daily = base_weights.reindex(returns_daily.index, method='ffill').fillna(0)
    nav_rp = run_backtest_daily(base_daily, returns_daily)
    met_rp = compute_metrics(nav_rp.pct_change().dropna(), freq='D')
    met_rp['strategy'] = '基础风险平价'

    # 策略 3: 6 因子动态风险平价
    dyn_daily = weekly_weights.reindex(returns_daily.index, method='ffill').fillna(0)
    nav_dyn = run_backtest_daily(dyn_daily, returns_daily)
    met_dyn = compute_metrics(nav_dyn.pct_change().dropna(), freq='D')
    met_dyn['strategy'] = '6因子动态'

    # 策略 4: 6 因子 + 趋势因子
    dyn_t_weights = dyn_daily * T
    dyn_t_weights = dyn_t_weights.div(dyn_t_weights.sum(axis=1), axis=0).fillna(0)
    nav_dyn_t = run_backtest_daily(dyn_t_weights, returns_daily)
    met_dyn_t = compute_metrics(nav_dyn_t.pct_change().dropna(), freq='D')
    met_dyn_t['strategy'] = '6因子+趋势'

    # 输出
    print(f"\n  指标对比:")
    df = pd.DataFrame([met_eq, met_rp, met_dyn, met_dyn_t])
    print(df[['strategy', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet', 'Vol', 'WinRate']].to_string(index=False))

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    ax0 = axes[0]
    nav_eq.plot(ax=ax0, label='等权基准', color='#94a3b8', linewidth=1.5, linestyle='--')
    nav_rp.plot(ax=ax0, label='基础风险平价', color='#f59e0b', linewidth=1.5, linestyle='-.')
    nav_dyn.plot(ax=ax0, label='6因子动态', color='#3b82f6', linewidth=2)
    nav_dyn_t.plot(ax=ax0, label='6因子+趋势', color='#10b981', linewidth=2, linestyle=':')
    ax0.set_title('v9 6 因子动态风险平价 (2008-2026)', fontsize=14, fontweight='bold')
    ax0.set_ylabel('NAV')
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    regimes_mapped = regimes['regime'].map(
        lambda x: list(REGIME_ALLOCATION.keys()).index(x) if x in REGIME_ALLOCATION else 0
    )
    colors = plt.cm.Set3(np.linspace(0, 1, 8))
    for i, regime_key in enumerate(REGIME_ALLOCATION.keys()):
        mask = regimes['regime'] == regime_key
        if mask.any():
            ax1.fill_between(regimes.index, 0, 1, where=mask, alpha=0.3, color=colors[i], label=REGIME_NAMES_CN.get(regime_key, regime_key))
    ax1.set_ylabel('宏观环境')
    ax1.set_title('三维宏观状态', fontsize=12)
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.set_ylim(0, 1)

    ax2 = axes[2]
    dyn_daily.mean(axis=1).plot(ax=ax2, color='#3b82f6', linewidth=1, label='平均仓位')
    ax2.set_ylabel('平均仓位比例')
    ax2.set_title('动态配置仓位', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "dynamic_risk_parity_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 保存
    df.to_csv(output_dir / "dynamic_risk_parity_results.csv", index=False)

    print(f"\n  {output_dir / 'dynamic_risk_parity_backtest.png'}")
    print(f"  {output_dir / 'dynamic_risk_parity_results.csv'}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()
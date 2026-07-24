# coding=utf-8
"""scripts/combo/export_v9_navs.py — 导出 v9 9 策略 NAV (日频) 为 parquet.

复用 v9_citic_all.py 的回测逻辑, 在周频权重基础上:
  1. 周频计算权重 (与 v9_citic_all.py 一致)
  2. 加载日频 ETF 收益 (v56_expanded_daily.parquet)
  3. 周权重 ffill 到日频 → 日频回测 → 日频 NAV

输出:
  - reports/momentum_etf_rotation/combo/v9_navs.parquet (日频, 2058 行)
  - reports/momentum_etf_rotation/combo/v9_metrics.csv

供 combo/nav_curves_html.py 加载, 整合到 STRATEGY_ITERATION_RECORD.html.
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

from QuantNodes.strategy.momentum_etf_rotation.v9.factor_allocator import (
    run_factor_allocator,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_risk_parity_base,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_galaxy import (
    compute_factor_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import run_backtest
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_all_weather import (
    run_all_weather,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_macro import (
    run_macro_allocation,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_multifactor import (
    build_multifactor_weights,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_rotation import (
    build_rotation_weights,
)


def load_weekly_data():
    """加载周频 ETF + 宏观数据 (用于权重计算)."""
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return macro, etf


def load_daily_etf():
    """加载日频 ETF 收益 (用于日频回测)."""
    data_dir = REPO / "data" / "high_freq_macro"
    daily = pd.read_parquet(data_dir / "v56_expanded_daily.parquet")
    return daily


def get_unified_window(strategies):
    first_valid = {}
    for name, w in strategies.items():
        active = w[w.sum(axis=1) > 0.05]
        first_valid[name] = active.index.min() if len(active) > 0 else w.index.max()
    return max(first_valid.values())


def backtest_daily(weekly_weights, daily_etf, cost_bps=5.0):
    """周权重 ffill 到日频 → 信号前等权 → 日频回测 → 日频 NAV."""
    common_etfs = [c for c in weekly_weights.columns if c in daily_etf.columns]
    w = weekly_weights[common_etfs].copy()
    r = daily_etf[common_etfs].copy()

    # 周权重 ffill 到日频
    w_daily = w.reindex(r.index, method='ffill')

    # 信号出现前: 用等权 (1/N) 替代 NaN
    n_etfs = len(common_etfs)
    eq_w = 1.0 / n_etfs
    w_daily = w_daily.fillna(eq_w)

    nav, ret, _ = run_backtest(w_daily, r, cost_bps=cost_bps, freq='D')
    return nav


def main():
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("导出 v9 9 策略 NAV (日频)")
    print("=" * 70)

    # 1. 周频数据 (用于权重计算)
    macro, etf = load_weekly_data()
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]
    print(f"\n周频数据: {etf_clean.shape[0]} 周, {etf_clean.shape[1]} ETF")

    # 2. 日频数据 (用于日频回测)
    daily_etf = load_daily_etf()
    print(f"日频数据: {daily_etf.shape[0]} 天, {daily_etf.shape[1]} ETF")

    # 3. 构造原版 5 策略 (周频权重)
    print(f"\n[Step 1] 构造原版 5 策略 (周频权重)")
    weights_g, factor_score, betas, used_macro = run_factor_allocator(
        returns_df=etf_clean,
        macro_df=macro,
        lookback_score=104,
        lookback_beta=52,
        floor=0.01,
        cap=0.15,
    )

    score_z = (factor_score - factor_score.rolling(52).mean()) / (factor_score.rolling(52).std() + 1e-10)
    active_level = (0.7 - 0.5 * score_z).clip(0.2, 1.0)

    galaxy_w = weights_g.copy()
    dynamic_w = galaxy_w.copy()
    dynamic_w[dynamic_w.columns] = 0
    for date in dynamic_w.index:
        if date in active_level.index:
            level = active_level.loc[date]
            if not np.isnan(level):
                dynamic_w.loc[date] = galaxy_w.loc[date] * level

    eq_w = pd.DataFrame(1.0 / len(etf_clean.columns),
                        index=etf_clean.index, columns=etf_clean.columns)

    bond_etfs = [c for c in etf_clean.columns if c in ['511260', '511010', '511090', '159937', '159816']]
    if not bond_etfs:
        bond_etfs = [c for c in etf_clean.columns if '511' in c or '国债' in c][:3]
    stock_etfs = [c for c in etf_clean.columns if c not in bond_etfs]
    w_6040 = pd.DataFrame(0.0, index=etf_clean.index, columns=etf_clean.columns)
    if stock_etfs:
        w_6040[stock_etfs] = 0.6 / len(stock_etfs)
    if bond_etfs:
        w_6040[bond_etfs] = 0.4 / len(bond_etfs)
    elif stock_etfs:
        w_6040[stock_etfs[:5]] = w_6040[stock_etfs[:5]] + 0.4 / 5

    ret_clean = etf_clean.replace([np.inf, -np.inf], 0).fillna(0)
    rp_w = compute_risk_parity_base(ret_clean, lookback=52)
    rp_w = rp_w.reindex(etf_clean.index, method='ffill').fillna(0)

    # 4. 构造中信 4 策略 (周频权重)
    print(f"\n[Step 2] 构造中信 4 策略 (周频权重)")
    aw_w, _ = run_all_weather(etf_clean, macro)
    ma_w, _ = run_macro_allocation(etf_clean, macro)
    br_w, _ = build_multifactor_weights(etf_clean, top_k=10)
    rt_w, _ = build_rotation_weights(etf_clean, top_k=5)

    strategies_all = {
        '等权基准': eq_w,
        '60/40股债': w_6040,
        '基础风险平价': rp_w,
        '银河因子配置': galaxy_w,
        '银河方案-动态仓位': dynamic_w,
        '中信里昂全天候': aw_w,
        '中信大类资产配置': ma_w,
        '中信多因子选股': br_w,
        '中信行业轮动': rt_w,
    }

    # 5. 找共同起点
    print(f"\n[Step 3] 找共同起点")
    common_start = get_unified_window(strategies_all)
    years = len(etf_clean.index[etf_clean.index >= common_start]) / 52
    print(f"  共同起点: {common_start.date()}")

    # 6. 周频权重 → 日频回测
    print(f"\n[Step 4] 9 策略日频回测 (含 5bp 成本)")
    navs_dict = {}
    results = []
    for name, w in strategies_all.items():
        try:
            w_aligned = w[w.index >= common_start]
            nav = backtest_daily(w_aligned, daily_etf)
            navs_dict[name] = nav
            metrics = compute_factor_metrics(w_aligned, etf_clean, freq='W')
            metrics['strategy'] = name
            metrics['group'] = '原版' if name in ['等权基准', '60/40股债', '基础风险平价', '银河因子配置', '银河方案-动态仓位'] else '中信'
            results.append(metrics)
        except Exception as e:
            print(f"  {name} 失败: {e}")

    df_results = pd.DataFrame(results).sort_values('Sharpe', ascending=False).reset_index(drop=True)

    print(f"\n  9 策略 Sharpe (周频权重):")
    for _, row in df_results.iterrows():
        print(f"    {row['strategy']:20s}: Sharpe={row['Sharpe']:.3f} Calmar={row['Calmar']:.3f} "
              f"MaxDD={row['MaxDD']:.2%} AnnRet={row['AnnRet']:.2%}")

    # 7. 保存日频 NAV
    print(f"\n[Step 5] 保存日频 NAV parquet")
    navs_df = pd.DataFrame(navs_dict)
    navs_df = navs_df.sort_index()
    out_path = out_dir / "v9_navs.parquet"
    navs_df.to_parquet(out_path)
    print(f"  shape: {navs_df.shape}")
    print(f"  columns: {list(navs_df.columns)}")
    print(f"  范围: {navs_df.index[0].date()} ~ {navs_df.index[-1].date()}")
    print(f"  non-null: {navs_df.notna().sum().sum()}")

    # 8. 保存 metrics CSV
    print(f"\n[Step 6] 保存 metrics CSV")
    csv_path = out_dir / "v9_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  保存: {csv_path}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()

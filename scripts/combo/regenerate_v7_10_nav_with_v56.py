# coding=utf-8
"""scripts/combo/regenerate_v7_10_nav_with_v56.py — 用 v56 重跑 v7.10 TV-PR NAV.

直接调用 v7 框架, 但传入 v56 日频数据替代 v7_6.
原 v7_10_gen_nav.py 用的是 load_daily_etf_returns() (v7_6 数据),
本脚本用新生成的 v56_expanded_daily.parquet (简单收益).

输出:
  reports/momentum_etf_rotation/combo/v7_10_nav_v56.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio, calculate_daily_nav,
)


def load_v56_daily_returns() -> pd.DataFrame:
    """加载 v56 日频 ETF 收益 (替代 v7_6)."""
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v56_expanded_daily.parquet")


def compute_metrics(nav, freq="D"):
    """日频或周频 NAV 计算 Sharpe/Calmar/MaxDD."""
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return {}
    freq_map = {"D": 252, "W": 52}
    periods = freq_map.get(freq, 252)
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (periods / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(periods)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0
    dd = (nav / nav.cummax() - 1).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 1e-10 else 0
    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "max_dd": dd,
    }


def main():
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("重新生成 v7.10 TV-PR 日频 NAV (基于 v56)")
    print("=" * 70)

    # 1. 加载 v7.10 周频因子数据
    print("\n[Step 1] 加载 v7.10 周频数据")
    X, Y, codes = load_v7_10_data()
    print(f"  X: {X.shape}, Y: {Y.shape}, codes: {len(codes)}")

    # 2. 加载 v56 日频 ETF 收益 (替代 v7_6)
    print("\n[Step 2] 加载 v56 日频 ETF 收益")
    daily_ret = load_v56_daily_returns()
    print(f"  v56 shape: {daily_ret.shape}")

    # 只保留 v7.10 的 ETF
    common_codes = [c for c in codes if c in daily_ret.columns]
    print(f"  公共 ETF: {len(common_codes)} 个")

    # 3. Beta 估计
    print("\n[Step 3] Beta 估计 (step=4)")
    t0 = time.time()
    beta = expanding_window_tvpr(
        Y, X, 0.06, 0.105,  # BEST_LAMBDA_TV, BEST_LAMBDA_L1
        min_history=52, max_iter=200, tol=1e-5, step=4,
    )
    print(f"  Beta 估计耗时: {time.time() - t0:.1f}s")
    print(f"  beta shape: {beta.shape}")

    # 4. 构造组合 + 日频 NAV
    print("\n[Step 4] 构造组合 + 日频 NAV")
    t0 = time.time()
    cfg = V7_6Config()
    nav_w, weights_df = construct_portfolio(Y, X, beta, cfg, return_weights=True)

    # weights_df 是宽格式 (date, code) → weight
    # 转换为 calculate_daily_nav 期望的长格式 (date, code, weight)
    weights_long = weights_df.reset_index().melt(
        id_vars=weights_df.index.name or 'index',
        var_name='code',
        value_name='weight'
    )
    weights_long = weights_long.rename(columns={weights_df.index.name or 'index': 'date'})

    # 移除 NaN/0 行
    weights_long = weights_long[weights_long['code'].notna()]
    weights_long = weights_long[weights_long['weight'] > 0]

    # 直接调用 v7.10 框架的 calculate_daily_nav
    nav_d = calculate_daily_nav(weights_long, daily_ret, cfg)
    nav_d = nav_d / nav_d.iloc[0]

    # 后处理: 中国假期跳过 (覆盖 ETF NaN fillna(0) 的处理)
    # 由于 calculate_daily_nav 内部 fillna(0), 中国假期会被算 0% 收益
    # 这里直接接受结果, 因为我们的目标是修复对数收益, 不是中国假期
    print(f"  NAV 构造耗时: {time.time() - t0:.1f}s")
    print(f"  v7.10 日频 NAV: {len(nav_d)} 天, {nav_d.index[0].date()} ~ {nav_d.index[-1].date()}")

    # 5. 保存
    print("\n[Step 5] 保存 NAV parquet")
    out_path = out_dir / "v7_10_nav_v56.parquet"
    nav_d.to_frame('v7.10 TV-PR (v56)').to_parquet(out_path)
    print(f"  保存: {out_path}")
    print(f"  shape: {nav_d.shape}")

    # 6. 计算 OOS 指标
    oos = nav_d.loc['2021-08-01':'2026-06-30']
    m = compute_metrics(oos, freq='D')
    print(f"\n  v7.10 OOS (2021-08~2026-06):")
    print(f"  Sharpe={m['sharpe']:.3f}, Calmar={m['calmar']:.3f}, "
          f"AnnRet={m['ann_ret']:.2%}, MaxDD={m['max_dd']:.2%}")

    # 7. 保存 metrics CSV
    df_metrics = pd.DataFrame([{
        'version': 'v7.10_TV-PR',
        'cost_bp': 5.0,  # 默认 5bp
        'sharpe': m['sharpe'],
        'calmar': m['calmar'],
        'ann_ret': m['ann_ret'],
        'ann_vol': m['ann_vol'],
        'max_dd': m['max_dd'],
    }])
    csv_path = out_dir / "v7_10_v56_results.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n保存: {csv_path}")

    print("\n" + "=" * 70)
    print("完成!")


def calculate_daily_nav_with_holiday_skip(weights_df, daily_ret, cfg):
    """与 v7 框架 calculate_daily_nav 类似, 但中国假期跳过 (ETF 数据全 NaN 当日).

    这是对 v7 框架的 calculate_daily_nav 函数的修改版.
    """
    common_codes = [c for c in weights_df.columns if c in daily_ret.columns]
    weights = weights_df[common_codes].copy()
    returns = daily_ret[common_codes].copy()

    # 对齐
    common = weights.index.intersection(returns.index)
    weights = weights.loc[common]
    returns = returns.loc[common]

    cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000 if cfg.cost_enabled else 0.0

    # 计算 NAV (中国假期 ETF 数据全 NaN, 跳过这些日)
    nav = pd.Series(1.0, index=returns.index, dtype=float)

    # 找到 weekly rebal dates
    weekly_dates = weights.index

    for i in range(1, len(returns)):
        d = returns.index[i]
        row = returns.iloc[i]

        # 中国假期判断: ETF 收益全 NaN 当日
        if row[common_codes].isna().all():
            nav.iloc[i] = nav.iloc[i - 1]  # 跳过, NAV 不变
            continue

        # 找到最近的 weekly weights (rebal date <= d)
        applicable_w = weekly_dates[weekly_dates <= d]
        if len(applicable_w) == 0:
            nav.iloc[i] = nav.iloc[i - 1]
            continue

        latest_w = weights.loc[applicable_w[-1]]

        # 计算当日收益 (fillna(0) for missing individual ETF)
        daily_ret_row = row.fillna(0.0)
        port_ret = float((latest_w * daily_ret_row).sum())

        # 扣 weekly 成本 (只在 weekly 调仓日扣)
        turnover = 0.0
        if d in weekly_dates:  # 是调仓日
            # 计算本周的 turnover
            if i > 0:
                prev_w = weights.loc[weekly_dates[weekly_dates < d][-1]] if len(weekly_dates[weekly_dates < d]) > 0 else latest_w
                turnover = float((latest_w - prev_w).abs().sum())
        port_ret -= turnover * cost_rate

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret)

    return nav


if __name__ == "__main__":
    main()
# coding=utf-8
"""v8 Jump Model 参数网格搜索.

网格: bear_threshold × jump_penalty × retrain_every
区间: OOS (2022-02-17 ~ 2026-06-30)
"""
from __future__ import annotations

import sys
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

REPO = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.common.metrics import performance_metrics_legacy as performance_metrics
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio_components,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import expanding_window_tvpr
from QuantNodes.strategy.momentum_etf_rotation.v7.adapters import load_v7_14_data_uniform
from QuantNodes.strategy.momentum_etf_rotation.v8.integration import (
    _compute_daily_nav_from_weights, position_sizing_weights, smooth_weekly_weights,
)

HF_DIR = REPO / "data" / "high_freq_macro"
OUT_DIR = REPO / "reports" / "momentum_etf_rotation"

OOS_START = pd.Timestamp("2022-02-17")


def main():
    print("[grid] 加载数据...")
    daily_returns = pd.read_parquet(HF_DIR / "v7_6_daily_etf_returns.parquet")
    X, Y, codes = load_v7_14_data_uniform()
    cfg = V7_6Config()
    beta = expanding_window_tvpr(Y, X, cfg.lambda_tv, cfg.lambda_l1,
                                 min_history=cfg.min_history, step=cfg.step)
    shares, prices, weekly_weights = construct_portfolio_components(Y, X, beta, cfg)
    print(f"[grid] 数据就绪: {len(weekly_weights)} 周, {len(daily_returns)} 天")

    # 网格搜索参数
    param_grid = {
        "bear_threshold": [0.20, 0.25, 0.30, 0.35, 0.40],
        "jump_penalty": [25, 50, 100],
        "retrain_every": [30, 60],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    print(f"[grid] 共 {len(combos)} 组参数组合")

    # 固定最优平滑参数 (来自 Step 4)
    SMOOTH_ALPHA = 0.7
    SMOOTH_THRESHOLD = 0.01
    COST_BP = 10.0

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        bt = params["bear_threshold"]
        jp = params["jump_penalty"]
        re = params["retrain_every"]

        try:
            # 计算 v8 方案B 权重
            w_b = position_sizing_weights(
                weekly_weights, daily_returns,
                jump_penalty=jp, retrain_every=re,
                bear_threshold=bt,
            )
            # 平滑
            w_smooth = smooth_weekly_weights(
                w_b, alpha=SMOOTH_ALPHA, min_trade_threshold=SMOOTH_THRESHOLD,
            )
            # NAV (含成本)
            nav = _compute_daily_nav_from_weights(w_smooth, daily_returns, cost_bp=COST_BP)

            # 全期和 OOS 指标
            m_full = performance_metrics(nav)
            m_oos = performance_metrics(nav.loc[OOS_START:])

            # 换手率
            diff = w_smooth.diff().abs().sum(axis=1)
            mask = diff > 1e-10
            turnover = diff[mask].mean() * 52 if mask.sum() > 0 else 0

            result = {
                **params,
                "full_ann_ret": m_full["ann_return"],
                "full_sharpe": m_full["sharpe"],
                "full_max_dd": m_full["max_drawdown"],
                "full_calmar": m_full["calmar"],
                "oos_ann_ret": m_oos["ann_return"],
                "oos_sharpe": m_oos["sharpe"],
                "oos_max_dd": m_oos["max_drawdown"],
                "oos_calmar": m_oos["calmar"],
                "turnover": turnover,
            }
            results.append(result)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(combos)}] bt={bt:.2f} jp={jp} re={re} → "
                      f"OOS Sharpe={m_oos['sharpe']:.3f}, Calmar={m_oos['calmar']:.3f}, "
                      f"Turnover={turnover:.1f}x")
        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] bt={bt:.2f} jp={jp} re={re} → ERROR: {e}")

    # 汇总
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "v8_grid_search.csv", index=False)
    print(f"\n[grid] 结果已保存: {OUT_DIR / 'v8_grid_search.csv'}")

    # Top-10 by OOS Calmar
    print("\n" + "=" * 100)
    print("Top-10 by OOS Calmar (扣成本 10bp, 平滑 alpha=0.7 t=0.01)")
    print("=" * 100)
    top = df.nlargest(10, "oos_calmar")
    hdr = f"{'bt':>6s} | {'jp':>4s} | {'re':>4s} | {'OOS_Ret':>8s} | {'OOS_Sharpe':>10s} | {'OOS_MaxDD':>10s} | {'OOS_Calmar':>11s} | {'Turnover':>8s}"
    print(hdr)
    print("-" * 100)
    for _, row in top.iterrows():
        bt_s = f"{row['bear_threshold']:>6.2f}"
        jp_s = f"{int(row['jump_penalty']):>4d}"
        re_s = f"{int(row['retrain_every']):>4d}"
        ret_s = f"{row['oos_ann_ret']*100:>7.2f}%"
        sh_s = f"{row['oos_sharpe']:>10.3f}"
        dd_s = f"{row['oos_max_dd']*100:>9.2f}%"
        cal_s = f"{row['oos_calmar']:>11.3f}"
        to_s = f"{row['turnover']:>7.1f}x"
        print(f"{bt_s} | {jp_s} | {re_s} | {ret_s} | {sh_s} | {dd_s} | {cal_s} | {to_s}")

    # Top-10 by OOS Sharpe
    print("\n" + "=" * 100)
    print("Top-10 by OOS Sharpe")
    print("=" * 100)
    top_s = df.nlargest(10, "oos_sharpe")
    for _, row in top_s.iterrows():
        bt_s = f"{row['bear_threshold']:>6.2f}"
        jp_s = f"{int(row['jump_penalty']):>4d}"
        re_s = f"{int(row['retrain_every']):>4d}"
        ret_s = f"{row['oos_ann_ret']*100:>7.2f}%"
        sh_s = f"{row['oos_sharpe']:>10.3f}"
        dd_s = f"{row['oos_max_dd']*100:>9.2f}%"
        cal_s = f"{row['oos_calmar']:>11.3f}"
        to_s = f"{row['turnover']:>7.1f}x"
        print(f"{bt_s} | {jp_s} | {re_s} | {ret_s} | {sh_s} | {dd_s} | {cal_s} | {to_s}")


if __name__ == "__main__":
    main()

"""scripts/cmf/test_vix_refine.py — 阶段4: VIX自适应参数细化扫描.

在test_vix_adaptive.py的最佳配置附近做密集网格扫描,寻找最优参数组合.
聚焦三种映射函数: zscore, smooth, ratio.
按Calmar排序输出Top20.

回测范围: 2020-2025
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")


def load_base():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    vix_df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].reindex(close.index).ffill()
    return close, vix


def get_raw_signal(close, vix):
    vix_pct = vix.rolling(252).rank(pct=True)
    trend = compute_trend_bnh_signal(
        trailing_stop=0.08, freq="daily"
    )["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    return final_base.clip(0, 1.2)


def scale_zscore(vix, lookback, target, power, floor, ceil):
    ma = vix.rolling(lookback).mean()
    std = vix.rolling(lookback).std()
    zscore = (vix - target) / std.replace(0, np.nan)
    factor = np.exp(-zscore * power).clip(floor, ceil)
    return factor.fillna(1.0)


def scale_smooth(vix, vix_low, vix_high, min_scale, max_scale):
    factor = max_scale - (vix - vix_low) / (vix_high - vix_low) * (max_scale - min_scale)
    return factor.clip(min_scale, max_scale)


def scale_ratio(vix, target_vix, floor, ceil):
    factor = (target_vix / vix).clip(floor, ceil)
    return factor


def evaluate(pos, close, params_str):
    r = calc_metrics(pos, close, "2020-01-01", "2025-12-31")
    return {
        "params": params_str,
        "ann_ret": r["ann_ret"],
        "sharpe": r["sharpe"],
        "max_dd": r["max_dd"],
        "calmar": r["calmar"],
        "turnover": r["turnover"],
    }


def refine_zscore(vix, close, raw):
    """细化zscore参数扫描."""
    results = []
    for lookback in [63, 90, 126, 189, 252]:
        for power in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            for target in [13, 14, 15, 16, 17]:
                for floor, ceil in [(0.3, 1.3), (0.3, 1.5), (0.4, 1.4), (0.5, 1.5)]:
                    factor = scale_zscore(vix, lookback, target, power, floor, ceil)
                    pos = (raw * factor).clip(0, 1.2)
                    s = f"zscore_lb{lookback}_p{power}_t{target}_[{floor},{ceil}]"
                    results.append(evaluate(pos, close, s))
    return results


def refine_smooth(vix, close, raw):
    """细化smooth参数扫描."""
    results = []
    for vix_low in [9, 10, 11, 12, 13]:
        for vix_high in [22, 25, 28, 30, 32]:
            for min_scale in [0.2, 0.3, 0.4]:
                for max_scale in [1.2, 1.3, 1.5, 1.7]:
                    if vix_low >= vix_high:
                        continue
                    factor = scale_smooth(vix, vix_low, vix_high, min_scale, max_scale)
                    pos = (raw * factor).clip(0, 1.2)
                    s = f"smooth_[{vix_low},{vix_high}]/[{min_scale},{max_scale}]"
                    results.append(evaluate(pos, close, s))
    return results


def refine_ratio(vix, close, raw):
    """细化ratio参数扫描."""
    results = []
    for target_vix in [13, 14, 15, 16, 17, 18, 19, 20]:
        for floor in [0.3, 0.4, 0.5]:
            for ceil in [1.2, 1.3, 1.4, 1.5]:
                factor = scale_ratio(vix, target_vix, floor, ceil)
                pos = (raw * factor).clip(0, 1.2)
                s = f"ratio_t{target_vix}_[{floor},{ceil}]"
                results.append(evaluate(pos, close, s))
    return results


def run_all_refinements():
    close, vix = load_base()
    raw = get_raw_signal(close, vix)

    r_base = calc_metrics(raw, close, "2020-01-01", "2025-12-31")
    print(f"=== 基础策略 ===")
    print(f"  年化={r_base['ann_ret']*100:.2f}%, Sharpe={r_base['sharpe']:.2f}, "
          f"MaxDD={r_base['max_dd']*100:.2f}%, Calmar={r_base['calmar']:.2f}\n")

    all_results = []

    print("=== Z-score细化扫描 ===")
    results = refine_zscore(vix, close, raw)
    print(f"  共 {len(results)} 配置")
    all_results.extend(results)

    print("\n=== Smooth细化扫描 ===")
    results = refine_smooth(vix, close, raw)
    print(f"  共 {len(results)} 配置")
    all_results.extend(results)

    print("\n=== Ratio细化扫描 ===")
    results = refine_ratio(vix, close, raw)
    print(f"  共 {len(results)} 配置")
    all_results.extend(results)

    print(f"\n=== 总计 {len(all_results)} 配置 ===\n")

    all_results.sort(key=lambda x: x["calmar"], reverse=True)

    print("=" * 110)
    print("Top 30 按Calmar排序:")
    print("=" * 110)
    print(f'{"配置":<55} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')
    print("-" * 110)
    for i, r in enumerate(all_results[:30], 1):
        print(f'{r["params"]:<55} '
              f'{r["ann_ret"]*100:>7.2f}% '
              f'{r["sharpe"]:>6.2f} '
              f'{r["max_dd"]*100:>7.2f}% '
              f'{r["calmar"]:>6.2f} '
              f'{r["turnover"]:>6.1f}')

    print("\n" + "=" * 110)
    print("Top 30 按Sharpe排序:")
    print("=" * 110)
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f'{"配置":<55} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')
    print("-" * 110)
    for i, r in enumerate(all_results[:30], 1):
        print(f'{r["params"]:<55} '
              f'{r["ann_ret"]*100:>7.2f}% '
              f'{r["sharpe"]:>6.2f} '
              f'{r["max_dd"]*100:>7.2f}% '
              f'{r["calmar"]:>6.2f} '
              f'{r["turnover"]:>6.1f}')


if __name__ == "__main__":
    run_all_refinements()

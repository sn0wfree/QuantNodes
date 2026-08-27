"""scripts/cmf/test_hysteresis.py — P1: 滞回带趋势信号替换trailing stop.

结构缺陷: 现状 gold_trend_bnh.compute_trend_bnh_signal 止损后 peak 重置且无
空仓状态记忆, 33次止损全部次日恢复满仓, 导致熊市 (2013/2015) 跟跌.

修复: 双阈值滞回带 (hysteresis)
- 持仓中: close < peak*(1-ts_out) → 平仓, 锁定空仓状态
- 空仓中: 保持空仓直到 close >= peak*(1-ts_in) 才重新入场
- ts_in = ratio * ts_out (ratio<1 保证重入线高于止损线)

验证窗口:
- OOS: 2010-2019
- IS:  2020-2026
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

OOS_START, OOS_END = "2010-01-01", "2019-12-31"
IS_START, IS_END = "2020-01-01", "2026-08-25"


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
    df = df.drop_duplicates("trade_date", keep="last")
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].sort_index().reindex(close.index).ffill()
    return close, vix


def hysteresis_trend(close, ts_out=0.08, ratio=0.5):
    """双阈值滞回带趋势信号.

    持仓中跌破 peak*(1-ts_out) 平仓; 空仓中收复到 peak*(1-ratio*ts_out) 才回仓.
    """
    ts_in = ratio * ts_out
    pos = pd.Series(np.nan, index=close.index)
    peak = close.iloc[0]
    state = 1.0
    values = close.values
    for i in range(len(close)):
        c = values[i]
        if state == 1.0:
            if c > peak:
                peak = c
            if c < peak * (1 - ts_out):
                state = 0.0
                peak = c  # 重置峰值锚定当前价, 重入线 = c*(1+...) 由新峰值跟踪
        else:
            if c > peak:
                peak = c
            if c >= peak * (1 - ts_in):
                state = 1.0
                peak = c
        pos.iloc[i] = state
    return pos


def get_raw_signal(close, vix, trend_pos):
    """用给定趋势组件组装完整基础信号."""
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_mom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    base = trend_pos * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    return (base + 0.3 * bo * vix_gate).clip(0, 1.2)


def run_p1_grid():
    close, vix = load_base()

    # 基线: 现状 trailing stop=0.08
    trend_old = compute_trend_bnh_signal(
        trailing_stop=0.08, freq="daily"
    )["position"].reindex(close.index).fillna(0)
    raw_old = get_raw_signal(close, vix, trend_old)

    oos_old = calc_metrics(raw_old, close, OOS_START, OOS_END)
    is_old = calc_metrics(raw_old, close, IS_START, IS_END)
    print("=" * 118)
    print("基线: 现状 trailing stop=0.08 完整策略")
    print("=" * 118)
    print(f'{"OOS 2010-2019":<16} 年化={oos_old["ann_ret"]*100:>7.2f}% '
          f'Sharpe={oos_old["sharpe"]:>5.2f} MaxDD={oos_old["max_dd"]*100:>7.2f}% '
          f'Calmar={oos_old["calmar"]:>6.2f}')
    print(f'{"IS 2020-2026":<16} 年化={is_old["ann_ret"]*100:>7.2f}% '
          f'Sharpe={is_old["sharpe"]:>5.2f} MaxDD={is_old["max_dd"]*100:>7.2f}% '
          f'Calmar={is_old["calmar"]:>6.2f}')

    # P1趋势组件单独对比
    hy_default = hysteresis_trend(close, ts_out=0.08, ratio=0.5)
    oos_t_old = calc_metrics(trend_old, close, OOS_START, OOS_END)
    oos_t_hy = calc_metrics(hy_default, close, OOS_START, OOS_END)
    is_t_old = calc_metrics(trend_old, close, IS_START, IS_END)
    is_t_hy = calc_metrics(hy_default, close, IS_START, IS_END)
    print("\n--- 趋势组件单独 ---")
    print(f'ts=0.08现状   OOS 年化={oos_t_old["ann_ret"]*100:>7.2f}% DD={oos_t_old["max_dd"]*100:>7.2f}% | '
          f'IS 年化={is_t_old["ann_ret"]*100:>7.2f}% DD={is_t_old["max_dd"]*100:>7.2f}%')
    print(f'hysteresis默认 OOS 年化={oos_t_hy["ann_ret"]*100:>7.2f}% DD={oos_t_hy["max_dd"]*100:>7.2f}% | '
          f'IS 年化={is_t_hy["ann_ret"]*100:>7.2f}% DD={is_t_hy["max_dd"]*100:>7.2f}%')

    # 网格扫描
    print("\n" + "=" * 118)
    print("P1 网格扫描: ts_out × ratio (完整策略)")
    print("=" * 118)
    results = []
    for ts_out in [0.05, 0.06, 0.08, 0.10, 0.12]:
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            t = hysteresis_trend(close, ts_out=ts_out, ratio=ratio)
            raw = get_raw_signal(close, vix, t)
            oos = calc_metrics(raw, close, OOS_START, OOS_END)
            is_ = calc_metrics(raw, close, IS_START, IS_END)
            results.append({
                "params": f"ts_out={ts_out:.2f}/ratio={ratio:.1f}",
                "oos_ann": oos["ann_ret"], "oos_sh": oos["sharpe"],
                "oos_dd": oos["max_dd"], "oos_cal": oos["calmar"],
                "is_ann": is_["ann_ret"], "is_sh": is_["sharpe"],
                "is_dd": is_["max_dd"], "is_cal": is_["calmar"],
            })

    print(f'{"配置":<26} {"OOS年化":>8} {"OOSSh":>6} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISSh":>6} {"ISDD":>8} {"ISCal":>7} {"均值Cal":>8}')
    print("-" * 118)
    for r in sorted(results, key=lambda x: (x["oos_cal"] + x["is_cal"]) / 2, reverse=True):
        avg = (r["oos_cal"] + r["is_cal"]) / 2
        print(f'{r["params"]:<26} {r["oos_ann"]*100:>7.2f}% {r["oos_sh"]:>6.2f} '
              f'{r["oos_dd"]*100:>7.2f}% {r["oos_cal"]:>7.2f} | '
              f'{r["is_ann"]*100:>7.2f}% {r["is_sh"]:>6.2f} '
              f'{r["is_dd"]*100:>7.2f}% {r["is_cal"]:>7.2f} {avg:>8.2f}')


if __name__ == "__main__":
    run_p1_grid()

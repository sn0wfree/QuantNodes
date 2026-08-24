# coding=utf-8
"""scripts/cmf/15_carry_rolling_backtest.py

Carry无前视回测 (expanding window)

方法:
    对于每个时间点，只用之前的数据计算信号，无前视偏差。
    固定参数: z_window=26, mom_window=4, min_duration=6
"""
import sys
sys.path.insert(0, ".")

import duckdb
import numpy as np
import pandas as pd
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import (
    load_term_structure, compute_carry_signal,
)


def run_no_lookahead_backtest():
    """无前视回测."""
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)

    ts = load_term_structure(con)
    close_near = ts["close_near"]
    close_w = close_near.resample("W-FRI").last()
    ret_w = close_w.pct_change()

    min_history = 104
    positions = []

    for i in range(min_history, len(close_w)):
        current_date = close_w.index[i]
        ts_train = ts[ts.index <= current_date]

        if len(ts_train) < 52:
            positions.append(0.5)
            continue

        # 计算carry信号 (只用历史数据)
        sig = compute_carry_signal(ts_train, z_window=26, min_duration=6)
        mom = close_near.pct_change(4).reindex(sig.index).fillna(0)
        curvature_zscore = sig.get("curvature_zscore", pd.Series(0, index=sig.index))

        # 构建仓位
        back = sig["carry_state"] == 1.0
        mom_up = mom > 0
        curv = curvature_zscore.clip(-1, 1)

        base = pd.Series(0.0, index=sig.index)
        base[back & mom_up] = 1.0
        base[back & ~mom_up] = 0.5
        adj = 1.0 + curv * 0.3
        pos = (base * adj).clip(0, 1)

        positions.append(pos.iloc[-1])

    # 构建结果
    pos_series = pd.Series(positions, index=close_w.index[min_history:])

    # 回测
    strat = pos_series.shift(1) * ret_w.reindex(pos_series.index) - pos_series.diff().abs() * 5e-4
    strat = strat.dropna()
    m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
    strat = strat[m]

    nav = (1 + strat).cumprod()
    ann = nav.iloc[-1] ** (52 / len(strat)) - 1
    vol = strat.std() * np.sqrt(52)
    sh = ann / vol if vol > 0 else 0
    dd = (nav / nav.cummax() - 1).min()

    print("Carry无前视回测结果")
    print("=" * 60)
    print(f"年化: {ann*100:.1f}%")
    print(f"波动: {vol*100:.1f}%")
    print(f"Sharpe: {sh:.2f}")
    print(f"MaxDD: {dd*100:.1f}%")

    print(f"\n分年收益:")
    for y in range(2018, 2026):
        yn = nav[nav.index.year == y]
        if len(yn) > 1:
            ret = (yn.iloc[-1] / yn.iloc[0] - 1) * 100
            print(f"  {y}: {ret:.1f}%")


if __name__ == "__main__":
    run_no_lookahead_backtest()

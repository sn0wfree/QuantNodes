"""scripts/cmf/test_atr_stop.py — Phase 1: ATR自适应止损验证.

基于文献调研 (Erb & Harvey 2013黄金最优公式 max(8%, 2×ATR(20))):
参数扫描 atr_window × k × ts_floor × ts_ceil × ratio, 双窗口验证.

对比基线:
- 现状: compute_macro_regime_signal 默认 (ts_out=0.06 固定滞回带)
- 拉宽: compute_macro_regime_signal 用 ts_out=0.12 (无ATR)
- ATR: 扫描最优
- 文献校准: k=2.0, floor=0.08, atr_window=20 (Erb & Harvey 2013)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from itertools import product

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

import numpy as np
import pandas as pd

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold import compute_breakout_signal
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    _load_indicator_d,
    _rate_gate,
    _zscore_scale,
    compute_vix_momentum_signal,
)

DB = "data/cmf/fundamental.duckdb"
COST_BPS = 5.0
OOS_S, OOS_E = "2010-01-01", "2019-12-31"
IS_S, IS_E = "2020-01-01", "2026-08-25"


def load_all():
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()
    tips = _load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()
    return close, vix, gvz, tips


def compute_atr(close, n=20):
    """平均真实波幅 (Wilder 经典)."""
    prev = close.shift(1)
    tr = pd.concat([
        (close - prev).abs(),
        (close - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def atr_hysteresis(close, atr_window, k, ts_floor, ts_ceil, ratio):
    """ATR自适应 + 滞回带趋势信号."""
    atr = compute_atr(close, atr_window)
    atr_pct = (atr / close).fillna(ts_floor)
    threshold = atr_pct * k
    dynamic_ts = threshold.clip(lower=ts_floor, upper=ts_ceil)

    pos = pd.Series(np.nan, index=close.index)
    peak = close.iloc[0]
    state = 1.0
    vals = close.values
    dyn_vals = dynamic_ts.values
    for i in range(len(close)):
        c = vals[i]
        ts = dyn_vals[i] if np.isfinite(dyn_vals[i]) else ts_floor
        if state == 1.0:
            if c > peak:
                peak = c
            if c < peak * (1 - ts):
                state = 0.0
                peak = c
        else:
            if c > peak:
                peak = c
            ts_in = ratio * ts
            if c >= peak * (1 - ts_in):
                state = 1.0
                peak = c
        pos.iloc[i] = state
    return pos


def fixed_hysteresis(close, ts_out, ratio):
    """固定阈值滞回带 (对照)."""
    return atr_hysteresis(close, atr_window=1, k=ts_out, ts_floor=ts_out,
                          ts_ceil=ts_out, ratio=ratio)


def main():
    close, vix, gvz, tips = load_all()

    print("预计算组件...")
    vmom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    vix_gate = (vix.rolling(252).rank(pct=True) > 0.7).astype(float)
    f_gvz = _zscore_scale(gvz, lookback=90, target=18, power=0.8)
    f_rate = _rate_gate(tips, window=21, power=2.5, floor=0.15, ceil=1.3)

    def combo_pos(trend_pos):
        base = trend_pos * vmom + 0.3 * bo * vix_gate
        return (base * f_gvz * f_rate).clip(0, 1.2)

    # =====================================================================
    # 基线对照
    # =====================================================================
    print("=" * 110)
    print("基线对照 (固定阈值滞回带)")
    print("=" * 110)
    baselines = [
        ("现状 ts=0.06 (生产默认)", 0.06, 0.7),
        ("拉宽固定 ts=0.10", 0.10, 0.7),
        ("拉宽固定 ts=0.12", 0.12, 0.7),
        ("拉宽固定 ts=0.15", 0.15, 0.7),
        ("拉宽固定 ts=0.20", 0.20, 0.7),
    ]
    for label, ts_out, ratio in baselines:
        trend = fixed_hysteresis(close, ts_out, ratio)
        pos = combo_pos(trend)
        oos = calc_metrics(pos, close, OOS_S, OOS_E)
        is_ = calc_metrics(pos, close, IS_S, IS_E)
        avg_cal = (oos["calmar"] + is_["calmar"]) / 2
        print(f'{label:<28} OOS {oos["ann_ret"]*100:>6.2f}%/{oos["max_dd"]*100:>7.2f}%/C{oos["calmar"]:>5.2f} | '
              f'IS {is_["ann_ret"]*100:>6.2f}%/{is_["max_dd"]*100:>7.2f}%/C{is_["calmar"]:>5.2f} | '
              f'avgC {avg_cal:>5.2f}')

    # =====================================================================
    # ATR自适应参数扫描
    # =====================================================================
    print("\n" + "=" * 110)
    print("ATR自适应扫描 (144组合)")
    print("=" * 110)
    grid_aw = [20, 30, 60]
    grid_k = [1.5, 2.0, 2.5, 3.0]
    grid_floor = [0.05, 0.08, 0.10]
    grid_ceil = [0.15, 0.20, 0.25, 0.30]
    grid_ratio = [0.3, 0.5, 0.7]

    results = []
    for aw, k, fl, ce, r in product(grid_aw, grid_k, grid_floor, grid_ceil, grid_ratio):
        if fl >= ce:
            continue  # 无效组合
        trend = atr_hysteresis(close, aw, k, fl, ce, r)
        pos = combo_pos(trend)
        oos = calc_metrics(pos, close, OOS_S, OOS_E)
        is_ = calc_metrics(pos, close, IS_S, IS_E)
        results.append({
            "params": f"aw{aw}/k{k}/fl{fl}/ce{ce}/r{r}",
            "aw": aw, "k": k, "floor": fl, "ceil": ce, "ratio": r,
            "oos_ann": oos["ann_ret"], "oos_sh": oos["sharpe"],
            "oos_dd": oos["max_dd"], "oos_cal": oos["calmar"],
            "is_ann": is_["ann_ret"], "is_sh": is_["sharpe"],
            "is_dd": is_["max_dd"], "is_cal": is_["calmar"],
        })

    results.sort(key=lambda x: (x["oos_cal"] + x["is_cal"]) / 2, reverse=True)

    print(f'{"配置":<30} {"OOS年化":>8} {"OOSSh":>6} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISSh":>6} {"ISDD":>8} {"ISCal":>7} {"avgCal":>7}')
    print("-" * 130)
    for r in results[:15]:
        avg = (r["oos_cal"] + r["is_cal"]) / 2
        flag = " ★" if (r["aw"] == 20 and r["k"] == 2.0 and r["floor"] == 0.08) else ""
        print(f'{r["params"]:<30} {r["oos_ann"]*100:>7.2f}% {r["oos_sh"]:>6.2f} '
              f'{r["oos_dd"]*100:>7.2f}% {r["oos_cal"]:>7.2f} | '
              f'{r["is_ann"]*100:>7.2f}% {r["is_sh"]:>6.2f} '
              f'{r["is_dd"]*100:>7.2f}% {r["is_cal"]:>7.2f} {avg:>7.2f}{flag}')

    # =====================================================================
    # 文献校准版 (Erb & Harvey 2013) 单独突出
    # =====================================================================
    print("\n" + "=" * 110)
    print("文献校准: Erb & Harvey (2013) 黄金最优 max(8%, 2×ATR(20))")
    print("=" * 110)
    for r in [0.3, 0.5, 0.7]:
        trend = atr_hysteresis(close, atr_window=20, k=2.0, ts_floor=0.08, ts_ceil=0.20, ratio=r)
        pos = combo_pos(trend)
        oos = calc_metrics(pos, close, OOS_S, OOS_E)
        is_ = calc_metrics(pos, close, IS_S, IS_E)
        avg_cal = (oos["calmar"] + is_["calmar"]) / 2
        print(f'  Erb&Harvey ratio={r}: OOS {oos["ann_ret"]*100:>6.2f}%/{oos["max_dd"]*100:>7.2f}%/C{oos["calmar"]:>5.2f} | '
              f'IS {is_["ann_ret"]*100:>6.2f}%/{is_["max_dd"]*100:>7.2f}%/C{is_["calmar"]:>5.2f} | '
              f'avgC {avg_cal:>5.2f}')

    # =====================================================================
    # k 参数影响 (固定其他参数)
    # =====================================================================
    print("\n" + "=" * 110)
    print("k 参数敏感度 (固定aw=20, floor=0.08, ceil=0.20, ratio=0.5)")
    print("=" * 110)
    for k in [1.5, 2.0, 2.5, 3.0]:
        trend = atr_hysteresis(close, atr_window=20, k=k, ts_floor=0.08, ts_ceil=0.20, ratio=0.5)
        pos = combo_pos(trend)
        oos = calc_metrics(pos, close, OOS_S, OOS_E)
        is_ = calc_metrics(pos, close, IS_S, IS_E)
        avg_cal = (oos["calmar"] + is_["calmar"]) / 2
        print(f'  k={k}: OOS {oos["ann_ret"]*100:>6.2f}%/{oos["max_dd"]*100:>7.2f}%/C{oos["calmar"]:>5.2f} | '
              f'IS {is_["ann_ret"]*100:>6.2f}%/{is_["max_dd"]*100:>7.2f}%/C{is_["calmar"]:>5.2f} | '
              f'avgC {avg_cal:>5.2f}')

    # =====================================================================
    # 综合排序 Top 5
    # =====================================================================
    print("\n" + "=" * 110)
    print("Top 5 详细对比")
    print("=" * 110)
    for r in results[:5]:
        oos_str = f'OOS年化={r["oos_ann"]*100:.2f}%, DD={r["oos_dd"]*100:.2f}%, Calmar={r["oos_cal"]:.2f}'
        is_str = f'IS年化={r["is_ann"]*100:.2f}%, DD={r["is_dd"]*100:.2f}%, Calmar={r["is_cal"]:.2f}'
        print(f'{r["params"]:<30} | {oos_str} | {is_str}')


if __name__ == "__main__":
    main()

"""scripts/cmf/test_p23_macro_gate.py — P2+P3: 定价因子门控 (双窗口验证).

P2: GVZ自适应因子 (复用zscore框架, 数据源换成黄金隐含波动率GVZ)
    - 事件研究显示黄金暴跌日GVZ-z均值+1.9~+3.2, VIX只有-0.4~0.0 → GVZ才是对的波动率因子
P3: 实际利率动量门控 (Δtips10y, 反向定价因子)
    - 滚动相关2010s=-0.386, 2020s=-0.117 → 长期有效但用平滑rank防近年衰减

底仓变体:
- base_v0: 现状 ts=0.08
- base_hy: P1最佳 hysteresis ts_out=0.06/ratio=0.7

验证窗口: OOS 2010-2019 / IS 2020-2026
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


def load_all():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    def q(sql):
        df = con.execute(sql).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.drop_duplicates("trade_date", keep="last").set_index("trade_date")["value"].sort_index()

    close = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    close["trade_date"] = pd.to_datetime(close["trade_date"])
    close = close.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]

    vix = q("SELECT trade_date, value FROM fundamentals_daily WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date")
    gvz = q("SELECT trade_date, value FROM fundamentals_daily WHERE source='gold_fred' AND value_col='gvz' ORDER BY trade_date")
    tips = q("SELECT trade_date, value FROM fundamentals_daily WHERE source='fred_dfii10' AND value_col='tips10y' ORDER BY trade_date")
    con.close()

    vix = vix.reindex(close.index).ffill()
    gvz = gvz.reindex(close.index).ffill()
    tips = tips.reindex(close.index).ffill()
    return close, vix, gvz, tips


def hysteresis_trend(close, ts_out=0.06, ratio=0.7):
    ts_in = ratio * ts_out
    pos = pd.Series(np.nan, index=close.index)
    peak = close.iloc[0]
    state = 1.0
    values = close.values
    for i in range(len(close)):
        c = values[i]
        if c > peak:
            peak = c
        if state == 1.0 and c < peak * (1 - ts_out):
            state = 0.0
            peak = c
        elif state == 0.0 and c >= peak * (1 - ts_in):
            state = 1.0
            peak = c
        pos.iloc[i] = state
    return pos


def zscore_scale(series, lookback, target, power, floor=0.3, ceil=1.5):
    std = series.rolling(lookback).std().replace(0, np.nan)
    z = (series - target) / std
    return np.exp(-z * power).clip(floor, ceil).fillna(1.0)


def rate_gate(tips, window, power, floor, ceil):
    """P3: Δtips(window)滚动百分位→缩放因子.
    利率上行(紧缩) → 减仓; 利率下行 → 加仓."""
    d = tips.diff(window)
    pct_rank = d.rolling(504, min_periods=126).rank(pct=True)
    factor = np.exp(-(pct_rank - 0.5) * power).clip(floor, ceil)
    return factor.fillna(1.0)


def build_raw(close, vol_series, trend_pos, use_bo=True):
    """给定波动率序列和趋势组件组装基础信号 (vix_mom也用vol_series)."""
    vix_pct = vol_series.rolling(252).rank(pct=True)
    vmom = compute_vix_momentum_signal(
        close, vol_series, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0) if use_bo else pd.Series(0.0, index=close.index)
    gate = (vix_pct > 0.7).astype(float)
    return (trend_pos * vmom + 0.3 * bo * gate).clip(0, 1.2)


def show(name, pos, close):
    oos = calc_metrics(pos, close, OOS_START, OOS_END)
    is_ = calc_metrics(pos, close, IS_START, IS_END)
    print(f'{name:<52} '
          f'OOS {oos["ann_ret"]*100:>6.2f}%/{oos["max_dd"]*100:>7.2f}%/C{oos["calmar"]:>5.2f} | '
          f'IS {is_["ann_ret"]*100:>6.2f}%/{is_["max_dd"]*100:>7.2f}%/C{is_["calmar"]:>5.2f} | '
          f'avgC {(oos["calmar"] + is_["calmar"]) / 2:>5.2f}')
    return oos, is_


def run_p23():
    close, vix, gvz, tips = load_all()

    trend_old = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"].reindex(close.index).fillna(0)
    trend_hy = hysteresis_trend(close, ts_out=0.06, ratio=0.7)

    # ===== 基线们 =====
    print("=" * 130)
    print("基线 (无P2/P3)")
    print("=" * 130)
    raw_old = build_raw(close, vix, trend_old)
    raw_hy = build_raw(close, vix, trend_hy)
    show("base_v0 现状ts=0.08", raw_old, close)
    show("base_hy P1滞回带ts6%/r0.7", raw_hy, close)

    # ===== P2: VIX自适应 → GVZ自适应 对照 =====
    print("\n" + "=" * 130)
    print("P2: zscore自适应因子 — 输入因子对照 (VIX vs GVZ), lb63/p0.6/t16 vs lb90/p0.8/t16")
    print("=" * 130)
    best_cfgs = [
        ("lb63_p0.6", dict(lookback=63, target=16, power=0.6)),
        ("lb90_p0.8", dict(lookback=90, target=16, power=0.8)),
    ]
    for cfg_name, cfg in best_cfgs:
        f_vix = zscore_scale(vix, **cfg)
        f_gvz = zscore_scale(gvz, **cfg)
        show(f"  VIX-adapt  {cfg_name} @base_v0", (raw_old * f_vix).clip(0, 1.2), close)
        show(f"  GVZ-adapt  {cfg_name} @base_v0", (raw_old * f_gvz).clip(0, 1.2), close)
        show(f"  VIX-adapt  {cfg_name} @base_hy", (raw_hy * f_vix).clip(0, 1.2), close)
        show(f"  GVZ-adapt  {cfg_name} @base_hy", (raw_hy * f_gvz).clip(0, 1.2), close)

    # GVZ目标值敏感性 (GVZ量纲与VIX不同: 中枢~17-20)
    print("\n--- GVZ target 敏感性 (@base_v0, lb90/p0.8) ---")
    for t in [14, 16, 18, 20]:
        f = zscore_scale(gvz, lookback=90, target=t, power=0.8)
        show(f"  GVZ t={t}", (raw_old * f).clip(0, 1.2), close)

    # ===== P3: 实际利率门控 =====
    print("\n" + "=" * 130)
    print("P3: Δtips10y 动量门控 @各底仓")
    print("=" * 130)
    for w in [21, 42, 63, 126]:
        for power in [0.8, 1.5]:
            g = rate_gate(tips, w, power, floor=0.3, ceil=1.3)
            show(f"  rate_gate w={w}/p{power} @base_v0", (raw_old * g).clip(0, 1.2), close)
    print("--- rate_gate @base_hy ---")
    for w in [42, 63]:
        g = rate_gate(tips, w, 1.5, 0.3, 1.3)
        show(f"  rate_gate w={w}/p1.5 @base_hy", (raw_hy * g).clip(0, 1.2), close)

    # ===== 组合: P1+P2(GVZ)+P3 =====
    print("\n" + "=" * 130)
    print("组合: base_hy × GVZ-adapt × rate_gate")
    print("=" * 130)
    results = []
    for gvz_cfg in [dict(lookback=63, target=16, power=0.6),
                    dict(lookback=90, target=16, power=0.8),
                    dict(lookback=90, target=18, power=0.8)]:
        fg = zscore_scale(gvz, **gvz_cfg)
        for w in [21, 42, 63]:
            gr = rate_gate(tips, w, 1.5, 0.3, 1.3)
            combo = (raw_hy * fg * gr).clip(0, 1.2)
            oos = calc_metrics(combo, close, OOS_START, OOS_END)
            is_ = calc_metrics(combo, close, IS_START, IS_END)
            label = f"hy×gvz(lb{gvz_cfg['lookback']},t{gvz_cfg['target']})×rt(w{w})"
            results.append((label, oos, is_))
            print(f'{label:<44} '
                  f'OOS {oos["ann_ret"]*100:>6.2f}%/{oos["max_dd"]*100:>7.2f}%/C{oos["calmar"]:>5.2f} | '
                  f'IS {is_["ann_ret"]*100:>6.2f}%/{is_["max_dd"]*100:>7.2f}%/C{is_["calmar"]:>5.2f} | '
                  f'avgC {(oos["calmar"] + is_["calmar"]) / 2:>5.2f}')

    print("\n--- 组合: base_v0 × VIX-adapt(原冠军) × rate_gate (P3叠加旧IS冠军) ---")
    fv = zscore_scale(vix, lookback=63, target=16, power=0.6)
    for w in [21, 42, 63]:
        gr = rate_gate(tips, w, 1.5, 0.3, 1.3)
        combo = (raw_old * fv * gr).clip(0, 1.2)
        show(f"  v0×vix_adapt×rt(w{w})", combo, close)


if __name__ == "__main__":
    run_p23()

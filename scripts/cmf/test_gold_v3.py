"""scripts/cmf/test_gold_v3.py — 完整风控体系 v3 (修正版).

核心思路:
    1. 基础仓位: 趋势B&H (8%止损) + VIX动量
    2. 波动率自适应加杠杆: VIX低位+平稳时加杠杆
    3. MaxDD控制: 净值回撤10%清仓+恢复
    4. 突破/趋势仅在压力/危机时使用
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_momentum_signal, compute_real_rate_signal,
    compute_risk_signal, compute_position_flow_signal, compute_trend_bnh_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")


def load_data():
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


def vix_momentum_signal(vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                       high_threshold=0.80, accel_threshold=0.08):
    position = pd.Series(1.0, index=vix.index)
    for i in range(1, len(vix)):
        vp = vix_pct.iloc[i] if i < len(vix_pct) else 0.5
        vc = vix_chg_5d.iloc[i] if i < len(vix_chg_5d) else 0
        m5 = vix_ma5.iloc[i] if i < len(vix_ma5) else vix.iloc[i]
        m20 = vix_ma20.iloc[i] if i < len(vix_ma20) else vix.iloc[i]
        vix_accel = m5 > m20
        if vp < 0.5:
            position.iloc[i] = 1.0
        elif vp < high_threshold:
            position.iloc[i] = 0.9 if (vix_accel and vc > accel_threshold) else 0.95
        else:
            if vix_accel and vc > accel_threshold:
                position.iloc[i] = 0.4
            elif vix_accel:
                position.iloc[i] = 0.6
            else:
                position.iloc[i] = 0.8
    return position


def full_risk_system(close, vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                     base_trend, vix_mom, breakout, momentum, real_rate, position_flow,
                     max_leverage=1.0, max_dd=-0.10, recovery_threshold=0.02):
    base_pos = base_trend * vix_mom

    leverage = pd.Series(1.0, index=close.index)
    for i in range(1, len(close)):
        vp = vix_pct.iloc[i]
        vix_accel = vix_ma5.iloc[i] > vix_ma20.iloc[i]
        if vp < 0.3 and not vix_accel:
            leverage.iloc[i] = max_leverage
        elif vp < 0.5 and not vix_accel:
            leverage.iloc[i] = max_leverage * 0.8
        elif vp < 0.7:
            leverage.iloc[i] = 0.7
        else:
            leverage.iloc[i] = 0.5

    position = base_pos * leverage
    position = position.clip(0, max_leverage)

    if max_dd is not None:
        ret = close.pct_change()
        strat_ret = position.shift(1) * ret
        nav = (1 + strat_ret.fillna(0)).cumprod()
        rolling_max = nav.cummax()
        dd = nav / rolling_max - 1

        clean_position = position.copy()
        in_drawdown = False
        for i in range(1, len(position)):
            if dd.iloc[i] < max_dd and not in_drawdown:
                clean_position.iloc[i:] = 0.0
                in_drawdown = True
                drawdown_peak = rolling_max.iloc[i]
            elif in_drawdown:
                recovery = nav.iloc[i] / drawdown_peak - 1
                if recovery > recovery_threshold:
                    target = base_pos.iloc[i] * vix_mom.iloc[i]
                    clean_position.iloc[i] = target
                    if recovery > abs(max_dd) * 0.5:
                        in_drawdown = False
                else:
                    clean_position.iloc[i] = 0.0
        position = clean_position

    return position


def calc_metrics(pos, close, start, end):
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    start_price = c.iloc[0]
    end_price = c.iloc[-1]
    total_ret = end_price / start_price - 1
    n_y = len(c) / 252

    ret = c.pct_change()
    sr = (pos.reindex(sub.index).fillna(0).shift(1) * ret).dropna()
    cost = pos.reindex(sub.index).fillna(0).diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()

    if n_y <= 0 or len(sr) == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}

    sr_start = c.iloc[0]
    sr_end = c.iloc[-1]
    period_return = (1 + sr).prod() - 1
    ann_ret = (1 + period_return) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)

    sr_sum = (sr.fillna(0) - 0).sum()
    nav = sr_start * (1 + sr).cumprod()
    dd = (nav / nav.cummax() - 1).min()

    to = pos.reindex(sub.index).fillna(0).diff().abs().sum() / 2 / n_y

    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0

    return {
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd": dd,
        "turnover": to,
        "calmar": calmar,
        "total_ret": total_ret,
    }


def run_tests():
    close, vix = load_data()
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_chg_5d = vix.pct_change(5)
    vix_ma5 = vix.rolling(5).mean()
    vix_ma20 = vix.rolling(20).mean()

    trend = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"].reindex(close.index).fillna(0)
    vix_mom = vix_momentum_signal(vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20)
    breakout = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    momentum = compute_momentum_signal(mom_window=52)["position"].reindex(close.index).fillna(0)
    real_rate = compute_real_rate_signal()["position"].reindex(close.index).fillna(0)
    position_flow = compute_position_flow_signal()["position"].reindex(close.index).fillna(0)

    bh_pos = pd.Series(1.0, index=close.index)
    bh_r = calc_metrics(bh_pos, close, "2018-01-01", "2025-12-31")
    print(f"\n=== B&H基准 (2018-2025) ===")
    print(f"  年化={bh_r['ann_ret']*100:.2f}%, Sharpe={bh_r['sharpe']:.2f}, "
          f"MaxDD={bh_r['max_dd']*100:.2f}%, Calmar={bh_r['calmar']:.2f}, 换手=0")

    results = []

    for start, end, range_label in [
        ("2018-01-01", "2025-12-31", "2018-2025"),
        ("2012-01-01", "2025-12-31", "2012-2025"),
        ("2008-01-01", "2025-12-31", "2008-2025"),
    ]:
        print(f"\n=== 数据范围: {range_label} ===")
        for max_dd in [-0.10, -0.12, -0.15]:
            for max_lev in [1.2, 1.5, 2.0]:
                pos = full_risk_system(
                    close, vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                    trend, vix_mom, breakout, momentum, real_rate, position_flow,
                    max_leverage=max_lev, max_dd=max_dd,
                )
                r = calc_metrics(pos, close, start, end)
                key = f"{range_label}|MaxDD={max_dd*100:.0f}%|Lev={max_lev}x"
                results.append({**r, "key": key, "range": range_label,
                                "max_dd": max_dd, "max_lev": max_lev})
                print(f"  {key}: 年化={r['ann_ret']*100:>6.2f}%, "
                      f"Sharpe={r['sharpe']:>5.2f}, MaxDD={r['max_dd']*100:>6.2f}%, "
                      f"Calmar={r['calmar']:>5.2f}, 换手={r['turnover']:>4.1f}")

    for range_name in ["2018-2025", "2012-2025", "2008-2025"]:
        print(f"\n=== {range_name} 最优参数 (by Calmar) ===")
        results_r = [r for r in results if r["range"] == range_name]
        for r in sorted(results_r, key=lambda x: -x["calmar"])[:5]:
            print(f"  {r['key']}: 年化={r['ann_ret']*100:.2f}%, "
                  f"Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']*100:.2f}%, "
                  f"Calmar={r['calmar']:.2f}")


if __name__ == "__main__":
    run_tests()

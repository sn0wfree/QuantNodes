"""scripts/cmf/test_gold_full_system.py — 黄金完整风控体系参数测试.

测试维度:
    1. MaxDD限制: -10% / -12% / -15%
    2. 杠杆上限: 1.2x / 1.5x / 2.0x
    3. 再优化频率: 月度 / 季度 / 半年
    4. 回测数据范围: 2018-2025 / 2012-2025 / 2008-2025
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


def detect_regime(vix_pct, price_ma200_ratio, vix_ma5, vix_ma20):
    regimes = pd.Series("normal", index=vix_pct.index)
    is_bull = (vix_pct < 0.5) & (price_ma200_ratio > 1.05)
    is_stress = (vix_pct >= 0.7) & (vix_pct < 0.85) & (price_ma200_ratio < 0.95)
    is_crisis = (vix_pct >= 0.85) | (vix_ma5 > vix_ma20) & (vix_pct > 0.7)
    regimes[is_bull] = "bull"
    regimes[is_stress] = "stress"
    regimes[is_crisis] = "crisis"
    return regimes


def dynamic_weights(regime):
    weight_map = {
        "bull": {"bo": 0.0, "mom": 0.20, "rr": 0.15, "risk": 0.10, "pf": 0.35, "trend": 0.0, "vix_mom": 0.20},
        "normal": {"bo": 0.0, "mom": 0.20, "rr": 0.15, "risk": 0.10, "pf": 0.35, "trend": 0.0, "vix_mom": 0.20},
        "stress": {"bo": 0.10, "mom": 0.15, "rr": 0.10, "risk": 0.05, "pf": 0.45, "trend": 0.10, "vix_mom": 0.05},
        "crisis": {"bo": 0.20, "mom": 0.05, "rr": 0.05, "risk": 0.0, "pf": 0.60, "trend": 0.20, "vix_mom": -0.10},
    }
    return weight_map.get(regime, weight_map["normal"])


def factor_score_strategy(close, vix, bo, mom, rr, risk, pf, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                          lookback=60, max_leverage=1.0, max_dd=-0.10):
    price_ma200 = close.rolling(200).mean()
    price_ratio = close / price_ma200
    regimes = detect_regime(vix_pct, price_ratio, vix_ma5, vix_ma20)

    def rolling_zscore(s, window=lookback):
        return (s - s.rolling(window).mean()) / s.rolling(window).std()

    score_bo = rolling_zscore(bo, lookback)
    score_mom = rolling_zscore(mom, lookback)
    score_rr = rolling_zscore(rr, lookback)
    score_risk = rolling_zscore(risk, lookback)
    score_pf = rolling_zscore(pf, lookback)

    position = pd.Series(0.0, index=close.index)
    for i in range(lookback, len(close)):
        regime = regimes.iloc[i]
        w = dynamic_weights(regime)

        score = (
            w["bo"] * score_bo.iloc[i] +
            w["mom"] * score_mom.iloc[i] +
            w["rr"] * score_rr.iloc[i] +
            w["risk"] * score_risk.iloc[i] +
            w["pf"] * score_pf.iloc[i]
        )
        pos = 1 / (1 + np.exp(-score))
        position.iloc[i] = pos

    vix_mom = vix_momentum_signal(vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20)
    position = position * vix_mom
    position = position.clip(0, max_leverage)

    if max_dd is not None:
        ret = close.pct_change()
        strat_ret = position.shift(1) * ret
        nav = (1 + strat_ret.fillna(0)).cumprod()
        rolling_max = nav.cummax()
        dd = nav / rolling_max - 1
        triggered = (dd < max_dd).astype(int).diff().fillna(0)
        triggered = triggered.replace(0, np.nan).ffill().fillna(0)
        for i in range(len(position)):
            if triggered.iloc[i] == 1:
                nav_so_far = nav.iloc[:i+1]
                peak_idx = nav_so_far.idxmax()
                recovery = nav.iloc[i] / nav.loc[peak_idx] - 1
                if recovery > 0:
                    position.iloc[i:] = position.iloc[i:].clip(0, max_leverage * 0.3)

    return position


def calc_metrics(pos, close, start, end):
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (pos.reindex(sub.index).fillna(0).shift(1) * ret).dropna()
    cost = pos.reindex(sub.index).fillna(0).diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()
    nav = c.iloc[0] * (1 + sr).cumprod()
    if len(sr) == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    tr = nav.iloc[-1] / c.iloc[0] - 1
    n_y = len(sr) / 252
    ar = (1 + tr) ** (1/n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    to = pos.reindex(sub.index).fillna(0).diff().abs().sum() / 2 / n_y
    calmar = ar / abs(dd) if dd != 0 else 0
    return {"ann_ret": ar, "sharpe": ar/vol if vol > 0 else 0, "max_dd": dd, "turnover": to, "calmar": calmar}


def run_tests():
    close, vix = load_data()
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_chg_5d = vix.pct_change(5)
    vix_ma5 = vix.rolling(5).mean()
    vix_ma20 = vix.rolling(20).mean()

    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    mom = compute_momentum_signal(mom_window=52)["position"].reindex(close.index).fillna(0)
    rr = compute_real_rate_signal()["position"].reindex(close.index).fillna(0)
    risk = compute_risk_signal()["position"].reindex(close.index).fillna(0)
    pf = compute_position_flow_signal()["position"].reindex(close.index).fillna(0)

    results = []

    for start, end, range_label in [
        ("2018-01-01", "2025-12-31", "2018-2025"),
        ("2012-01-01", "2025-12-31", "2012-2025"),
        ("2008-01-01", "2025-12-31", "2008-2025"),
    ]:
        print(f"\n=== 数据范围: {range_label} ===")
        for max_dd in [-0.10, -0.12, -0.15]:
            for max_lev in [1.2, 1.5, 2.0]:
                pos = factor_score_strategy(
                    close, vix, bo, mom, rr, risk, pf,
                    vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                    max_leverage=max_lev, max_dd=max_dd,
                )
                r = calc_metrics(pos, close, start, end)
                key = f"{range_label}|MaxDD={max_dd*100:.0f}%|Lev={max_lev}x"
                results.append({**r, "key": key, "range": range_label,
                                "max_dd": max_dd, "max_lev": max_lev})
                print(f"  {key}: 年化={r['ann_ret']*100:>6.2f}%, "
                      f"Sharpe={r['sharpe']:>5.2f}, MaxDD={r['max_dd']*100:>6.2f}%, "
                      f"Calmar={r['calmar']:>5.2f}, 换手={r['turnover']:>4.1f}")

    print("\n=== 2018-2025 最优参数 ===")
    results_2018 = [r for r in results if r["range"] == "2018-2025"]
    for r in sorted(results_2018, key=lambda x: -x["calmar"])[:5]:
        print(f"  {r['key']}: 年化={r['ann_ret']*100:.2f}%, "
              f"Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']*100:.2f}%, "
              f"Calmar={r['calmar']:.2f}")

    print("\n=== 2012-2025 最优参数 ===")
    results_2012 = [r for r in results if r["range"] == "2012-2025"]
    for r in sorted(results_2012, key=lambda x: -x["calmar"])[:5]:
        print(f"  {r['key']}: 年化={r['ann_ret']*100:.2f}%, "
              f"Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']*100:.2f}%, "
              f"Calmar={r['calmar']:.2f}")

    print("\n=== 2008-2025 最优参数 ===")
    results_2008 = [r for r in results if r["range"] == "2008-2025"]
    for r in sorted(results_2008, key=lambda x: -x["calmar"])[:5]:
        print(f"  {r['key']}: 年化={r['ann_ret']*100:.2f}%, "
              f"Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']*100:.2f}%, "
              f"Calmar={r['calmar']:.2f}")


if __name__ == "__main__":
    run_tests()

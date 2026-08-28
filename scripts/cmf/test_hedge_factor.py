"""scripts/cmf/test_hedge_factor.py — 阶段1: 对冲因子增强.

使用~/Public/gold/HedgeFactor_Composite_Factor.xlsx中的预计算因子,
对基础策略(趋势B&H 8% + VIX动量 80%/8% + 突破alpha=0.3)进行增强.

回测范围: 2018-2025 (与基础策略一致)
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

from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")
HEDGE_PATH = Path.home() / "Public" / "gold" / "HedgeFactor_Composite_Factor.xlsx"


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


def load_hedge_factor():
    df = pd.read_excel(HEDGE_PATH)
    df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])
    df = df.set_index("Unnamed: 0")
    return df


def calc_metrics(pos, close, start, end, cost_bps=5.0):
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * cost_bps / 10000
    sr = (sr - cost).dropna()
    if len(sr) == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    start_price = c.iloc[0]
    nav = start_price * (1 + sr).cumprod()
    total_ret = nav.iloc[-1] / start_price - 1
    n_y = len(c) / 252
    ar = (1 + total_ret) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    to = sub.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "turnover": to,
            "calmar": ar / abs(dd) if dd != 0 else 0}


def calc_2026(pos, close, start="2026-01-01", end="2026-08-25"):
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    sr = (sub.shift(1) * c.pct_change()).dropna()
    cost = sub.diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()
    return (1 + sr).prod() - 1 if len(sr) > 0 else 0


def run_hedge_factor_tests():
    close, vix = load_data()
    hedge = load_hedge_factor()
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_ma5 = vix.rolling(5).mean()
    vix_ma20 = vix.rolling(20).mean()

    trend = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)

    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    final_base = final_base.clip(0, 1.2)

    # 对齐HedgeFactor到close的index
    hedge_aligned = hedge.reindex(close.index)

    # 基础策略(无对冲因子)
    print("\n=== 对冲因子增强测试 (2018-2025) ===")
    print(f'{"配置":<45} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')
    print("-" * 90)

    configs = [
        ("基础策略 (无对冲因子)", final_base),
    ]

    # 方案1: 直接用HedgeFactor_Composite_Factor
    for factor_name, factor_col in [
        ("HedgeFactor_Composite_Factor", "HedgeFactor_Composite_Factor"),
        ("HedgeFactor_ETF60Fwd40", "HedgeFactor_ETF60Fwd40"),
        ("HedgeFactor_vol_weigheted", "HedgeFactor_vol_weigheted"),
    ]:
        if factor_col in hedge_aligned.columns:
            factor = hedge_aligned[factor_col].fillna(0)
            for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
                combined = final_base * (1 + w * factor)
                combined = combined.clip(0, 1.5)
                configs.append((f"+{factor_col[:15]} w={w}", combined))

    # 方案2: 对冲因子作为仓位缩放
    for factor_name, factor_col in [
        ("HedgeFactor_Composite_Factor", "HedgeFactor_Composite_Factor"),
    ]:
        if factor_col in hedge_aligned.columns:
            factor = hedge_aligned[factor_col].fillna(0)
            for w in [0.1, 0.2, 0.3]:
                # 因子为正时加仓, 为负时减仓
                scale = 1 + w * factor
                combined = final_base * scale
                combined = combined.clip(0, 1.5)
                configs.append((f"scale*{factor_col[:10]} w={w}", combined))

    results = []
    for name, pos in configs:
        r = calc_metrics(pos, close, "2018-01-01", "2025-12-31")
        r_2026 = calc_2026(pos, close)
        results.append({
            "name": name,
            "ar": r["ann_ret"],
            "sharpe": r["sharpe"],
            "dd": r["max_dd"],
            "calmar": r["calmar"],
            "to": r["turnover"],
            "r_2026": r_2026,
        })
        print(f'  {name:<43} {r["ann_ret"]*100:>6.2f}% {r["sharpe"]:>6.2f} '
              f'{r["max_dd"]*100:>6.2f}% {r["calmar"]:>6.2f} {r["turnover"]:>5.1f}  '
              f'2026: {r_2026*100:>+6.2f}%')

    print("\n=== Top 5 by Calmar (2018-2025) ===")
    for r in sorted(results, key=lambda x: -x["calmar"])[:5]:
        print(f'  {r["name"]}: 年化={r["ar"]*100:.2f}%, '
              f'Sharpe={r["sharpe"]:.2f}, MaxDD={r["dd"]*100:.2f}%, '
              f'Calmar={r["calmar"]:.2f}, 2026={r["r_2026"]*100:+.2f}%')

    print("\n=== 最佳推荐 ===")
    best = max(results, key=lambda x: x["calmar"])
    print(f"推荐: {best['name']}")
    print(f"  年化={best['ar']*100:.2f}%, Sharpe={best['sharpe']:.2f}, "
          f"MaxDD={best['dd']*100:.2f}%, Calmar={best['calmar']:.2f}")


if __name__ == "__main__":
    run_hedge_factor_tests()

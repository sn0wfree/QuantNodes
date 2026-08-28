"""scripts/cmf/prepare_gold_overview_data.py — 综合介绍图数据准备 (缓存).

计算并缓存绘图所需全部数据 → reports/gold/overview_data.pkl
后续绘图直接加载缓存, 节约时间.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

import duckdb
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal, compute_risk_overlay,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import _load_indicator_d

CACHE = "reports/gold/overview_data.pkl"
COST_BPS = 5.0


def main():
    # ===== 1. 加载价格与信号 =====
    print("[1/6] 加载价格与生产信号 (shift2)...")
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    # 使用渐进式换月调整后的价格 (4天线性插值)
    df = con.execute("SELECT trade_date, open, close FROM futures_main_close_au_adj ORDER BY trade_date").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    close = df["close"]
    open_price = df["open"]

    sig = compute_layered_ensemble_signal()
    pos_base = sig["position"]

    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill().shift(2)
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill().shift(2)

    pos_15_lev = (pos_base * 1.5).clip(0, 1.8)
    pos_15 = compute_risk_overlay(close=close, position=pos_15_lev, vix=vix, gvz=gvz,
                                  hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03,
                                  vol_cap=0.20, cooldown_days=5)
    for i in range(1, len(pos_15)):
        if abs(pos_15.iloc[i] - pos_15.iloc[i-1]) < 0.05:
            pos_15.iloc[i] = pos_15.iloc[i-1]

    bh_pos = pd.Series(1.0, index=close.index)

    # ===== 2. 净值计算 (无前视O2O) =====
    print("[2/6] 计算无前视O2O净值...")

    def nav_o2o(pos, s="2010-01-01", e="2026-08-27"):
        sub = pos[(pos.index >= s) & (pos.index <= e)]
        o = open_price.reindex(sub.index)
        ret_fwd = o.pct_change().shift(-1)
        sr = (sub.shift(1) * ret_fwd).dropna()
        cost = sub.diff().abs() * COST_BPS / 10000
        sr = (sr - cost).dropna()
        return (1 + sr).cumprod(), sr

    nav_base, sr_base = nav_o2o(pos_base)
    nav_15, sr_15 = nav_o2o(pos_15)
    nav_bh, sr_bh = nav_o2o(bh_pos)

    def get_metrics(nav):
        sr = nav.pct_change().dropna()
        ar = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
        vol = sr.std() * np.sqrt(252)
        dd = (nav / nav.cummax() - 1).min()
        return {"cagr": ar, "vol": vol, "sharpe": ar / vol if vol > 0 else 0,
                "mdd": dd, "calmar": ar / abs(dd) if dd != 0 else 0,
                "total": nav.iloc[-1] / nav.iloc[0] - 1}

    metrics_15 = get_metrics(nav_15)
    metrics_base = get_metrics(nav_base)
    metrics_bh = get_metrics(nav_bh)

    # ===== 3. Walk-Forward拼接 (每年滚动, 仅测试段) =====
    print("[3/6] Walk-Forward 15折拼接...")
    start_y = close.index[0].year + 4
    end_y = close.index[-1].year
    wf_rets = []
    wf_spans = []
    y = start_y
    while y <= end_y:
        test_end = close.index[-1] if y == end_y else pd.Timestamp(f"{y}-12-31")
        test_start = pd.Timestamp(f"{y}-01-01")
        sub = pos_15[(pos_15.index >= test_start) & (pos_15.index <= test_end)]
        o = open_price.reindex(sub.index)
        ret_fwd = o.pct_change().shift(-1)
        sr = (sub.shift(1) * ret_fwd).dropna()
        cost = sub.diff().abs() * COST_BPS / 10000
        net = (sr - cost).dropna()
        if len(net) > 0:
            wf_rets.append(net)
            wf_spans.append((net.index[0], net.index[-1]))
        y += 1
    wf_combined = pd.concat(wf_rets).sort_index()
    wf_combined = wf_combined[~wf_combined.index.duplicated(keep='first')]
    wf_nav = (1 + wf_combined).cumprod()

    # ===== 4. 2026纯OOS =====
    print("[4/6] 2026纯OOS数据...")
    m26 = pos_15["2026-01-01":"2026-08-27"]
    o26 = open_price["2026-01-01":"2026-08-27"]
    c26 = close["2026-01-01":"2026-08-27"]
    ret_fwd = o26.pct_change().shift(-1)
    sr26 = (m26.shift(1) * ret_fwd).dropna()
    cost26 = m26.diff().abs() * COST_BPS / 10000
    sr26 = (sr26 - cost26).dropna()
    nav26 = (1 + sr26).cumprod()

    # ===== 5. 逐年指标表 =====
    print("[5/6] 逐年指标表 (区间收益, 非年化)...")
    yearly_rows = []
    for year in range(2010, 2027):
        row = {"year": year}
        for name, pos in [("L15", pos_15), ("BH", bh_pos)]:
            sub = pos[pos.index.year == year]
            c = close.reindex(sub.index)
            o = open_price.reindex(sub.index)
            if len(sub) < 20:
                for k in ['ret', 'vol', 'mdd', 'sharpe', 'calmar']:
                    row[f"{name}_{k}"] = np.nan
                continue
            ret_fwd = o.pct_change().shift(-1)
            sr = (sub.shift(1) * ret_fwd).dropna()
            cost = sub.diff().abs() * COST_BPS / 10000
            sr = (sr - cost).dropna()
            if len(sr) < 10:
                for k in ['ret', 'vol', 'mdd', 'sharpe', 'calmar']:
                    row[f"{name}_{k}"] = np.nan
                continue
            nav = (1 + sr).cumprod()
            ret = nav.iloc[-1] - 1  # 区间收益 (非年化)
            vol = sr.std() * np.sqrt(252)
            dd = (nav / nav.cummax() - 1).min()
            n_y = len(sr) / 252
            ann = (nav.iloc[-1]) ** (1 / n_y) - 1
            row[f"{name}_ret"] = ret
            row[f"{name}_vol"] = vol
            row[f"{name}_mdd"] = dd
            row[f"{name}_sharpe"] = ann / vol if vol > 0 else 0
            row[f"{name}_calmar"] = ann / abs(dd) if dd != 0 else 0
        yearly_rows.append(row)
    yearly_df = pd.DataFrame(yearly_rows).set_index("year")

    # ===== 6. 大跌场景窗口 =====
    print("[6/6] 大跌场景窗口...")
    scenarios = {
        "2013 Taper紧缩": ("2013-02-15", "2013-08-15"),
        "2020 COVID冲击": ("2020-02-01", "2020-06-30"),
        "2022 激进加息": ("2022-03-01", "2022-10-31"),
        "2026 崩盘": ("2026-01-15", "2026-04-30"),
    }
    scenario_data = {}
    for name, (s, e) in scenarios.items():
        c_win = close[s:e]
        o_win = open_price[s:e]
        p_win = pos_15[s:e]
        p_base_win = pos_base[s:e]
        scenario_data[name] = {
            "close": c_win, "open": o_win, "pos": p_win, "pos_base": p_base_win,
        }

    # ===== 保存缓存 =====
    data = {
        "font_cjk": "Noto Sans CJK JP",
        "close": close, "open_price": open_price,
        "pos_base": pos_base, "pos_15": pos_15,
        "nav_base": nav_base, "nav_15": nav_15, "nav_bh": nav_bh,
        "metrics_15": metrics_15, "metrics_base": metrics_base, "metrics_bh": metrics_bh,
        "wf_nav": wf_nav, "wf_spans": wf_spans,
        "nav26": nav26, "close26": c26, "pos26": m26, "open26": o26,
        "yearly": yearly_df,
        "scenarios": scenario_data,
        "generated_at": pd.Timestamp.now(),
    }
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✓ 数据缓存已保存: {CACHE}")
    print(f"  数据截至: {close.index[-1].date()}")
    print(f"  L=1.5 CAGR={metrics_15['cagr']*100:.2f}% MDD={metrics_15['mdd']*100:.2f}%")
    print(f"  B&H  CAGR={metrics_bh['cagr']*100:.2f}% MDD={metrics_bh['mdd']*100:.2f}%")
    print(f"  WF拼接: {len(wf_spans)}折, 起止={wf_spans[0][0].date()}~{wf_spans[-1][1].date()}")


if __name__ == "__main__":
    main()

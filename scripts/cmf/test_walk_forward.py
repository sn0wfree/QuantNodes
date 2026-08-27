"""scripts/cmf/test_walk_forward.py — 生产参数 walk-forward 滚动验证.

方法:
    滚动窗口: train 4年 选参 → test 2年 评估, 测试窗连续拼接 2014-2026
    每折仅在train数据上选参数 (无未来信息泄漏, 所有因子均为后视滚动)
    选参目标: Calmar最大 (主) / Sharpe最大 (敏感性对照)

参数空间 (生产默认 ts=.06/r.7/gvz90/t18/p.8/rt w21):
    ts_out ∈ {0.05,0.06,0.08,0.10,0.12} × ratio ∈ {0.5,0.6,0.7}
    × gvz(lb∈{63,90} × t∈{16,18} × p∈{0.6,0.8}) × rate_w ∈ {21,42}
    = 240 组合/折

对照组 (同一拼接span): WF-Calmar选择 / WF-Sharpe选择 /
    固定生产默认参数 / 旧策略基线 ts=0.08
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

from QuantNodes.strategy.commodity_fundamental.gold import compute_breakout_signal
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    _load_indicator_d,
    _rate_gate,
    _zscore_scale,
    compute_vix_momentum_signal,
)

DB = "data/cmf/fundamental.duckdb"
COST_BPS = 5.0

FOLDS = [
    ("F1", "2010-01-01", "2013-12-31", "2014-01-01", "2015-12-31"),
    ("F2", "2012-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("F3", "2014-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("F4", "2016-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("F5", "2018-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("F6", "2020-01-01", "2023-12-31", "2024-01-01", "2026-08-25"),
]


def load_all():
    import duckdb

    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = (
        df.drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")["close"]
    )
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()
    tips = _load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()
    return close, vix, gvz, tips


def hysteresis(close, ts_out, ratio):
    ts_in = ratio * ts_out
    pos = pd.Series(np.nan, index=close.index)
    peak = close.iloc[0]
    state = 1.0
    vals = close.values
    for i in range(len(close)):
        c = vals[i]
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


def metrics(pos, close, s, e):
    sub = pos[(pos.index >= s) & (pos.index <= e)]
    c = close.reindex(sub.index)
    if len(c) < 60:
        return {"ann_ret": np.nan, "sharpe": np.nan, "max_dd": np.nan, "calmar": np.nan}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
    sr = (sr - cost).dropna()
    if len(sr) == 0:
        return {"ann_ret": np.nan, "sharpe": np.nan, "max_dd": np.nan, "calmar": np.nan}
    sp = c.iloc[0]
    nav = sp * (1 + sr).cumprod()
    n_y = len(c) / 252
    ar = (nav.iloc[-1] / sp) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0}


def main():
    close, vix, gvz, tips = load_all()

    print("预计算组件...")
    vmom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    vix_gate = (vix.rolling(252).rank(pct=True) > 0.7).astype(float)

    # 基线旧趋势组件 (旧ts=0.08 trailing stop语义)
    # 旧 ts=0.08 trailing stop (与compute_trend_bnh_signal一致的语义)
    pos = pd.Series(np.nan, index=close.index)
    peak = close.iloc[0]
    vals = close.values
    st = 1.0
    for i in range(len(close)):
        c = vals[i]
        if c > peak:
            peak = c
        if c < peak * (1 - 0.08):
            st = 0.0
            peak = c
        else:
            st = 1.0
        pos.iloc[i] = st
    trend_old = pos
    baseline_pos = (trend_old * vmom + 0.3 * bo * vix_gate).clip(0, 1.2)

    # 参数网格
    grid_ts = [0.05, 0.06, 0.08, 0.10, 0.12]
    grid_ratio = [0.5, 0.6, 0.7]
    grid_gvz = [(63, 16, 0.6), (63, 16, 0.8), (63, 18, 0.6), (63, 18, 0.8),
                (90, 16, 0.6), (90, 16, 0.8), (90, 18, 0.6), (90, 18, 0.8)]
    grid_rw = [21, 42]

    trends = {(t, r): hysteresis(close, t, r) for t, r in product(grid_ts, grid_ratio)}
    raws = {k: (tr * vmom + 0.3 * bo * vix_gate).clip(0, 1.2)
            for k, tr in trends.items()}
    f_gvz = {(lb, t, p): _zscore_scale(gvz, lb, t, p) for lb, t, p in grid_gvz}
    f_rate = {w: _rate_gate(tips, w, 2.5, 0.15, 1.3) for w in grid_rw}

    DEFAULT = dict(ts=0.06, ratio=0.7, gvz=(90, 18, 0.8), rw=21)

    def combo_pos(params):
        return (raws[(params["ts"], params["ratio"])]
                * f_gvz[params["gvz"]]
                * f_rate[params["rw"]]).clip(0, 1.2)

    all_param_keys = list(product(grid_ts, grid_ratio, grid_gvz, grid_rw))

    results_by_metric = {}
    for metric_name in ["calmar", "sharpe"]:
        folds_rows = []
        stitched_rets = []
        for fname, ts_, te_, vs_, ve_ in FOLDS:
            best_key, best_v = None, -np.inf
            for key in all_param_keys:
                params = dict(zip(("ts", "ratio", "gvz", "rw"), key))
                m = metrics(combo_pos(params), close, ts_, te_)
                v = m[metric_name]
                if not np.isnan(v) and v > best_v:
                    best_v, best_key = v, key
            params = dict(zip(("ts", "ratio", "gvz", "rw"), best_key))
            pos_best = combo_pos(params)
            m_test = metrics(pos_best, close, vs_, ve_)
            m_fixed = metrics(combo_pos(DEFAULT), close, vs_, ve_)
            m_base = metrics(baseline_pos, close, vs_, ve_)
            folds_rows.append({
                "fold": fname, "sel": str(best_key),
                "test_ann": m_test["ann_ret"], "test_dd": m_test["max_dd"],
                "fixed_ann": m_fixed["ann_ret"], "fixed_dd": m_fixed["max_dd"],
                "base_ann": m_base["ann_ret"], "base_dd": m_base["max_dd"],
            })
            c_sub = close.reindex(pos_best.index)[vs_:ve_]
            sr = (pos_best.shift(1) * c_sub.pct_change()).dropna()
            sr = sr - pos_best.diff().abs() * COST_BPS / 10000
            stitched_rets.append(sr.dropna())
        wf_pos = pd.concat(stitched_rets)

        def stitch_metrics(rets, span_s, span_e):
            r = rets[(rets.index >= span_s) & (rets.index <= span_e)]
            nav = (1 + r).cumprod()
            n_y = len(r) / 252
            ar = nav.iloc[-1] ** (1 / n_y) - 1
            vol = r.std() * np.sqrt(252)
            dd = (nav / nav.cummax() - 1).min()
            return ar, (ar / vol if vol > 0 else 0), dd, (ar / abs(dd) if dd != 0 else 0)

        results_by_metric[metric_name] = {
            "folds": pd.DataFrame(folds_rows),
            "wf_ar": stitch_metrics(wf_pos, "2014-01-01", "2026-08-25"),
            "fixed": metrics(combo_pos(DEFAULT), close, "2014-01-01", "2026-08-25"),
            "base": metrics(baseline_pos, close, "2014-01-01", "2026-08-25"),
        }

    # ===== 输出 =====
    for metric_name in ["calmar", "sharpe"]:
        R = results_by_metric[metric_name]
        print("\n" + "=" * 118)
        print(f"Walk-Forward 逐折明细 (选参目标: {metric_name})")
        print("=" * 118)
        print(f'{"折":<4} {"测试期":<11} {"所选参数":<34} | '
              f'{"WF年化":>7} {"WF MaxDD":>9} | '
              f'{"固定年化":>8} {"固定DD":>8} | {"基线年化":>8} {"基线DD":>8}')
        print("-" * 118)
        for _, rrow in R["folds"].iterrows():
            te_span = FOLDS[[f[0] for f in FOLDS].index(rrow["fold"])][4][:4]
            print(f'{rrow["fold"]:<4} {te_span:<11} {rrow["sel"]:<34} | '
                  f'{rrow["test_ann"]*100:>6.2f}% {rrow["test_dd"]*100:>8.2f}% | '
                  f'{rrow["fixed_ann"]*100:>7.2f}% {rrow["fixed_dd"]*100:>7.2f}% | '
                  f'{rrow["base_ann"]*100:>7.2f}% {rrow["base_dd"]*100:>7.2f}%')

        print("\n--- Walk-Forward 拼接 OOS (2014-2026, 全部为逐折样本外) ---")
        ar, sh, dd, ca = R["wf_ar"]
        fx = R["fixed"]
        bs = R["base"]
        print(f'{"方案":<26} {"年化":>8} {"Sharpe":>7} {"MaxDD":>9} {"Calmar":>7}')
        print("-" * 70)
        print(f'{"WF-calmar选择" if metric_name=="calmar" else "WF-sharpe选择":<24} '
              f'{ar*100:>7.2f}% {sh:>7.2f} {dd*100:>8.2f}% {ca:>7.2f}')
        print(f'{"固定默认参数":<24} {fx["ann_ret"]*100:>7.2f}% {fx["sharpe"]:>7.2f} '
              f'{fx["max_dd"]*100:>8.2f}% {fx["calmar"]:>7.2f}')
        print(f'{"旧基线 ts=0.08":<24} {bs["ann_ret"]*100:>7.2f}% {bs["sharpe"]:>7.2f} '
              f'{bs["max_dd"]*100:>8.2f}% {bs["calmar"]:>7.2f}')

    # 参数稳定性
    print("\n" + "=" * 80)
    print("参数稳定性 (两准则所选key)")
    sel_counts = {}
    for mn in ["calmar", "sharpe"]:
        for s in results_by_metric[mn]["folds"]["sel"]:
            sel_counts[s] = sel_counts.get(s, 0) + 1
    for k, v in sorted(sel_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}/12 次")


if __name__ == "__main__":
    main()

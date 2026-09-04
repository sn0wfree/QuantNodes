# coding=utf-8
"""cta_cu_bias_audit.py — CU溢价TC 因子选择偏差审计 (P1+P2).

审计对象: cta_rebuild_portfolio.py build_cu 的因子纳入逻辑
    原版:  cons["agree"] — IS与OOS五分位价差方向一致才纳入 (OOS数据参与选择 = 数据窥探)
    修正:  cons["is_significant"] — 仅IS显著即纳入 (纯IS信息)

步骤:
    P1: 4因子两种门槛的入选情况 + close口径NAV 全样本/IS/OOS 对比
    P2: 修正版CU替换后重跑 R2/E1/E2 组合, 量化偏差传染

输出: reports/cta/cu_bias_audit.csv + reports/cta/cu_bias_navs.parquet
"""
from __future__ import annotations

import sys
import warnings

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.factor_research import (  # noqa: E402
    is_oos_direction_consistency, rolling_zscore, vol_target_backtest,
)
from cta_shared.portfolio_tools import (  # noqa: E402
    DEFAULT_LEV_COST, dynamic_leverage, levered_portfolio, multi_window_factors,
    turnover_cost,
)

warnings.filterwarnings("ignore")

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
CMF_DB = "data/cmf/fundamental.duckdb"
IS_END = "2022-01-01"
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0
W_MAIN = 0.55
REPORT_DIR = "reports/cta"

CON = duckdb.connect(CTA_DB, read_only=True)
CON2 = duckdb.connect(CMF_DB, read_only=True)


def load_smoothed_full(tbl: str) -> pd.DataFrame:
    df = CON.execute(
        f"SELECT trade_date, open, close, open_interest FROM smoothed_{tbl}_daily "
        f"WHERE close IS NOT NULL AND open IS NOT NULL ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.drop_duplicates("trade_date", keep="last").set_index("trade_date")


def build_cu_signals(gate: str) -> tuple[pd.DataFrame, dict]:
    """build_cu 复刻, gate: 'agree'(原版) | 'is_only'(修正). 返回(信号, 选择详情)."""
    prem = CON2.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name='洋山铜溢价' ORDER BY trade_date").df()
    tc = CON2.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name LIKE '%粗炼费%' ORDER BY trade_date").df()
    for d in (prem, tc):
        d["trade_date"] = pd.to_datetime(d["trade_date"])
    prem = prem.set_index("trade_date")["value"].sort_index()
    tc = tc.set_index("trade_date")["value"].sort_index()

    df = load_smoothed_full("cu")
    close = df["close"]
    cn_days = close.index
    prem_d = prem.reindex(cn_days.union(prem.index)).ffill().reindex(cn_days)
    tc_d = tc.reindex(cn_days.union(tc.index)).ffill().reindex(cn_days)
    factors = {
        "P1": rolling_zscore(prem_d, 252),
        "P2": rolling_zscore(prem_d.pct_change(20), 252),
        "T1": rolling_zscore(tc_d, 252),
        "T2": rolling_zscore(tc_d.pct_change(20), 252),
    }
    factors = {k: v.shift(1) for k, v in factors.items()}  # PIT
    fwd5 = close.pct_change(5, fill_method=None).shift(-5)
    sig = pd.Series(0.0, index=cn_days)
    details = {}
    for name, f in factors.items():
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END, min_is_spread=0.05)
        if gate == "agree":
            use = bool(cons and cons["agree"] and abs(cons["is_spread"]) > 0.05)
        else:
            use = bool(cons and cons["is_significant"])
        details[name] = {**(cons or {}), "use": use}
        if use:
            sig += (np.sign(f) * np.sign(cons["is_spread"])
                    * (f.abs() > 1.0).astype(float)).fillna(0)
    return np.sign(sig).to_frame("CU"), details


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 50:
        return {"sharpe": np.nan, "oos": np.nan, "is": np.nan}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    is_r = r[r.index < IS_END]
    oos_r = r[r.index >= IS_END]
    def _s(x):
        return x.mean() / x.std() * np.sqrt(252) if len(x) > 50 and x.std() > 0 else np.nan
    return {"sharpe": ann / vol if vol > 0 else np.nan,
            "is": _s(is_r), "oos": _s(oos_r)}


def cu_nav(gate: str) -> pd.Series:
    sig, _ = build_cu_signals(gate)
    close = load_smoothed_full("cu")[["close"]].rename(columns={"close": "CU"})
    res = vol_target_backtest(sig, close, cost_bps={"CU": 3.0}, target_vol=0.08, pit_shift=0)
    return res["nav"].pct_change(fill_method=None).fillna(0)


def combo_net(main_ret: pd.Series, sub_rets: dict[str, pd.Series],
              sub_w: dict[str, float]) -> pd.Series:
    weights = {"主CTA": W_MAIN}
    for k, v in sub_w.items():
        weights[k] = v * (1 - W_MAIN) if v <= 1 else v / 100  # 兼容两种口径
    rets = pd.DataFrame({"主CTA": main_ret, **sub_rets})
    rets = rets.reindex(rets.dropna().index.union(rets.index)).fillna(0)
    common = rets.dropna().index
    rets = rets.loc[common]
    fac_A = multi_window_factors(rets, skip_cols=("主CTA",), windows=(20, 40, 60), cut=0.5)
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    wf = pd.DataFrame({col: weights.get(col, 0) * fac_A[col] for col in rets.columns})
    gross = levered_portfolio(rets * fac_A, weights, lev_series=lev,
                              cost_annual=DEFAULT_LEV_COST)
    tc = turnover_cost(wf, cost_bps=COST_BPS, w_main=W_MAIN, lev_series=lev)
    return (gross - tc).dropna()


def main() -> None:
    print("=" * 100)
    print("P1: CU因子选择偏差 — 原版agree门槛 vs 修正纯IS门槛")
    print("=" * 100)

    navs = {}
    rows = []
    for gate, label in [("agree", "原版agree"), ("is_only", "修正ISonly")]:
        sig, det = build_cu_signals(gate)
        n_use = sum(d["use"] for d in det.values())
        print(f"\n[{label}] 纳入 {n_use}/4 因子")
        for name, d in det.items():
            print(f"  {name}: is_spread={d.get('is_spread', np.nan):+.3f} "
                  f"oos_spread={d.get('oos_spread', np.nan):+.3f} "
                  f"agree={d.get('agree')} use={d['use']}")
            if gate == "agree":
                rows.append({"factor": name, "is_spread": round(d.get("is_spread", np.nan), 4),
                             "oos_spread": round(d.get("oos_spread", np.nan), 4),
                             "agree": d.get("agree"), "use_agree": d["use"]})
        navs[gate] = cu_nav(gate)
        st = stats(navs[gate])
        print(f"  → CU NAV [{label}]: 全样本 {st['sharpe']:.3f} | "
              f"IS {st['is']:.3f} | OOS {st['oos']:.3f}")
        rows.append({"factor": f"NAV_{label}", "is_spread": np.nan, "oos_spread": np.nan,
                     "agree": None, "use_agree": None})
        # 追加Sharpe列到该行
        rows[-1].update({"sharpe_full": round(st["sharpe"], 3),
                         "sharpe_is": round(st["is"], 3), "sharpe_oos": round(st["oos"], 3)})

    d_full = stats(navs["agree"])
    d_fix = stats(navs["is_only"])
    print("\n" + "=" * 100)
    print(f"P1 结论: CU全样本 {d_full['sharpe']:.3f}→{d_fix['sharpe']:.3f} "
          f"(Δ{d_fix['sharpe']-d_full['sharpe']:+.3f}) | "
          f"OOS {d_full['oos']:.3f}→{d_fix['oos']:.3f} "
          f"(Δ{d_fix['oos']-d_full['oos']:+.3f})")

    # P2: 修正版CU重跑组合
    print("\n" + "=" * 100)
    print("P2: 修正版CU替换后 R2/E1/E2 组合传染")
    print("=" * 100)
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_golden = pd.read_parquet(golden_path)["nav"]
    main_golden = main_golden[main_golden.index <= "2025-12-31"]
    main_ret = main_golden.pct_change(fill_method=None).fillna(0)

    sub = {}
    for name, f in [("T10y", "rebuild_subnav_T10y_open.csv"),
                    ("jjm_coke", "rebuild_subnav_jjm_coke_open.csv"),
                    ("ZN库存", "rebuild_subnav_ZN库存_open.csv")]:
        nav = pd.read_csv(f"{REPORT_DIR}/{f}", index_col=0, parse_dates=True).iloc[:, 0]
        nav = nav[nav.index <= "2025-12-31"]
        sub[name] = nav.pct_change(fill_method=None).fillna(0)

    PORTS = {
        "R2复刻": {"CU溢价TC": 25, "T10y": 10, "jjm_coke": 10},
        "E1扩容(50/50)": {"CU溢价TC": 27.78, "T10y": 11.11, "jjm_coke": 11.11},
        "E2扩容(45/55)": {"CU溢价TC": 30.56, "T10y": 12.22, "jjm_coke": 12.22},
    }
    w_main_map = {"R2复刻": 0.55, "E1扩容(50/50)": 0.50, "E2扩容(45/55)": 0.45}

    comp_rows = []
    for port, w_map in PORTS.items():
        wm = w_main_map[port]
        for ver, cu_r in [("原版CU", navs["agree"]), ("修正CU", navs["is_only"])]:
            subs = {**{k: v for k, v in sub.items()},
                    "CU溢价TC": cu_r}
            # 临时改W_MAIN: combo_net用全局W_MAIN, 传入前调整
            global W_MAIN
            W_MAIN = wm
            net = combo_net(main_ret, subs, w_map)
            W_MAIN = 0.55
            st = stats(net)
            tag = f"{port} [{ver}]"
            print(f"  {tag:28s} 全样本 {st['sharpe']:.3f} | OOS {st['oos']:.3f}")
            comp_rows.append({"portfolio": port, "version": ver,
                              "sharpe_full": round(st["sharpe"], 3),
                              "sharpe_oos": round(st["oos"], 3)})
        a = [r for r in comp_rows if r["portfolio"] == port and r["version"] == "原版CU"][0]
        b = [r for r in comp_rows if r["portfolio"] == port and r["version"] == "修正CU"][0]
        print(f"    → Δ全样本 {b['sharpe_full']-a['sharpe_full']:+.3f}, "
              f"ΔOOS {b['sharpe_oos']-a['sharpe_oos']:+.3f}")

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/cu_bias_audit.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(f"{REPORT_DIR}/cu_bias_combo.csv", index=False)
    pd.DataFrame({"agree_原版": navs["agree"], "is_only_修正": navs["is_only"]}).to_parquet(
        f"{REPORT_DIR}/cu_bias_navs.parquet")
    print("\n落盘: cu_bias_audit.csv / cu_bias_combo.csv / cu_bias_navs.parquet")


if __name__ == "__main__":
    main()

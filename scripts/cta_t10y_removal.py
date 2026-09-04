# coding=utf-8
"""cta_t10y_removal.py — 第0步: T10y剔除基线 (审计§21刀尖结论的落地).

T10y 因 oi5 刀尖效应被剔除, 两种机械处置:
    放大版:   CU/jjm 等比放大补齐子层 (维持子层总权重)
    还主CTA:  T10y 权重退还主CTA (子层其余不动)

对照: 原 R2 (含T10y) 基线. 决策规则同 §20.
输出: reports/cta/t10y_removal.csv
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.portfolio_tools import (  # noqa: E402
    DEFAULT_LEV_COST, dynamic_leverage, levered_portfolio, multi_window_factors,
    turnover_cost,
)

REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0
SEGMENTS = [("IS 2019-2021", "2019-01-01", "2022-01-01"),
            ("2022-2023", "2022-01-01", "2024-01-01"),
            ("2024", "2024-01-01", "2025-01-01"),
            ("2025", "2025-01-01", "2026-01-01"),
            ("OOS 2022-25", "2022-01-01", "2026-01-01"),
            ("全样本", None, None)]


def load_rets() -> pd.DataFrame:
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_nav = pd.read_parquet(golden_path)["nav"]
    rets = {"主CTA": main_nav.pct_change(fill_method=None).fillna(0)}
    for name, f in [("CU溢价TC", "rebuild_subnav_CU溢价TC_open.csv"),
                    ("T10y", "rebuild_subnav_T10y_open.csv"),
                    ("jjm_coke", "rebuild_subnav_jjm_coke_open.csv"),
                    ("ZN库存", "rebuild_subnav_ZN库存_open.csv")]:
        nav = pd.read_csv(f"{REPORT_DIR}/{f}", index_col=0, parse_dates=True).iloc[:, 0]
        nav = nav[nav.index <= "2025-12-31"]
        rets[name] = nav.pct_change(fill_method=None).fillna(0)
    df = pd.DataFrame(rets)
    return df.reindex(df.dropna(how="all").index).fillna(0)


def run_portfolio(rets: pd.DataFrame, weights_pct: dict) -> pd.Series:
    """weights_pct: 子策略为绝对百分比, 主CTA为小数 — 内部统一转小数."""
    weights = {"主CTA": weights_pct["主CTA"]}
    for k, v in weights_pct.items():
        if k != "主CTA":
            weights[k] = v / 100
    fac_A = multi_window_factors(rets, skip_cols=("主CTA",), windows=(20, 40, 60), cut=0.5)
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    wf = pd.DataFrame({c: weights.get(c, 0) * fac_A[c] for c in rets.columns})
    gross = levered_portfolio(rets * fac_A, weights, lev_series=lev,
                              cost_annual=DEFAULT_LEV_COST)
    tc = turnover_cost(wf, cost_bps=COST_BPS, w_main=weights["主CTA"], lev_series=lev)
    return (gross - tc).dropna()


def seg_metrics(net: pd.Series, lo: str | None, hi: str | None) -> dict:
    r = net.dropna()
    if lo:
        r = r[r.index >= lo]
    if hi:
        r = r[r.index < hi]
    if len(r) < 50:
        return {"sharpe": float("nan"), "ann_ret": float("nan"), "max_dd": float("nan")}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * (252 ** 0.5)
    dd = (nav / nav.cummax() - 1).min()
    return {"sharpe": ann / vol if vol > 0 else float("nan"),
            "ann_ret": ann, "max_dd": dd}


# (标签, 权重) — 子策略为绝对百分比
PORTS = {
    "R2原(对照)":     ({"主CTA": 0.55, "CU溢价TC": 25, "T10y": 10, "jjm_coke": 10}),
    "R2-R放大":       ({"主CTA": 0.55, "CU溢价TC": 32.14, "jjm_coke": 12.86}),
    "R2-C还主CTA":    ({"主CTA": 0.65, "CU溢价TC": 25, "jjm_coke": 10}),
    "E1原(对照)":     ({"主CTA": 0.50, "CU溢价TC": 27.78, "T10y": 11.11, "jjm_coke": 11.11}),
    "E1-R放大":       ({"主CTA": 0.50, "CU溢价TC": 35.71, "jjm_coke": 14.29}),
    "E1-C还主CTA":    ({"主CTA": 0.6111, "CU溢价TC": 27.78, "jjm_coke": 11.11}),
    "E2原(对照)":     ({"主CTA": 0.45, "CU溢价TC": 30.56, "T10y": 12.22, "jjm_coke": 12.22}),
    "E2-R放大":       ({"主CTA": 0.45, "CU溢价TC": 39.31, "jjm_coke": 15.69}),
    "E2-C还主CTA":    ({"主CTA": 0.5722, "CU溢价TC": 30.56, "jjm_coke": 12.22}),
}


def main() -> None:
    rets = load_rets()
    nets = {}
    rows = []
    for label, w in PORTS.items():
        net = run_portfolio(rets, w)
        nets[label] = net
        line = f"{label:14s}"
        vals = {}
        for name, lo, hi in SEGMENTS:
            m = seg_metrics(net, lo, hi)
            vals[name] = m
            line += f" {name.split()[0]}:{m['sharpe']:.3f}"
        m_all = vals["全样本"]
        print(f"{line}  | AR {m_all['ann_ret']:+.2%} DD {m_all['max_dd']:+.2%}")
        rows.append({"variant": label,
                     **{f"{n}_shp": round(vals[n]["sharpe"], 3) for n, _, _ in SEGMENTS},
                     "ann_ret": round(m_all["ann_ret"], 4),
                     "max_dd": round(m_all["max_dd"], 4)})

    print("\n=== 对照判定 (vs 各自含T10y原版) ===")
    for base_l, variants in [("R2原(对照)", ["R2-R放大", "R2-C还主CTA"]),
                              ("E1原(对照)", ["E1-R放大", "E1-C还主CTA"]),
                              ("E2原(对照)", ["E2-R放大", "E2-C还主CTA"])]:
        b = nets[base_l]
        b_all = seg_metrics(b, None, None)
        b_oos = seg_metrics(b, "2022-01-01", "2026-01-01")
        for v in variants:
            m = nets[v]
            m_all = seg_metrics(m, None, None)
            m_oos = seg_metrics(m, "2022-01-01", "2026-01-01")
            ok = (m_all["sharpe"] >= b_all["sharpe"] * 0.98
                  and m_oos["sharpe"] >= b_oos["sharpe"] * 0.98
                  and m_all["max_dd"] >= b_all["max_dd"] - 0.001)
            print(f"  {v:14s} Δ全样本{m_all['sharpe']-b_all['sharpe']:+.3f} "
                  f"ΔOOS{m_oos['sharpe']-b_oos['sharpe']:+.3f} "
                  f"DD {m_all['max_dd']:+.2%} → {'PASS' if ok else 'FAIL'}")

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/t10y_removal.csv", index=False)
    for label, net in nets.items():
        safe = label.replace("(", "_").replace(")", "")
        net.to_frame("net_ret").to_parquet(f"{REPORT_DIR}/t10y_free_{safe}.parquet")
    print("\n落盘: t10y_removal.csv + t10y_free_*.parquet")


if __name__ == "__main__":
    main()

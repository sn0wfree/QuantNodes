# coding=utf-8
"""cta_r2_scale_test.py — 子层扩容实验: 主CTA占比 55%→50%/45%.

逻辑: 2024 子策略层全正 (CU+15.1/jjm+3.7/ZN+7.4) 而主CTA剔EC归零 —
加大已验证有效层的权重, 直接对症弱年. 风险: 强年稀释 (子层Sharpe 0.75~1.0 < 主CTA 2.31).

方案:
    R2复刻(55/45)      基线对账 (≈并行会话 2.948)
    R3b复刻(55/45)     并行会话最优对照 (≈2.995)
    E1 扩容(50/50)      子层等比放大
    E2 扩容(45/55)      子层等比放大(更激进)
    E3 R3b+扩容(50/50)  R3b子层 + 扩容叠加

组合机制与 rebuild_portfolio.py 逐行对齐 (渐进A 20x40x60 + 动态杠杆[1.0,1.5] + 5bp).
主CTA: golden (HEAD口径); 子策略: rebuild_subnav_*_open.csv (PIT修正).

决策规则 (预设): 通过 = 2024净Sharpe改善>0.1 AND 全样本净Sharpe恶化<2%
                 AND MaxDD不恶化(±0.1pp) AND OOS不恶化

输出: reports/cta/r2_scale_test.csv + 年度分解
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.portfolio_tools import (  # noqa: E402
    DEFAULT_LEV_COST, dynamic_leverage, levered_portfolio, multi_window_factors,
    turnover_cost,
)

REPORT_DIR = Path("reports/cta")
GOLDEN = REPORT_DIR / "overfit_final_navs" / "基线_严格OOS0.5,60_2018.parquet"
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0
IS_END = "2022-01-01"

SUB_FILES = {
    "CU溢价TC": "rebuild_subnav_CU溢价TC_open.csv",
    "T10y": "rebuild_subnav_T10y_open.csv",
    "jjm_coke": "rebuild_subnav_jjm_coke_open.csv",
    "ZN库存": "rebuild_subnav_ZN库存_open.csv",
}

# (标签, w_main, {子策略: 子层内百分比})
PORTFOLIOS = {
    "R2复刻(55/45)":      (0.55, {"CU溢价TC": 25, "T10y": 10, "jjm_coke": 10}),
    "R3b复刻(55/45)":     (0.55, {"CU溢价TC": 25, "T10y": 5, "jjm_coke": 10, "ZN库存": 5}),
    "E1扩容(50/50)":      (0.50, {"CU溢价TC": 27.78, "T10y": 11.11, "jjm_coke": 11.11}),
    "E2扩容(45/55)":      (0.45, {"CU溢价TC": 30.56, "T10y": 12.22, "jjm_coke": 12.22}),
    "E3 R3b扩容(50/50)":  (0.50, {"CU溢价TC": 27.78, "T10y": 5.56, "jjm_coke": 11.11,
                                  "ZN库存": 5.56}),
}
SEGMENTS = [("IS 2019-2021", "2019-01-01", "2022-01-01"),
            ("2022-2023", "2022-01-01", "2024-01-01"),
            ("2024", "2024-01-01", "2025-01-01"),
            ("2025", "2025-01-01", "2026-01-01"),
            ("OOS 2022-25", "2022-01-01", "2026-01-01"),
            ("全样本", None, None)]


def load_rets() -> pd.DataFrame:
    main_nav = pd.read_parquet(GOLDEN)["nav"]
    rets = {"主CTA": main_nav.pct_change(fill_method=None).fillna(0)}
    for name, f in SUB_FILES.items():
        nav = pd.read_csv(REPORT_DIR / f, index_col=0, parse_dates=True).iloc[:, 0]
        nav = nav[nav.index <= "2025-12-31"]
        rets[name] = nav.pct_change(fill_method=None).fillna(0)
    df = pd.DataFrame(rets)
    return df.reindex(df.dropna(how="all").index).fillna(0)


def run_portfolio(rets: pd.DataFrame, w_main: float,
                  sub_pct: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """与 rebuild_portfolio.combo_run 逐行对齐 (gross, net).

    sub_pct 为绝对百分比 (合计 = 100 - w_main×100), 不再乘 (1-w_main).
    """
    weights = {"主CTA": w_main}
    for k, pct in sub_pct.items():
        weights[k] = pct / 100
    fac_A = multi_window_factors(rets, skip_cols=("主CTA",), windows=(20, 40, 60), cut=0.5)
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    wf = pd.DataFrame({col: weights.get(col, 0) * fac_A[col] for col in rets.columns})
    gross = levered_portfolio(rets * fac_A, weights, lev_series=lev,
                              cost_annual=DEFAULT_LEV_COST)
    tc = turnover_cost(wf, cost_bps=COST_BPS, w_main=w_main, lev_series=lev)
    return gross, gross - tc


def seg_metrics(net: pd.Series, lo: str | None, hi: str | None) -> dict:
    r = net.dropna()
    if lo:
        r = r[r.index >= lo]
    if hi:
        r = r[r.index < hi]
    if len(r) < 50:
        return {"sharpe": np.nan, "ann_ret": np.nan, "max_dd": np.nan}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return {"sharpe": ann / vol if vol > 0 else np.nan,
            "ann_ret": ann, "max_dd": dd}


def main() -> None:
    rets = load_rets()
    print(f"收益矩阵: {rets.shape}, {rets.index.min().date()}~{rets.index.max().date()}")

    nets = {}
    for label, (w_main, sub_pct) in PORTFOLIOS.items():
        _, net = run_portfolio(rets, w_main, sub_pct)
        nets[label] = net
        m = seg_metrics(net, None, None)
        print(f"  {label:22s} 净Shp {m['sharpe']:.3f} 净AR {m['ann_ret']:+.2%} "
              f"净DD {m['max_dd']:+.2%}", flush=True)

    rows = []
    for label, net in nets.items():
        for name, lo, hi in SEGMENTS:
            m = seg_metrics(net, lo, hi)
            rows.append({"variant": label, "segment": name,
                         "sharpe": round(m["sharpe"], 3),
                         "ann_ret": round(m["ann_ret"], 4),
                         "max_dd": round(m["max_dd"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "r2_scale_test.csv", index=False)

    base = nets["R2复刻(55/45)"]
    print("\n=== 分段净Sharpe ===")
    seg_lookup = {name: (lo, hi) for name, lo, hi in SEGMENTS}
    for name, _, _ in SEGMENTS:
        lo, hi = seg_lookup[name]
        line = f"{name:12s} " + "  ".join(
            f"{label.split('(')[0]}:{seg_metrics(net, lo, hi)['sharpe']:.3f}"
            for label, net in nets.items())
        print(line)

    # 年度收益 (2024/2025 重点)
    print("\n=== 年度净收益 ===")
    yearly = pd.DataFrame({label: net.groupby(net.index.year).apply(
        lambda x: (1 + x).prod() - 1) for label, net in nets.items()})
    print(yearly.round(4).to_string())
    yearly.to_csv(REPORT_DIR / "r2_scale_test_yearly.csv")

    # 预设决策规则
    print("\n=== 决策规则 (2024ΔShp>+0.1 AND 全样本恶化<2% AND DD不恶化 AND OOS不恶化) ===")
    for label, net in nets.items():
        if label == "R2复刻(55/45)":
            continue
        m24 = seg_metrics(net, "2024-01-01", "2025-01-01")
        b24 = seg_metrics(base, "2024-01-01", "2025-01-01")
        m_all = seg_metrics(net, None, None)
        b_all = seg_metrics(base, None, None)
        m_oos = seg_metrics(net, "2022-01-01", "2026-01-01")
        b_oos = seg_metrics(base, "2022-01-01", "2026-01-01")
        ok = (m24["sharpe"] - b24["sharpe"] > 0.1
              and m_all["sharpe"] >= b_all["sharpe"] * 0.98
              and m_all["max_dd"] >= b_all["max_dd"] - 0.001
              and m_oos["sharpe"] >= b_oos["sharpe"])
        print(f"  {label:22s} 2024Δ{m24['sharpe']-b24['sharpe']:+.3f} "
              f"全样本Δ{(m_all['sharpe']/b_all['sharpe']-1)*100:+.1f}% "
              f"DD {m_all['max_dd']:+.2%} (基线{b_all['max_dd']:+.2%}) "
              f"OOSΔ{m_oos['sharpe']-b_oos['sharpe']:+.3f} → "
              f"{'PASS' if ok else 'FAIL'}")

    # 全部方案 nav 落盘 (供叠加分析与文档引用)
    for label, net in nets.items():
        safe = label.replace("(", "_").replace("/", "_")
        net.to_frame("net_ret").to_parquet(REPORT_DIR / f"r2_scale_{safe}.parquet")
    print("\nnet 序列已落盘: reports/cta/r2_scale_*.parquet")


if __name__ == "__main__":
    main()

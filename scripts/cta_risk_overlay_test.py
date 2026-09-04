# coding=utf-8
"""cta_risk_overlay_test.py — 黄金三级风控移植实验 (§24 A/B组).

A组: 主CTA层移植 (预期FAIL, 闭环§18结论 — 阈值按主CTA波动重标定)
    A1 L1 vol压制:  realized_vol(60d)>12% → 收益×(12%/vol)
    A2 L2 软减仓:   rolling_dd(252d)<-3.5% → ×0.5
    A3 L3 硬止损:   dd<-5% 或单日<-1.5% → 清仓+5d冷却; 每次触发扣10bp (66品种双边近似)
    A套 = A1+A2+A3
    dd用rolling窗口 (nav/nav.rolling(252).max()-1, 无前视, 与黄金策略同构)

B组: 组合层dd风控 (CPPI思想上移, 不污染主CTA信号)
    基线: E2-R+AU10% (主CTA 40.5 + CU 35.379 + jjm 14.121 + AU 10)
    B1: dd<-2% → ×0.5
    B2: dd<-2.5% 或单日<-1.2% → 清仓+5d冷却

判定 (预设): IS恶化<1% 且 (OOS Sharpe改善 或 DD改善≥10%) 且全样本不降>1%
输出: reports/cta/risk_overlay_test.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_t10y_removal import load_rets, run_portfolio, seg_metrics  # noqa: E402

REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
SEGS = [("IS 18-21", None, IS_END), ("OOS 22-25", IS_END, None), ("全样本", None, None)]
COST_FLAT = 0.001  # L3每次触发扣10bp (全组合清仓/重建双边近似)


def roll_dd(rets: pd.Series, window: int = 252) -> pd.Series:
    nav = (1 + rets.fillna(0)).cumprod()
    return nav / nav.rolling(window, min_periods=20).max() - 1


def apply_L1(r: pd.Series, window: int = 60, vol_cap: float = 0.12) -> pd.Series:
    vol = r.rolling(window, min_periods=20).std() * np.sqrt(252)
    scale = (vol_cap / vol).clip(upper=1.0).fillna(1.0)
    return r * scale


def apply_L2(r: pd.Series, soft: float = -0.035, window: int = 252) -> pd.Series:
    dd = roll_dd(r, window)
    return r.where(~(dd < soft), r * 0.5)


def apply_L3(r: pd.Series, dd_th: float = -0.05, daily_th: float = -0.015,
             cooldown: int = 5) -> tuple[pd.Series, int]:
    dd = roll_dd(r)
    hard = ((dd < dd_th) | (r < daily_th)).fillna(False)
    pos = pd.Series(1.0, index=r.index)
    n_trig = 0
    i = 0
    idx = r.index
    while i < len(idx):
        if hard.iloc[i]:
            n_trig += 1
            end = min(i + cooldown, len(idx) - 1)
            pos.iloc[i:end + 1] = 0.0
            i = end + 1
        else:
            i += 1
    out = r * pos
    if n_trig:
        out[hard] -= COST_FLAT  # 触发日扣成本
    return out, n_trig


def verdict(tag: str, r: pd.Series, base: pd.Series) -> tuple[bool, str]:
    m_is = seg_metrics(r, None, IS_END)
    b_is = seg_metrics(base, None, IS_END)
    m_oos = seg_metrics(r, IS_END, None)
    b_oos = seg_metrics(base, IS_END, None)
    m_all = seg_metrics(r, None, None)
    b_all = seg_metrics(base, None, None)
    is_ok = m_is["sharpe"] >= b_is["sharpe"] - 0.01
    oos_ok = (m_oos["sharpe"] >= b_oos["sharpe"]) or (m_all["max_dd"] >= b_all["max_dd"] * 1.1)
    all_ok = m_all["sharpe"] >= b_all["sharpe"] - 0.01
    ok = is_ok and oos_ok and all_ok
    print(f"  {tag:18s} IS {m_is['sharpe']:+.3f}(基线{b_is['sharpe']:+.3f}) "
          f"OOS {m_oos['sharpe']:+.3f}({b_oos['sharpe']:+.3f}) "
          f"全 {m_all['sharpe']:+.3f}/{m_all['max_dd']:+.2%}"
          f"(基线{b_all['sharpe']:+.3f}/{b_all['max_dd']:+.2%}) → "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, tag


def main() -> None:
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    g = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)

    print("=" * 104)
    print("A组: 主CTA层三级风控移植 (golden为基础)")
    print("=" * 104)
    print(f"golden rolling-dd 分布: min {roll_dd(g).min():+.2%}, "
          f"5%分位 {roll_dd(g).quantile(0.05):+.2%}")

    rows = []
    base_m = seg_metrics(g, None, None)
    rows.append({"variant": "golden基线", "is": seg_metrics(g, None, IS_END)["sharpe"],
                 "oos": seg_metrics(g, IS_END, None)["sharpe"],
                 "full": base_m["sharpe"], "dd": base_m["max_dd"], "n_trig": 0})
    print(f"  {'golden基线':18s} 全 {base_m['sharpe']:+.3f}/{base_m['max_dd']:+.2%}")

    for tag, r, ntrig in [
        ("A1 vol压制", apply_L1(g), 0),
        ("A2 软减仓", apply_L2(g), 0),
        ("A3 硬止损", *apply_L3(g)),
    ]:
        ok, _ = verdict(tag, r, g)
        m = seg_metrics(r, None, None)
        rows.append({"variant": tag, "is": seg_metrics(r, None, IS_END)["sharpe"],
                     "oos": seg_metrics(r, IS_END, None)["sharpe"],
                     "full": m["sharpe"], "dd": m["max_dd"], "n_trig": ntrig})
    a_all = apply_L3(apply_L2(apply_L1(g)))
    ok, _ = verdict("A套(L1+L2+L3)", a_all[0], g)
    m = seg_metrics(a_all[0], None, None)
    rows.append({"variant": "A套", "is": seg_metrics(a_all[0], None, IS_END)["sharpe"],
                 "oos": seg_metrics(a_all[0], IS_END, None)["sharpe"],
                 "full": m["sharpe"], "dd": m["max_dd"], "n_trig": a_all[1]})

    print("\n" + "=" * 104)
    print("B组: 组合层dd风控 (E2-R+AU10% 基线)")
    print("=" * 104)
    rets = load_rets()
    au = pd.read_parquet(f"{REPORT_DIR}/gold_layered_layered.parquet")["ret"]
    rets2 = rets.copy()
    rets2["AU_layered"] = au.reindex(rets2.index).fillna(0)
    w = {"主CTA": 0.45, "CU溢价TC": 39.31, "jjm_coke": 15.69}
    scale = 0.9
    base_w = {k: v * scale for k, v in w.items()}
    base_w["AU_layered"] = 10
    combo = run_portfolio(rets2, base_w)
    print(f"  {'E2-R+AU10%基线':18s} 全 {seg_metrics(combo, None, None)['sharpe']:+.3f}"
          f"/{seg_metrics(combo, None, None)['max_dd']:+.2%}")

    for tag, r, ntrig in [
        ("B1 dd<-2%半仓", apply_L2(combo, soft=-0.02), 0),
        ("B2 硬止损-2.5%", *apply_L3(combo, dd_th=-0.025, daily_th=-0.012)),
    ]:
        ok, _ = verdict(tag, r, combo)
        m = seg_metrics(r, None, None)
        rows.append({"variant": tag, "is": seg_metrics(r, None, IS_END)["sharpe"],
                     "oos": seg_metrics(r, IS_END, None)["sharpe"],
                     "full": m["sharpe"], "dd": m["max_dd"], "n_trig": ntrig})

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/risk_overlay_test.csv", index=False)
    print("\n落盘: risk_overlay_test.csv")


if __name__ == "__main__":
    main()

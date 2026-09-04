# coding=utf-8
"""cta_gold_layered_eval.py — AU策略方案B (layered ensemble) 完整评估.

背景: §22评估对象搞错 (评估了gold_combined周频简化版). 用户确认的正式方案 =
gold_ensemble.compute_layered_ensemble_signal: 4路信号×三级风控, 日频o2o.
历史对照 (2010-26): 裸B&H 0.569/DD-44.5% → layered 0.922/DD-21.5% (风控砍半DD).

步骤:
    1. layered默认参数信号 → 体系口径重跑 (smoothed_au open, vt8%, cost2bp, pit_shift=1)
    2. 四格对照: 裸B&H / B&H+vt / layered+vt × 全样本/IS/OOS (Sharpe+MaxDD+Calmar)
    3. 风控参数敏感性: hard_cap{-0.08,-0.10,-0.15} × vol_cap{0.12,0.15,0.20}
    4. 门槛: OOS Shp≥0.15 + IS/OOS同向 + DD改善≥10% + r(主CTA)<0.3
    5. PASS → E2-R组合接入 {5%,10%,15%}

输出: reports/cta/gold_layered_eval.csv + gold_layered_navs.parquet
"""
from __future__ import annotations

import sys
import warnings

import duckdb
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.factor_research import vol_target_backtest  # noqa: E402

warnings.filterwarnings("ignore")

from QuantNodes.strategy.commodity_fundamental.gold.gold_ensemble import (  # noqa: E402
    compute_layered_ensemble_signal,
)

REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
COST = 2.0
TV = 0.08
CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"


def load_au() -> pd.DataFrame:
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, open, close FROM smoothed_au_daily "
        "WHERE close IS NOT NULL AND open IS NOT NULL ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = (df.drop_duplicates("trade_date", keep="last")
          .set_index("trade_date")[["open", "close"]])
    return df[df.index <= "2025-12-31"]


def vt_nav(sig: pd.Series, au: pd.DataFrame) -> pd.Series:
    """体系口径: o2o执行 + vol_target (与子策略一致)."""
    sig_df = sig.reindex(au.index).fillna(0).to_frame("AU_L")
    ret_open = (au["open"].shift(-1) / au["open"] - 1)
    res = vol_target_backtest(sig_df, au[["close"]].rename(columns={"close": "AU_L"}),
                              cost_bps={"AU_L": COST}, target_vol=TV, pit_shift=1,
                              ret_df=ret_open.shift(1).to_frame("AU_L"))
    return res["nav"].pct_change(fill_method=None).fillna(0)


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 50:
        return {"shp": float("nan"), "ar": float("nan"), "dd": float("nan")}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * 252 ** 0.5
    dd = (nav / nav.cummax() - 1).min()
    return {"shp": ann / vol if vol > 0 else float("nan"),
            "ar": ann, "dd": dd}


def seg(r: pd.Series, lo: str | None, hi: str | None) -> pd.Series:
    r = r.dropna()
    if lo:
        r = r[r.index >= lo]
    if hi:
        r = r[r.index < hi]
    return r


SEGS = [("全样本", None, None), ("IS 18-21", None, IS_END),
        ("OOS 22-25", IS_END, None)]


def main() -> None:
    au = load_au()
    print("=" * 100)
    print("AU方案B (layered ensemble) 评估 — 体系口径 o2o + vt8%")
    print("=" * 100)

    # 1. 信号提取 (默认参数)
    sig_df = compute_layered_ensemble_signal()
    sig = sig_df["position"]
    print(f"信号: {sig.index.min().date()}~{sig.index.max().date()}, "
          f"非零占比 {((sig.fillna(0) != 0).mean()):.0%}, "
          f"均值 {sig.fillna(0).clip(0, 1.2).mean():.2f}")

    # 2. 三条NAV
    sig1 = pd.DataFrame({"AU": 1.0}, index=au.index)
    bh_raw = vol_target_backtest(sig1, au[["close"]].rename(columns={"close": "AU"}),
                                 cost_bps={"AU": COST}, target_vol=1.0, pit_shift=1,
                                 ret_df=((au["open"].shift(-1) / au["open"] - 1)
                                         .shift(1).to_frame("AU")))
    bh_raw_r = bh_raw["nav"].pct_change(fill_method=None).fillna(0)
    bh_vt = vol_target_backtest(sig1, au[["close"]].rename(columns={"close": "AU"}),
                                cost_bps={"AU": COST}, target_vol=TV, pit_shift=1,
                                ret_df=((au["open"].shift(-1) / au["open"] - 1)
                                        .shift(1).to_frame("AU")))
    bh_vt_r = bh_vt["nav"].pct_change(fill_method=None).fillna(0)
    lay_r = vt_nav(sig, au)

    print(f"\n{'对照':16s}", end="")
    for sname, _, _ in SEGS:
        print(f" | {sname:^22s}", end="")
    print("\n" + "-" * 100)
    rows = []
    for label, r in [("裸B&H(无风控)", bh_raw_r), ("B&H+vt8%", bh_vt_r),
                     ("layered+vt8%", lay_r)]:
        line = f"{label:16s}"
        rec = {"variant": label}
        for sname, lo, hi in SEGS:
            m = metrics(seg(r, lo, hi))
            line += f" | {m['shp']:+.2f}/{m['dd']:+.1%}     "
            rec[sname] = (round(m["shp"], 3), round(m["dd"], 4))
        print(line)
        rows.append(rec)
        r.to_frame("ret").to_parquet(
            f"{REPORT_DIR}/gold_layered_{label.split('(')[0].split('+')[0].strip()}.parquet")

    # 3. 门槛判定
    lay_all = metrics(lay_r)
    lay_is = metrics(seg(lay_r, None, IS_END))
    lay_oos = metrics(seg(lay_r, IS_END, None))
    metrics(seg(bh_vt_r, None, IS_END))
    metrics(seg(bh_vt_r, IS_END, None))
    bh_all = metrics(bh_vt_r)
    dd_impr = 1 - (lay_all["dd"] / bh_all["dd"])
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_r = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)
    common = lay_r.index.intersection(main_r.index)
    r_main = lay_r[common].corr(main_r[common])

    same_dir = (lay_is["shp"] > 0) == (lay_oos["shp"] > 0)
    ok = (lay_oos["shp"] >= 0.15 and same_dir and dd_impr >= 0.10
          and abs(r_main) < 0.3)
    print(f"\n门槛: OOS Shp {lay_oos['shp']:+.3f} (≥0.15) | IS/OOS同向 {same_dir} | "
          f"DD改善 {dd_impr:.0%} (≥10%) | r(主CTA) {r_main:+.3f} (<0.3)")
    print(f"→ {'PASS' if ok else 'FAIL'}")

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/gold_layered_eval.csv", index=False)

    # 4. 敏感性 (仅当PASS才有意义, 但无论如何跑 — 防刀尖证据)
    print("\n=== 风控参数敏感性 (hard_cap × vol_cap, 其余默认) ===")
    sens_rows = []
    for hc in (-0.08, -0.10, -0.15):
        for vc in (0.12, 0.15, 0.20):
            try:
                sg = compute_layered_ensemble_signal(hard_cap=hc, vol_cap=vc)
                r = vt_nav(sg["position"], au)
                m_all, m_oos = metrics(r), metrics(seg(r, IS_END, None))
                print(f"  hc={hc:+.2f} vc={vc:.2f}: 全样本 {m_all['shp']:+.3f}/"
                      f"{m_all['dd']:+.1%} | OOS {m_oos['shp']:+.3f}/{m_oos['dd']:+.1%}")
                sens_rows.append({"hard_cap": hc, "vol_cap": vc,
                                  "shp": round(m_all["shp"], 3),
                                  "dd": round(m_all["dd"], 4),
                                  "oos_shp": round(m_oos["shp"], 3)})
            except Exception as e:
                print(f"  hc={hc:+.2f} vc={vc:.2f}: ERROR {e}")
    pd.DataFrame(sens_rows).to_csv(f"{REPORT_DIR}/gold_layered_sens.csv", index=False)

    # 5. 组合接入 (仅PASS时)
    if ok:
        print("\n=== E2-R 组合接入测试 ===")
        sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")
        from cta_t10y_removal import load_rets, run_portfolio  # noqa: E402
        rets = load_rets()
        base_w = {"主CTA": 0.45, "CU溢价TC": 39.31, "jjm_coke": 15.69}
        base = run_portfolio(rets, base_w)
        for x in (5, 10, 15):
            # 黄金挤入: 主CTA与CU/jjm等比压缩 (维持结构)
            scale = (100 - x) / 100
            w = {k: v * scale for k, v in base_w.items()}
            w["AU_layered"] = x
            rets2 = rets.copy()
            rets2["AU_layered"] = lay_r.reindex(rets2.index).fillna(0)
            net = run_portfolio(rets2, w)
            m_all, m_is = metrics(net), metrics(seg(net, None, IS_END))
            m_base, m_base_is = metrics(base), metrics(seg(base, None, IS_END))
            dd_im = 1 - (m_all["dd"] / m_base["dd"])
            verdict = ("PASS" if (m_all["shp"] >= m_base["shp"] * 0.98
                                  and dd_im >= 0.10 and m_is["shp"] >= m_base_is["shp"] * 0.98)
                       else "FAIL")
            print(f"  +AU {x:2d}%: 全样本 {m_all['shp']:.3f} (基线{m_base['shp']:.3f}) "
                  f"DD {m_all['dd']:+.2%} (基线{m_base['dd']:+.2%}, 改善{dd_im:.0%}) "
                  f"IS {m_is['shp']:.3f} (基线IS {m_base_is['shp']:.3f}) → {verdict}")

    print("\n落盘: gold_layered_eval.csv / gold_layered_sens.csv / gold_layered_*.parquet")


if __name__ == "__main__":
    main()

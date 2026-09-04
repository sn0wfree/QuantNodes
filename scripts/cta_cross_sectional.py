# coding=utf-8
"""cta_cross_sectional.py — §26 横截面三策略: basis动量 / 截面动量 / VRP.

共同框架 (§26.3冻结): 排序top/bottom 1/3, ±1/n_active, 周度再平衡(W-FRI),
vol_target 8%, cost 3bp, pit_shift=1, 执行收益=smoothed主力close.
纪律 (§26.4): 先算毛价差, <3%/年直接归档零调参; 周换手≤40x.
门槛: OOS≥0.15 + IS/OOS同向 + r(主CTA)<0.4 + 分年无单年依赖.

S1 basis momentum: Δbasis 12周动量 (basis = carry原始值, term_structure rank1/2)
S2 横截面动量: 4/13/26/52周收益排序均值 (smoothed主力close)
S3 VRP横截面: vrp_percentile排序 (vrp_daily)
输出: reports/cta/cross_sectional.csv + 各nav parquet
"""
from __future__ import annotations

import sys
import warnings

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.factor_research import vol_target_backtest  # noqa: E402

warnings.filterwarnings("ignore")

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
COST = 3.0
TV = 0.08
TIER = 3
MIN_GROSS = 0.03   # 毛价差纪律线 3%/年
MAX_TURNOVER = 40  # 周度换手纪律线


def load_smoothed_panel(con) -> pd.DataFrame:
    """全品种 smoothed close 面板."""
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()
              if t[0].startswith("smoothed_")]
    frames = {}
    for t in tables:
        p = t.replace("smoothed_", "").replace("_daily", "").upper()
        d = con.execute(
            f"SELECT trade_date, close FROM {t} "
            f"WHERE close IS NOT NULL AND close > 0 ORDER BY trade_date").df()
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        s = d.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
        if len(s) > 750:
            frames[p] = s
    return pd.DataFrame(frames)


def build_signal(score: pd.DataFrame, weekly: bool = True) -> pd.DataFrame:
    """横截面排序 → ±1/n_active, 周度持有."""
    ranks = score.rank(axis=1, pct=True)
    sig = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    sig[ranks >= 1 - 1 / TIER] = 1.0
    sig[ranks <= 1 / TIER] = -1.0
    sig[score.isna()] = 0.0
    if weekly:
        anchors = score.index.to_series().resample("W-FRI").last().dropna()
        sig = sig.loc[sig.index.isin(anchors)].reindex(sig.index).ffill().fillna(0)
    n_active = (sig != 0).sum(axis=1).clip(lower=1)
    return sig.div(n_active, axis=0)


def run_bt(sig: pd.DataFrame, close: pd.DataFrame, label: str) -> dict:
    sig = sig.reindex(close.index).fillna(0)
    rets = close.pct_change(fill_method=None).reindex(sig.index)
    res = vol_target_backtest(sig, close.reindex(sig.index), cost_bps=COST,
                              target_vol=TV, pit_shift=1, ret_df=rets)
    r = res["nav"].pct_change(fill_method=None).fillna(0)

    # 毛价差 (多-空日均, 无成本)
    n = (sig != 0).sum(axis=1).clip(lower=1)
    long_leg = (sig.clip(lower=0) * rets).sum(axis=1) / n
    short_leg = (-sig.clip(upper=0) * rets).sum(axis=1) / n
    gross_ann = (long_leg - short_leg).mean() * 252

    def s(x):
        x = x.dropna()
        if len(x) < 50 or x.std() == 0:
            return np.nan
        return x.mean() / x.std() * np.sqrt(252)
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    to = res["turnover_annual"]

    gate_gross = gross_ann >= MIN_GROSS
    gate_to = to <= MAX_TURNOVER
    return {"label": label, "shp": s(r), "ann": ann, "dd": dd,
            "is": s(r[r.index < IS_END]), "oos": s(r[r.index >= IS_END]),
            "gross_ann": gross_ann, "turnover": to,
            "gate_gross": gate_gross, "gate_to": gate_to, "ret": r}


def main() -> None:
    con = duckdb.connect(CTA_DB, read_only=True)
    close = load_smoothed_panel(con)
    print(f"smoothed面板: {close.shape[1]}品种, 截止 {close.index.max().date()}")
    close = close[close.index <= "2025-12-31"]

    # ── S1: basis momentum (Δbasis 12周) ──
    ts = con.execute("""
        SELECT trade_date, product, rank, close, days_to_expiry
        FROM term_structure_daily
        WHERE rank <= 2 AND close > 0 AND days_to_expiry > 0
        ORDER BY trade_date""").df()
    ts["trade_date"] = pd.to_datetime(ts["trade_date"])
    p1 = ts[ts["rank"] == 1].set_index(["trade_date", "product"])
    p2 = ts[ts["rank"] == 2].set_index(["trade_date", "product"])
    ci = p1.index.intersection(p2.index)
    dt = (p1.loc[ci, "days_to_expiry"] - p2.loc[ci, "days_to_expiry"]).clip(lower=1)
    basis = ((p1.loc[ci, "close"] - p2.loc[ci, "close"]) / p2.loc[ci, "close"]
             * (365 / dt)).rename("basis").reset_index()
    basis_p = basis.pivot_table(index="trade_date", columns="product", values="basis")
    common1 = [p for p in basis_p.columns if p in close.columns]
    basis_p = basis_p[common1].reindex(close.index).ffill(limit=5)
    bm_score = basis_p.pct_change(60, fill_method=None)
    s1 = run_bt(build_signal(bm_score), close[common1], "S1_basis_momentum")

    # ── S2: 横截面动量 (4/13/26/52周收益均值) ──
    rets_w = [close.pct_change(w, fill_method=None) for w in (20, 65, 130, 260)]
    mom_score = sum(r.fillna(0) for r in rets_w) / 4
    mom_score = mom_score.replace(0, np.nan)
    common2 = [p for p in mom_score.columns if close[p].notna().sum() > 750]
    s2 = run_bt(build_signal(mom_score[common2]), close[common2], "S2_xsec_momentum")

    # ── S3: VRP 横截面 (vrp_percentile, 低VRP做多? vrp高=IV相对RV贵 → 做空波动率贵) ──
    vrp = con.execute("""
        SELECT trade_date, product, vrp_percentile FROM vrp_daily
        WHERE vrp_percentile IS NOT NULL""").df()
    vrp["trade_date"] = pd.to_datetime(vrp["trade_date"])
    vrp_p = vrp.pivot_table(index="trade_date", columns="product", values="vrp_percentile")
    common3 = [p for p in vrp_p.columns if p in close.columns]
    # vrp高 (IV贵) → 做空; vrp低 → 做多: score 取负让 top=做多
    vrp_score = -vrp_p[common3].reindex(close.index).ffill(limit=5)
    s3 = run_bt(build_signal(vrp_score), close[common3], "S3_vrp_xsec")
    con.close()

    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_r = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)

    print(f"\n{'策略':22s} {'全样本':>7s} {'IS':>7s} {'OOS':>7s} {'DD':>7s} "
          f"{'毛价差/年':>9s} {'换手':>6s}  r(主CTA)  纪律门槛")
    print("-" * 104)
    rows = []
    for s in (s1, s2, s3):
        common = s["ret"].index.intersection(main_r.index)
        r_m = s["ret"][common].corr(main_r[common])
        same_dir = (s["is"] > 0) == (s["oos"] > 0)
        gate = s["gate_gross"] and s["gate_to"]
        ok = (gate and s["oos"] >= 0.15 and same_dir and abs(r_m) < 0.4)
        print(f"{s['label']:22s} {s['shp']:+7.3f} {s['is']:+7.3f} {s['oos']:+7.3f} "
              f"{s['dd']:>+7.1%} {s['gross_ann']:>+9.2%} {s['turnover']:>5.1f}x  "
              f"{r_m:+8.3f}  毛价差{'✓' if s['gate_gross'] else '✗'} "
              f"换手{'✓' if s['gate_to'] else '✗'} 同向{'✓' if same_dir else '✗'} "
              f"→ {'PASS' if ok else 'FAIL'}")
        rows.append({"strategy": s["label"], "shp": round(s["shp"], 3),
                     "ann": round(s["ann"], 4), "dd": round(s["dd"], 4),
                     "is": round(s["is"], 3), "oos": round(s["oos"], 3),
                     "gross_ann": round(s["gross_ann"], 4),
                     "turnover": round(s["turnover"], 1), "r_main": round(r_m, 3),
                     "pass": ok})
        s["ret"].to_frame("ret").to_parquet(
            f"{REPORT_DIR}/{s['label']}_navs.parquet")

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/cross_sectional.csv", index=False)
    print("\n落盘: cross_sectional.csv + *_navs.parquet")


if __name__ == "__main__":
    main()

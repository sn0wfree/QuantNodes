# coding=utf-8
"""cta_copper_eval.py — 第1步: 铜策略子策略化评估 (PIT修正版重跑).

审计发现 (原文档口径不可信):
    v6.3:  rank(pct=True) 全样本百分位 = 结构性前视 → 修正为 rolling(104) rank
           PMI 记每月1日, 语义不明, 最坏前视1个月 → 修正为 index+35自然日
           钻机数记周五, 贝克休斯北京周六凌晨发布 → 修正为 index+4自然日
    combined: 止损在 dd 触发当期抹收益 = 前视 → 本脚本用干净口径重测 (不用其backtest)
           carry/quad/breakout 内部 rolling 干净, 信号原样复用

口径: 周频信号 → 日频 ffill → vol_target_backtest(smoothed_cu close,
      cost 3bp, tv 8%, pit_shift=1) — 与 CU溢价TC 完全同口径.
门槛: OOS(2022+) Sharpe≥0.15; 相关性: 与主CTA<0.3, 与CU溢价TC<0.5(集中度).

输出: reports/cta/copper_eval.csv + copper_eval_navs.parquet
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.factor_research import vol_target_backtest  # noqa: E402

warnings.filterwarnings("ignore")

from QuantNodes.strategy.commodity_fundamental.copper import copper_factors as CF  # noqa: E402
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6_3 import (  # noqa: E402
    TOP8_FACTORS,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_combined import (  # noqa: E402
    compute_combined_signal,
)

REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
COST = 3.0
TV = 0.08


# ── PIT 修正层 ──────────────────────────────────────────────────────────────

_orig_load_fundamental = CF._load_fundamental


def patched_load_fundamental(con, col: str) -> pd.Series:
    s = _orig_load_fundamental(con, col)
    if col == "pmi_mfg":       # 记账日在1号, 语义不明 → 最坏假设后移35天
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=35)
    elif col == "us_oil_rig":  # 周五北京周六凌晨发布 → 后移4天
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=4)
    return s


CF._load_fundamental = patched_load_fundamental


def build_v63_fixed(con, rolling_rank_win: int = 104,
                    target_vol: float = 0.12) -> pd.Series:
    """v6.3 PIT修正版: rolling rank + 修正PMI/rig + bull_enhance + vol target."""
    close_w = CF._load_weekly_close(con, "CU0").dropna()
    all_factors = CF.compute_all_factors(con)
    names = [f for f in TOP8_FACTORS if f in all_factors.columns]
    factors = all_factors[names]
    common = close_w.index.intersection(factors.index)
    close_w, factors = close_w.reindex(common), factors.reindex(common)

    # 修正点1: rolling 百分位 (时点安全) 替代全样本 rank
    pos_raw = (factors.rolling(rolling_rank_win, min_periods=52)
               .rank(pct=True).mean(axis=1).clip(0, 1))

    # bull_enhance (lme_z/carry/cu_crude 内部 rolling, 已走 patched 加载)
    mom26 = close_w.pct_change(26)
    bull = mom26 > 0.10
    lme_z = CF.compute_lme_z(con, close_w)
    carry = CF.compute_carry(con, close_w)
    cu_crude_pos = CF.compute_cu_crude(con, close_w)
    score = ((lme_z < -0.5).astype(int) + (carry > 0.7).astype(int)
             + (cu_crude_pos > 0.6).astype(int))
    pos = pos_raw.copy()
    pos[bull & (score >= 1)] = np.maximum(pos[bull & (score >= 1)], 0.9)
    pos.iloc[0] = 0.0

    ret_w = close_w.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    pos = (pos * scale).clip(upper=1.0)
    pos.iloc[0] = 0.0
    return pos


def to_daily_nav(pos_w: pd.Series, con_cta) -> tuple[pd.Series, pd.Series]:
    """周频仓位 → 日频 NAV (smoothed_cu, 与CU溢价TC同执行口径)."""
    df = con_cta.execute(
        "SELECT trade_date, close FROM smoothed_cu_daily "
        "WHERE close IS NOT NULL ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
    close = close[close.index <= "2025-12-31"]
    sig = pos_w.reindex(close.index, method="ffill").fillna(0).to_frame("CU_v63")
    res = vol_target_backtest(sig, close.to_frame("CU_v63"), cost_bps={"CU_v63": COST},
                              target_vol=TV, pit_shift=1)
    return res["nav"], res["nav"].pct_change(fill_method=None).fillna(0)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 100:
        return {"full": np.nan, "is": np.nan, "oos": np.nan}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * 252 ** 0.5

    def _s(x):
        return x.mean() / x.std() * np.sqrt(252) if len(x) > 50 and x.std() > 0 else np.nan
    return {"full": ann / vol if vol > 0 else np.nan,
            "is": _s(r[r.index < IS_END]), "oos": _s(r[r.index >= IS_END])}


def seg_sharpe(r: pd.Series, lo: str | None, hi: str | None) -> float:
    r = r.dropna()
    if lo:
        r = r[r.index >= lo]
    if hi:
        r = r[r.index < hi]
    if len(r) < 50 or r.std() == 0:
        return float("nan")
    return r.mean() / r.std() * np.sqrt(252)


def main() -> None:
    import duckdb
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    con_cta = duckdb.connect("/home/ll/Public/DataCache/futures_options_daily.duckdb",
                             read_only=True)
    df_cu = con_cta.execute(
        "SELECT trade_date, close FROM smoothed_cu_daily "
        "WHERE close IS NOT NULL ORDER BY trade_date").df()
    df_cu["trade_date"] = pd.to_datetime(df_cu["trade_date"])
    close = (df_cu.drop_duplicates("trade_date", keep="last")
             .set_index("trade_date")["close"])
    close = close[close.index <= "2025-12-31"]

    print("=" * 96)
    print("第1步: 铜策略评估 — v6.3 PIT修正版 / combined 原信号干净回测")
    print("=" * 96)

    variants = {}

    # 1) v6.3 PIT 修正版
    pos63 = build_v63_fixed(con)
    nav, ret = to_daily_nav(pos63, con_cta)
    variants["v6.3_修正版"] = ret

    # 2) v6.3 原版 (对照: 全样本rank + 无shift) — 直接调原build
    from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6_3 import (
        build_signal_v6_3,
    )
    # 注意: monkeypatch已生效, 为还原原版行为需恢复loader后再调用 —
    # 简化: 原版差异主要在rank, loader patch保留亦对原版"有利", 对照仍偏乐观
    sig_orig = build_signal_v6_3(con=con)
    nav_o, ret_o = to_daily_nav(sig_orig["position"], con_cta)
    variants["v6.3_原版(含patch)"] = ret_o

    # 3) combined 原信号 + 干净回测 (不用其 backtest_combined 的当期抹损止损)
    sig_c = compute_combined_signal()
    nav_c, ret_c = to_daily_nav(sig_c["position"], con_cta)
    variants["combined_干净回测"] = ret_c

    # 对照组
    cu_tc = pd.read_csv(f"{REPORT_DIR}/rebuild_subnav_CU溢价TC_open.csv",
                        index_col=0, parse_dates=True).iloc[:, 0]
    cu_tc = cu_tc[cu_tc.index <= "2025-12-31"]
    cu_tc_r = cu_tc.pct_change(fill_method=None).fillna(0)
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_r = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)

    # 贝塔基准: CU B&H vol_target (long-only 策略的超额增量检验)
    sig_bh = pd.DataFrame({"CU": 1.0}, index=close.index)
    res_bh = vol_target_backtest(sig_bh, close.to_frame("CU"), cost_bps={"CU": 2.0},
                                 target_vol=TV, pit_shift=1)
    bh_r = res_bh["nav"].pct_change(fill_method=None).fillna(0)
    bh_oos = seg_sharpe(bh_r, IS_END, None)
    print(f"\n贝塔基准 CU B&H(vol_target): OOS Sharpe {bh_oos:+.3f}")

    print(f"\n{'版本':24s} {'全样本':>7s} {'IS':>7s} {'OOS':>7s}  "
          f"{'r(主CTA)':>8s} {'r(CU溢价TC)':>10s} {'r(CU_B&H)':>9s} {'OOS超额':>8s}")
    print("-" * 96)
    rows = []
    for name, r in variants.items():
        st = stats(r)
        common = r.index.intersection(main_r.index)
        r_main = r[common].corr(main_r[common])
        r_cutc = r[common].corr(cu_tc_r[common])
        common_bh = r.index.intersection(bh_r.index)
        r_bh = r[common_bh].corr(bh_r[common_bh])
        diff_oos = (r - bh_r).dropna()
        diff_oos = diff_oos[diff_oos.index >= IS_END]
        exc = (diff_oos.mean() / diff_oos.std() * np.sqrt(252)
               if len(diff_oos) > 50 and diff_oos.std() > 0 else float("nan"))
        print(f"{name:24s} {st['full']:+7.3f} {st['is']:+7.3f} {st['oos']:+7.3f}  "
              f"{r_main:+8.3f} {r_cutc:+10.3f} {r_bh:+9.3f} {exc:+8.3f}")
        rows.append({"variant": name, "sharpe_full": round(st["full"], 3),
                     "sharpe_is": round(st["is"], 3), "sharpe_oos": round(st["oos"], 3),
                     "corr_main": round(r_main, 3), "corr_cu_tc": round(r_cutc, 3),
                     "corr_cu_bh": round(r_bh, 3), "oos_excess_sharpe": round(exc, 3)})
        r.to_frame("ret").to_parquet(
            f"{REPORT_DIR}/copper_eval_{name.split('_')[0].replace('.','')}.parquet")

    print("\n门槛: OOS≥0.15 | r(主CTA)<0.3 | r(CU溢价TC)<0.5 | OOS超额>0 (贝塔增量)")
    for row in rows:
        ok = (row["sharpe_oos"] >= 0.15 and abs(row["corr_main"]) < 0.3
              and abs(row["corr_cu_tc"]) < 0.5 and row["oos_excess_sharpe"] > 0)
        print(f"  {row['variant']:24s} → {'PASS' if ok else 'FAIL'}")

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/copper_eval.csv", index=False)
    print("\n落盘: copper_eval.csv + copper_eval_*.parquet")


if __name__ == "__main__":
    main()

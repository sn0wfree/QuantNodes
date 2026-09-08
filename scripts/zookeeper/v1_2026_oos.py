# coding: utf-8
"""v1_2026_oos.py — §28 V1 真样本外检验: 组合延伸到 2026-09-04 (独立脚本).

不动现有脚本; 纯 import 复用:
  - FactorModuleComboStrategy + FINAL_LUMP_CFG/CALIBRATED_COSTS (§17 golden 同管线)
  - cta_rebuild_portfolio 的子策略信号逻辑 (复制实现, 数据源参数化)
  - gold_ensemble.compute_layered_ensemble_signal (AU layered, cmf 库, 不动)
  - cta_t10y_removal.run_portfolio (§27 组合管线)

Phase A 口径对齐 (end=2025-12-31):
  A1 主CTA重跑 (_tmp_2026 库, start=2018) → 对比 golden Sharpe 2.609
  A2 子策略重建 (_tmp_2026 库) → 对比既有 rebuild_subnav CSV
Phase B 延伸 (end=2026-09-04)
Phase C 2026 段组合业绩 + 预先声明判定表 (§28.23):
  Sharpe>=2.0 确认 / 1.0-2.0 部分成立 / 0.3-1.0 嫌疑重 / <0.3 过拟合证实
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import fields
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_final_baseline import FINAL_LUMP_CFG  # noqa: E402
from cta_overfit_final_check import CALIBRATED_COSTS  # noqa: E402
from cta_shared.factor_research import (  # noqa: E402
    is_oos_direction_consistency,
    rolling_zscore,
    vol_target_backtest,
)
from cta_t10y_removal import run_portfolio  # noqa: E402
from QuantNodes.strategy.cta.data_loader import CtaDataLoader  # noqa: E402
from QuantNodes.strategy.cta.strategies.combo import ComboConfig  # noqa: E402
from QuantNodes.strategy.cta.strategies.factor_modules import (  # noqa: E402
    FactorModuleComboStrategy,
)

TMP_DB = "/home/ll/Public/QuantNodes/export/_tmp_2026.duckdb"
CMF_DB = "/home/ll/Public/QuantNodes/data/cmf/fundamental.duckdb"
REPORT_DIR = Path("/home/ll/Public/QuantNodes/reports/cta")
OUT_DIR = REPORT_DIR / "zookeeper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_END = "2022-01-01"
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0
BASE_LAMBDA_TV = 0.5
W_MAIN = 0.45  # E2-R 权重 (run_portfolio 语义: 主CTA小数)

# E3-R (主CTA70) 定版权重 (主CTA小数 + 子策略百分比, run_portfolio /100)
E2R_AU_WEIGHTS = {"主CTA": 0.70, "CU溢价TC": 10, "jjm_coke": 10,
                  "AU_layered": 10}

# 预先声明判定表 (§28.23)
VERDICT = [(2.0, "确认 (过拟合嫌疑解除)"),
           (1.0, "部分成立 (OOS衰减正常, 可定版但DD预期上调)"),
           (0.3, "嫌疑重 (需 V2/V3 补验)"),
           (0.0, "过拟合证实 (降级 E2-R+AU10%)")]


def verdict_of(shp26: float) -> str:
    for th, msg in VERDICT:
        if shp26 >= th:
            return msg
    return VERDICT[-1][1]


# ── A1: 主 CTA 重跑 (v11 golden 管线, 数据源可切换) ─────────────────────────

def run_main_cta(duckdb_path: str, start: str, end: str) -> pd.Series:
    cfg_kw = dict(FINAL_LUMP_CFG)
    cfg_kw["tvpr_lambda_tv"] = BASE_LAMBDA_TV
    cfg_kw["momentum_filter_window"] = 60
    cfg_kw.update(CALIBRATED_COSTS)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    loader = CtaDataLoader(duckdb_path=duckdb_path)
    s = FactorModuleComboStrategy(
        config=cfg, data_loader=loader,
        products=list(__import__("QuantNodes.strategy.cta.strategies.base",
                                 fromlist=["DEFAULT_PRODUCTS"]).DEFAULT_PRODUCTS),
        start=start, end=end,
    )
    r = s.run()
    return r.nav


# ── A2: 子策略重建 (复制 cta_rebuild_portfolio 逻辑, 库参数化) ────────────────

def load_smoothed_full(tbl: str, end: str) -> pd.DataFrame:
    con = duckdb.connect(TMP_DB, read_only=True)
    df = con.execute(
        f"SELECT trade_date, open, close, open_interest FROM smoothed_{tbl}_daily "
        f"WHERE close IS NOT NULL AND open IS NOT NULL AND trade_date <= '{end}' "
        f"ORDER BY trade_date").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.drop_duplicates("trade_date", keep="last").set_index("trade_date")


def load_tfapi(index_name: str) -> pd.Series:
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        f"WHERE index_name LIKE '{index_name}' ORDER BY trade_date").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].sort_index()


def build_cu(end: str) -> tuple:
    prem = load_tfapi("洋山铜溢价")
    tc = load_tfapi("%粗炼费%")
    df = load_smoothed_full("cu", end)
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
    for name, f in factors.items():
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END, min_is_spread=0.05)
        if cons and cons["agree"] and abs(cons["is_spread"]) > 0.05:
            sig += (np.sign(f) * np.sign(cons["is_spread"])
                    * (f.abs() > 1.0).astype(float)).fillna(0)
    sig = np.sign(sig).to_frame("CU")
    return sig, close.to_frame("CU"), df["open"], 3.0, 0.08, 0


def build_jjm(end: str) -> tuple:
    d1, d2 = load_smoothed_full("j", end), load_smoothed_full("jm", end)
    common = d1.index.intersection(d2.index)
    c1, c2 = d1["close"].reindex(common), d2["close"].reindex(common)
    spread = np.log(c1 / c2)
    roll = spread.rolling(120, min_periods=60)
    z = (spread - roll.mean()) / roll.std()
    sig = pd.DataFrame(
        {"jjm_coke": (-np.sign(z) * (z.abs() > 1.0).astype(float)).fillna(0).values},
        index=common)
    close = pd.DataFrame({"jjm_coke": c1.values}, index=common)
    return sig, close, d1["open"].reindex(common), 2.0, 0.10, 1


def strat_nav(sig, close, open_s, cost, tv, pit) -> tuple[pd.Series, pd.Series]:
    """close 口径 NAV (对齐 rebuild_subnav_*_close/open 的 close 版)."""
    res_c = vol_target_backtest(sig, close, cost_bps={sig.columns[0]: cost},
                                target_vol=tv, pit_shift=pit)
    ret_open = (open_s.shift(-1) / open_s - 1).reindex(close.index)
    res_o = vol_target_backtest(sig, close, cost_bps={sig.columns[0]: cost},
                                target_vol=tv, pit_shift=pit,
                                ret_df=ret_open.shift(1).to_frame(sig.columns[0]))
    return res_c["nav"], res_o["nav"]


def build_au_layered(end: str) -> pd.Series:
    """AU layered o2o + vt8% (体系口径, 与 gold_layered_layered.parquet 同)."""
    from QuantNodes.strategy.commodity_fundamental.gold.gold_ensemble import (
        compute_layered_ensemble_signal,
    )
    con = duckdb.connect(TMP_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, open, close FROM smoothed_au_daily "
        "WHERE close IS NOT NULL AND open IS NOT NULL AND trade_date <= ? "
        "ORDER BY trade_date", [end]).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    au = (df.drop_duplicates("trade_date", keep="last")
          .set_index("trade_date")[["open", "close"]])
    sig_df = compute_layered_ensemble_signal()
    sig = sig_df["position"]
    sig_ = sig.reindex(au.index).fillna(0).to_frame("AU_L")
    ret_open = (au["open"].shift(-1) / au["open"] - 1)
    res = vol_target_backtest(sig_, au[["close"]].rename(columns={"close": "AU_L"}),
                              cost_bps={"AU_L": 0.02}, target_vol=0.08, pit_shift=1,
                              ret_df=ret_open.shift(1).to_frame("AU_L"))
    return res["nav"]


# ── 工具 ────────────────────────────────────────────────────────────────────

def stats_of(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"AR": np.nan, "Vol": np.nan, "Sharpe": np.nan,
                "MaxDD": np.nan, "Calmar": np.nan, "days": len(r)}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return {"AR": round(ann, 4), "Vol": round(vol, 4),
            "Sharpe": round(ann / vol, 3) if vol > 0 else np.nan,
            "MaxDD": round(dd, 4), "Calmar": round(ann / abs(dd), 2),
            "days": len(r)}


def pct(nav: pd.Series) -> pd.Series:
    return nav.pct_change(fill_method=None).fillna(0)


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--skip-main", action="store_true",
                    help="跳过主CTA重跑 (调试子策略时用)")
    args = ap.parse_args()

    # ── Phase A: 口径对齐 (end=2025-12-31) ──
    nav_main_align = None
    if args.phase in ("a", "all"):
        print("=" * 100)
        print("Phase A: 口径对齐 (数据源=_tmp_2026, end=2025-12-31)")
        print("=" * 100)
        if not args.skip_main:
            nav_main_align = run_main_cta(TMP_DB, "2018-01-01", "2025-12-31")
            st = stats_of(pct(nav_main_align)[pct(nav_main_align).index >= "2019-01-01"])
            print(f"A1 主CTA重跑: Sharpe={st['Sharpe']:.3f} (golden参照 2.609) "
                  f"AR={st['AR']:.2%} DD={st['MaxDD']:.2%}")
        # 子策略对齐
        for name, builder in [("CU溢价TC", build_cu), ("jjm_coke", build_jjm)]:
            sig, close, open_s, cost, tv, pit = builder("2025-12-31")
            nav_c, nav_o = strat_nav(sig, close, open_s, cost, tv, pit)
            st_c = stats_of(pct(nav_c)[pct(nav_c).index >= "2019-01-01"])
            # 参照 (既有 CSV)
            ref_c = pd.read_csv(REPORT_DIR / f"rebuild_subnav_{name}_close.csv",
                                index_col=0, parse_dates=True).iloc[:, 0]
            st_ref = stats_of(pct(ref_c)[pct(ref_c).index >= "2019-01-01"])
            print(f"A2 {name}: 重跑 Sharpe={st_c['Sharpe']:.3f} "
                  f"(参照 {st_ref['Sharpe']:.3f}) AR {st_c['AR']:.2%}/{st_ref['AR']:.2%}")
        au_ = build_au_layered("2025-12-31")
        st_au = stats_of(pct(au_)[pct(au_).index >= "2019-01-01"])
        ref_au = pd.read_parquet(REPORT_DIR / "gold_layered_layered.parquet")["ret"]
        st_ref_au = stats_of(ref_au[ref_au.index >= "2019-01-01"])
        print(f"A2 AU_layered: 重跑 Sharpe={st_au['Sharpe']:.3f} "
              f"(参照 {st_ref_au['Sharpe']:.3f})")

    # ── Phase B: 延伸 2026-09-04 ──
    if args.phase in ("b", "all"):
        print("\n" + "=" * 100)
        print("Phase B: 延伸到 2026-09-04")
        print("=" * 100)
        nav_main = run_main_cta(TMP_DB, "2018-01-01", "2026-09-04")
        nav_main.to_frame("nav").to_parquet(OUT_DIR / "v1_main_cta_2026.parquet")
        st26 = stats_of(pct(nav_main)[pct(nav_main).index >= "2026-01-01"])
        print(f"B1 主CTA 2026段: Sharpe={st26['Sharpe']:.3f} AR={st26['AR']:.2%} "
              f"DD={st26['MaxDD']:.2%} ({st26['days']}日)")
        # 子策略 2026
        for name, builder in [("CU溢价TC", build_cu), ("jjm_coke", build_jjm)]:
            sig, close, open_s, cost, tv, pit = builder("2026-09-04")
            nav_c, nav_o = strat_nav(sig, close, open_s, cost, tv, pit)
            nav_c.to_frame("nav").to_csv(OUT_DIR / f"v1_subnav_{name}_close_2026.csv")
            st = stats_of(pct(nav_c)[pct(nav_c).index >= "2026-01-01"])
            print(f"B2 {name} 2026段: Sharpe={st['Sharpe']:.3f} AR={st['AR']:.2%} "
                  f"DD={st['MaxDD']:.2%}")
        au26 = build_au_layered("2026-09-04")
        au26.to_frame("nav").to_csv(OUT_DIR / "v1_subnav_AU_layered_2026.csv")
        st = stats_of(pct(au26)[pct(au26).index >= "2026-01-01"])
        print(f"B2 AU_layered 2026段: Sharpe={st['Sharpe']:.3f} AR={st['AR']:.2%} "
              f"DD={st['MaxDD']:.2%}")

    # ── Phase C: 2026 组合业绩 + 判定 ──
    if args.phase in ("c", "all"):
        print("\n" + "=" * 100)
        print("Phase C: E2-R+AU10% 组合 2026 段 (真样本外)")
        print("=" * 100)
        nav_main = (pd.read_parquet(OUT_DIR / "v1_main_cta_2026.parquet")["nav"]
                    if (OUT_DIR / "v1_main_cta_2026.parquet").exists()
                    else run_main_cta(TMP_DB, "2018-01-01", "2026-09-04"))
        rets = {"主CTA": pct(nav_main)}
        for name in ["CU溢价TC", "jjm_coke"]:
            nav = pd.read_csv(OUT_DIR / f"v1_subnav_{name}_close_2026.csv",
                              index_col=0, parse_dates=True).iloc[:, 0]
            rets[name] = pct(nav)
        au26 = pd.read_csv(OUT_DIR / "v1_subnav_AU_layered_2026.csv",
                           index_col=0, parse_dates=True).iloc[:, 0]
        rets["AU_layered"] = pct(au26)
        df = pd.DataFrame(rets)
        df = df.reindex(df.dropna(how="all").index).fillna(0)
        print(f"组合输入: {df.shape}, {df.index[0].date()} ~ {df.index[-1].date()}")

        net = run_portfolio(df, E2R_AU_WEIGHTS)
        segs = [("2019-2025 全样本", "2019-01-01", "2026-01-01"),
                ("2026 真OOS (1-9月)", "2026-01-01", None)]
        rows = []
        for lab, lo, hi in segs:
            x = net.dropna()
            if lo:
                x = x[x.index >= lo]
            if hi:
                x = x[x.index < hi]
            st = stats_of(x)
            rows.append({"window": lab, **st})
        out = pd.DataFrame(rows)
        out.to_csv(OUT_DIR / "v1_2026_oos.csv", index=False)
        print(out.to_string(index=False))
        shp26 = rows[-1]["Sharpe"]
        print(f"\n判定 (预先声明 §28.23): 2026 段 Sharpe={shp26:.3f} → {verdict_of(shp26)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""cta_au_substrategy.py — 黄金子品种策略 (独有信号源).

信号池 (AU单品种, 与CU溢价+TC方法论一致):
    G1 ETF持仓水平 z-score(252日) — 西方投资需求 (日频2004起)
    G2 ETF持仓20日变化 z-score — 投资需求边际
    G3 央行购金3月累计 z-score — 央行需求 (月频2016起, 2022-26购金潮核心驱动)
    G4 GVZ水平 z-score(252日) — 黄金波动率状态
    C1 CFTC投机净持仓 z-score(52周) — 投机拥挤度 (周频2010起, PIT+3日)
    C2 CFTC商业净持仓 z-score — 产业方向

经济方向 (供IC验证参考, 不预设):
    ETF: 正向 (投资流入→涨)
    央行: 正向 (购金→涨)
    GVZ: 反向? (波动率极端→避险需求→涨? 或恐慌后反转→跌? 由数据定)
    CFTC mm: 反向 (拥挤度, CFTC全组教训) — 由分品种独立验证定

方法论: cta_shared组件库, IS/OOS方向一致性≥60%, OOS Sharpe≥0.15

用法:
    python3 scripts/cta_au_substrategy.py            # 全流程
    python3 scripts/cta_au_substrategy.py --ic-only  # 仅验证
"""
from __future__ import annotations

import argparse
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from cta_shared.factor_research import (  # noqa: E402
    REPORT_DIR, compute_ic, is_oos_direction_consistency, period_stats,
    rolling_zscore, summarize_factor_test, vol_target_backtest, yearly_returns,
)

CMF_DB = "data/cmf/fundamental.duckdb"
CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
IS_END = "2022-01-01"
MIN_IS_SPREAD = 0.05
Z_EXTREME = 1.0
COST_BPS = 2.0  # AU: 滑点0.25+手续费0.02, 保守2bp


def _cmf_value(value_col: str, source: str = "gold_position") -> pd.Series:
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source=? AND value_col=? ORDER BY trade_date",
        [source, value_col],
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


def load_cftc_gold() -> pd.DataFrame:
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT report_date, oi, mm_net, pm_long, pm_short FROM tfapi_cftc "
        "WHERE market LIKE 'GOLD - COMMODITY%' ORDER BY report_date"
    ).df()
    con.close()
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.set_index("report_date")


def load_au_close() -> pd.Series:
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM smoothed_au_daily "
        "WHERE close IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def build_factors(cn_days: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """6因子, 日频对齐+PIT."""
    etf = _cmf_value("etf_tonnes")
    cb = _cmf_value("cb_reserves")
    gvz = _cmf_value("gvz", "gold_fred")
    cftc = load_cftc_gold()

    # 日频ffill对齐
    etf_d = etf.reindex(cn_days.union(etf.index)).ffill().reindex(cn_days)
    gvz_d = gvz.reindex(cn_days.union(gvz.index)).ffill().reindex(cn_days)

    # 央行购金: 月度变化→3月累计(月频→日频ffill)
    cb_chg = cb if cb.iloc[0] < 100 else cb.diff()  # 值域<100吨=月变化量
    cb_3m = cb_chg.rolling(3).sum() if isinstance(cb_chg, pd.Series) else None
    cb_3m = cb_3m.reindex(cn_days.union(cb_3m.index)).ffill().reindex(cn_days)
    cb_3m = cb_3m.shift(30)  # 月频数据滞后30天PIT

    # CFTC: 周频→日历ffill→PIT+3交易日
    oi = cftc["oi"].replace(0, np.nan)
    mm_net_pct = cftc["mm_net"] / oi
    com_net = (cftc["pm_long"] - cftc["pm_short"]) / oi
    cftc_d = pd.DataFrame({"mm": mm_net_pct, "com": com_net}).reindex(
        cn_days.union(cftc.index)).ffill().reindex(cn_days).shift(3)

    out = {
        "G1_etf_level": rolling_zscore(etf_d, 252),
        "G2_etf_chg": rolling_zscore(etf_d.pct_change(20), 252),
        "G3_cb_buy": rolling_zscore(cb_3m, 756),
        "G4_gvz_level": rolling_zscore(gvz_d, 252),
        "C1_mm_level": rolling_zscore(cftc_d["mm"], 252),
        "C2_com_level": rolling_zscore(cftc_d["com"], 252),
    }
    return {k: v.shift(1) for k, v in out.items()}  # PIT: 次日可用


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()

    close = load_au_close()
    factors = build_factors(close.index)
    fwd5 = close.pct_change(5, fill_method=None).shift(-5)

    print("=== AU: 黄金独有信号验证 (未来5日) ===")
    robust = []
    for name, f in factors.items():
        ic = compute_ic(f, fwd5, min_obs=300)
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END,
                                            min_is_spread=MIN_IS_SPREAD)
        print("  " + summarize_factor_test(ic, cons, name))
        if cons and cons["agree"] and abs(cons["is_spread"]) > MIN_IS_SPREAD:
            robust.append((name, cons))

    n_sig = sum(1 for f in factors.values()
                if (c := is_oos_direction_consistency(f, fwd5, IS_END, MIN_IS_SPREAD))
                and c["is_significant"])
    if n_sig and len(robust) / n_sig < 0.6:
        print(f"\n❌ 一致率{len(robust)}/{n_sig}<60% → 止损归档")
        return
    if not robust:
        print("\n❌ 无稳健因子 → 归档")
        return
    if args.ic_only:
        return

    print(f"\n=== IS定方向回测 ({len(robust)}个稳健因子) ===")
    sig = pd.Series(0.0, index=close.index)
    for name, cons in robust:
        sig += (np.sign(factors[name]) * np.sign(cons["is_spread"])
                * (factors[name].abs() > Z_EXTREME).astype(float)).fillna(0)
    sig = np.sign(sig)

    res = vol_target_backtest(sig.to_frame("AU"), close.to_frame("AU"),
                              cost_bps={"AU": COST_BPS}, target_vol=0.08)
    st = period_stats(res["pnl"], is_end=IS_END)
    print(f"  全期: Sharpe={res['sharpe']:.3f} AnnRet={res['ann_ret']:+.2%} "
          f"MaxDD={res['max_dd']:.2%} 换手={res['turnover_annual']:.1f}x")
    print(f"  IS:   Sharpe={st['is']['sharpe']:.3f} AnnRet={st['is']['ann_ret']:+.2%}")
    print(f"  OOS:  Sharpe={st['oos']['sharpe']:.3f} AnnRet={st['oos']['ann_ret']:+.2%} "
          f"MaxDD={st['oos']['max_dd']:.2%}")

    if st["oos"]["sharpe"] < 0.15:
        print("❌ OOS Sharpe<0.15 → 止损归档")
        return

    yr = yearly_returns(res["pnl"])
    print("\n年度: " + "  ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
    pd.DataFrame({"nav": res["nav"]}).to_csv(REPORT_DIR / "au_sub_nav.csv")
    yr.to_frame("ann_ret").to_csv(REPORT_DIR / "au_sub_yearly.csv")
    print(f"\n✓ 通过OOS门槛, 已保存: {REPORT_DIR}/au_sub_*.csv")


if __name__ == "__main__":
    main()

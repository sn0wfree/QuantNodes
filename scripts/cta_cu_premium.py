#!/usr/bin/env python3
"""cta_cu_premium.py — 铜品种独特信号: 洋山铜溢价 + 冶炼TC.

信号池 (CU单品种, 与锌库存方法论一致):
    P1 洋山溢价水平 z-score(252日) — 进口需求紧张度 (溢价高=国内需求强)
    P2 洋山溢价变化 20日 z-score — 需求边际变化
    T1 TC水平 z-score(252日) — 矿供应紧张度 (TC低=矿短缺→冶炼减产→利多)
    T2 TC变化 20日 z-score — 供应边际变化

经济方向 (供IC验证参考, 不预设):
    溢价: 正向 (需求强→涨)
    TC:   反向 (TC低=矿紧→涨)

数据:
    洋山铜溢价: 日频 2012起 (tfapi_indicators)
    铜冶炼TC:   周频 2011起 (tfapi_indicators)
    CU价格:     smoothed_cu_daily

方法论: cta_shared组件库, IS(2018-2022)/OOS(2022+)方向一致性≥60%,
        OOS Sharpe≥0.15

用法:
    python3 scripts/cta_cu_premium.py            # 全流程
    python3 scripts/cta_cu_premium.py --ic-only  # 仅验证
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
COST_BPS = 3.0  # CU: 滑点1.33+手续费0.05, 保守3bp


def load_premium() -> pd.Series:
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name='洋山铜溢价' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


def load_tc() -> pd.Series:
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name LIKE '%粗炼费%' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


def load_cu_close() -> pd.Series:
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM smoothed_cu_daily "
        "WHERE close IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def build_factors(premium: pd.Series, tc: pd.Series,
                  cn_days: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """4因子, 日频对齐+PIT(次日可用, shift 1)."""
    prem_d = premium.reindex(cn_days.union(premium.index)).ffill().reindex(cn_days)
    tc_d = tc.reindex(cn_days.union(tc.index)).ffill().reindex(cn_days)
    out = {
        "P1_prem_level": rolling_zscore(prem_d, 252),
        "P2_prem_chg": rolling_zscore(prem_d.pct_change(20), 252),
        "T1_tc_level": rolling_zscore(tc_d, 252),
        "T2_tc_chg": rolling_zscore(tc_d.pct_change(20), 252),
    }
    return {k: v.shift(1) for k, v in out.items()}  # PIT: 次日可用


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()

    close = load_cu_close()
    factors = build_factors(load_premium(), load_tc(), close.index)
    fwd5 = close.pct_change(5, fill_method=None).shift(-5)

    print("=== CU: 洋山溢价+TC 因子验证 (未来5日) ===")
    robust = []
    for name, f in factors.items():
        ic = compute_ic(f, fwd5, min_obs=300)
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END,
                                            min_is_spread=MIN_IS_SPREAD)
        print("  " + summarize_factor_test(ic, cons, name))
        if cons and cons["agree"] and abs(cons["is_spread"]) > MIN_IS_SPREAD:
            robust.append((name, cons))

    n_sig = sum(1 for _, c in [(n, is_oos_direction_consistency(f, fwd5, IS_END, MIN_IS_SPREAD))
                               for n, f in factors.items()] if c and c["is_significant"])
    if len(robust) / max(n_sig, 1) < 0.6:
        print(f"\n❌ 一致率{len(robust)}/{n_sig}<60% → 止损归档")
        return
    if not robust:
        print("\n❌ 无IS显著且方向一致的因子 → 归档")
        return
    if args.ic_only:
        return

    print(f"\n=== IS定方向回测 ({len(robust)}个稳健因子) ===")
    sig = pd.Series(0.0, index=close.index)
    for name, cons in robust:
        sig += (np.sign(factors[name]) * np.sign(cons["is_spread"])
                * (factors[name].abs() > Z_EXTREME).astype(float)).fillna(0)
    sig = np.sign(sig)

    res = vol_target_backtest(sig.to_frame("CU"), close.to_frame("CU"),
                              cost_bps={"CU": COST_BPS}, target_vol=0.08)
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
    pd.DataFrame({"nav": res["nav"]}).to_csv(REPORT_DIR / "cu_premium_nav.csv")
    yr.to_frame("ann_ret").to_csv(REPORT_DIR / "cu_premium_yearly.csv")
    print(f"\n✓ 通过OOS门槛, 已保存: {REPORT_DIR}/cu_premium_*.csv")


if __name__ == "__main__":
    main()

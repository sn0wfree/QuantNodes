#!/usr/bin/env python3
"""cta_m_deep.py — M豆粕经纪持仓深入因子 (扩展版).

复用 cta_shared.factor_research 组件库, 在原 cta_broker_holding.py 基础上新增6个细粒度因子.

新增因子:
    N1 持仓突变率 — CR5净持仓5日变化z-score (已有, 对照)
    N2 经纪排名变化 — 头部CR5进出 (排名变化→调仓方向)
    N3 持仓集中度突变 — CR5占比20日变化
    N4 净持仓不对称 — 多头经纪net_long分布偏度 (散户机构分歧)
    N5 持仓-成交比 — 持仓变化÷成交量 (真实调仓vs换手)
    N6 经纪集中度CR10 — CR10 vs CR5 (更宽头部信息)

方法论:
    - 因子独立 IS/OOS一致性 (cta_shared.is_oos_direction_consistency)
    - 与原F3/F4/F5合并, IS定方向回测
    - OOS Sharpe≥0.15 止损
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
COST_BPS = 5.5  # M: 滑点3.3+手续费1.5, 保守5.5bp


def load_m_broker() -> pd.DataFrame:
    """M经纪持仓日频表 (汇总到品种级, 列: broker, volume, long, short, net_long)."""
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, broker, SUM(volume) AS volume, SUM(long) AS long, "
        "SUM(short) AS short, SUM(net_long) AS net_long "
        "FROM tfapi_holding WHERE symbol_prefix='M' "
        "GROUP BY trade_date, broker ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["trade_date", "volume"], ascending=[True, False])
    return df


def load_m_close() -> pd.Series:
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM smoothed_m_daily "
        "WHERE close IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def build_factors(df: pd.DataFrame) -> dict[str, pd.Series]:
    """从经纪级持仓构造日频因子面板 (返回多列DataFrame)."""
    df = df.copy()
    df["rank"] = df.groupby("trade_date").cumcount() + 1
    df["is_cr5"] = df["rank"] <= 5
    df["is_cr10"] = df["rank"] <= 10

    daily = df.groupby("trade_date").agg(
        total_long=("long", "sum"), total_short=("short", "sum"),
        n_brokers=("broker", "nunique"),
    )
    daily["total_oi"] = (daily["total_long"] + daily["total_short"]) / 2
    daily["total_volume"] = df.groupby("trade_date")["volume"].sum()

    cr5 = df[df["is_cr5"]].groupby("trade_date").agg(
        cr5_net=("net_long", "sum"), cr5_oi=("long", "sum"),
        cr5_volume=("volume", "sum"), cr5_long=("long", "sum"),
    )
    cr10 = df[df["is_cr10"]].groupby("trade_date").agg(
        cr10_oi=("long", "sum"), cr10_volume=("volume", "sum"),
    )

    # N2 经纪排名变化: 每日排名CR5的"进入/退出"事件 (标记新进=1, 退出=-1)
    cr5_set_t = df[df["is_cr5"]].groupby("trade_date")["broker"].apply(set)
    cr10_set_t = df[df["is_cr10"]].groupby("trade_date")["broker"].apply(set)

    panel = daily.join(cr5).join(cr10)
    panel["cr5_net_pct"] = panel["cr5_net"] / panel["total_oi"]
    panel["cr5_share"] = panel["cr5_volume"] / panel["total_volume"]

    # N3 持仓集中度突变
    panel["cr5_share_chg20"] = panel["cr5_share"].diff(20)

    # N4 净持仓不对称 (多头经纪net_long分布的偏度, 横截面)
    skew_by_day = df.groupby("trade_date")["net_long"].agg(lambda x: x.skew())
    # N5 持仓-成交比 (持仓变化÷成交量, top5)
    cr5_chg = panel["cr5_net"].diff()
    cr5_vol = panel["cr5_volume"].rolling(5).sum()
    panel["cr5_pos_vol_ratio"] = cr5_chg / cr5_vol.replace(0, np.nan)

    # N2 排名变化: 新进入头部=+, 退出=-
    n2 = []
    prev_cr5 = set()
    prev_cr10 = set()
    dates_sorted = sorted(cr5_set_t.index)
    for d in dates_sorted:
        new5 = cr5_set_t[d] - prev_cr5
        out5 = prev_cr5 - cr5_set_t[d]
        new10 = cr10_set_t[d] - prev_cr10
        out10 = prev_cr10 - cr10_set_t[d]
        # 新进头部得分(+1/经纪), 退出得分(-1/经纪)
        net_churn = (len(new10) + len(new5)) - (len(out10) + len(out5))
        n2.append({"date": d, "n2_churn": net_churn})
        prev_cr5 = cr5_set_t[d]
        prev_cr10 = cr10_set_t[d]
    churn = pd.DataFrame(n2).set_index("date")["n2_churn"]

    factors = pd.DataFrame(index=panel.index)
    factors["N1_cr5_net_chg"] = rolling_zscore(panel["cr5_net_pct"].diff(5), 130)
    factors["N2_churn"] = churn.reindex(factors.index).ffill()
    factors["N3_cr5_share_chg"] = rolling_zscore(panel["cr5_share_chg20"], 130)
    factors["N4_skew"] = skew_by_day.reindex(factors.index).ffill().pipe(
        lambda s: rolling_zscore(s, 130))
    factors["N5_pos_vol_ratio"] = rolling_zscore(panel["cr5_pos_vol_ratio"], 130)

    # N6 CR10 vs CR5
    factors["N6_cr10_minus_cr5"] = (
        panel["cr10_oi"] / panel["total_oi"] - panel["cr5_oi"] / panel["total_oi"]
    )

    return factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()

    df = load_m_broker()
    close = load_m_close()
    factors = build_factors(df)
    factors = factors.reindex(close.index).ffill().shift(1)  # PIT+日频对齐

    fwd5 = close.pct_change(5, fill_method=None).shift(-5)

    print("=== M: 6个新因子验证 (未来5日) ===")
    robust = []
    for name in factors.columns:
        f = factors[name]
        ic = compute_ic(f, fwd5, min_obs=300)
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END,
                                            min_is_spread=MIN_IS_SPREAD)
        print("  " + summarize_factor_test(ic, cons, name))
        if cons and cons["agree"] and abs(cons["is_spread"]) > MIN_IS_SPREAD:
            robust.append((name, cons))

    n_sig = sum(1 for col in factors.columns
                if (c := is_oos_direction_consistency(
                    factors[col], fwd5, IS_END, MIN_IS_SPREAD))
                and c["is_significant"])
    if n_sig and len(robust) / n_sig < 0.6:
        print(f"\n❌ 一致率{len(robust)}/{n_sig}<60% → 新因子止损, 保留原M策略(F3+F4+F5 OOS 0.58)")
        return
    if not robust:
        print("\n❌ 无新稳健因子 → 保留原M策略")
        return
    if args.ic_only:
        return

    print(f"\n=== IS定方向回测 ({len(robust)}个新稳健因子) ===")
    sig = pd.Series(0.0, index=close.index)
    for name, cons in robust:
        sig += (np.sign(factors[name]) * np.sign(cons["is_spread"])
                * (factors[name].abs() > Z_EXTREME).astype(float)).fillna(0)
    sig = np.sign(sig)

    res = vol_target_backtest(sig.to_frame("M"), close.to_frame("M"),
                              cost_bps={"M": COST_BPS}, target_vol=0.08)
    st = period_stats(res["pnl"], is_end=IS_END)
    print(f"  新因子OOS: Sharpe={st['oos']['sharpe']:.3f} AnnRet={st['oos']['ann_ret']:+.2%}")
    print("  (对照: 原M策略F3+F4+F5组合 OOS Sharpe=0.581)")
    print(f"  全期换手: {res['turnover_annual']:.1f}x")

    if st["oos"]["sharpe"] < 0.15:
        print("❌ 新因子OOS<0.15 → 不合并")
        return

    yr = yearly_returns(res["pnl"])
    print("\n年度: " + "  ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
    pd.DataFrame({"nav": res["nav"]}).to_csv(REPORT_DIR / "m_deep_nav.csv")
    yr.to_frame("ann_ret").to_csv(REPORT_DIR / "m_deep_yearly.csv")
    print(f"\n✓ 新因子有改善, 已保存: {REPORT_DIR}/m_deep_*.csv")


if __name__ == "__main__":
    main()

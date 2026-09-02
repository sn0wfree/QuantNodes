#!/usr/bin/env python3
"""cta_broker_holding.py — 经纪持仓因子子策略 (CU/M 国内聪明钱).

数据: tfapi_holding (162-168家期货公司日频持仓, 2020起)
合约处理: 品种级汇总 (移仓总量守恒, 天然消除换月噪声)

因子 (5个, 品种级):
    F1: CR5净持仓占比 z-score (52日) — 机构方向(水平)
    F2: CR5净持仓5日变化 z-score (26日) — 机构调仓(跟单核心)
    F3: CR5市场份额 — 主力控盘度
    F4: 多空分歧度 — 前5多 vs 前5空对冲比
    F5: 散户-机构差 — 尾部经纪与头部方向差

方法论 (CFTC教训):
    IC + 五分位分层 → IS(2020-2022)/OOS(2023-2026)方向一致性
    一致率<60%止损 | OOS Sharpe<0.15止损

用法:
    python3 scripts/cta_broker_holding.py           # 全流程
    python3 scripts/cta_broker_holding.py --ic-only # 仅IC+分层
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

CMF_DB = "data/cmf/fundamental.duckdb"
CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
REPORT_DIR = Path("reports/cta")

PRODUCTS = ["CU", "M"]
Z_WIN = 52          # 52日z-score窗口
CHG_WIN = 5         # 5日变化
IS_END = "2023-01-01"  # IS=2020-2022, OOS=2023-2026


def load_broker_daily(prefix: str) -> pd.DataFrame:
    """品种级汇总: 按(trade_date, broker)聚合全合约持仓 (移仓守恒)."""
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, broker, SUM(volume) AS volume, SUM(long) AS long, "
        "SUM(short) AS short, SUM(net_long) AS net_long "
        "FROM tfapi_holding WHERE symbol_prefix=? "
        "GROUP BY trade_date, broker ORDER BY trade_date",
        [prefix],
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_cn_close(products: list[str]) -> pd.DataFrame:
    """国内品种日频收盘价."""
    con = duckdb.connect(CTA_DB, read_only=True)
    out = {}
    for p in products:
        df = con.execute(
            f"SELECT trade_date, close FROM smoothed_{p.lower()}_daily "
            f"WHERE close IS NOT NULL ORDER BY trade_date"
        ).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        out[p] = df.set_index("trade_date")["close"]
    con.close()
    return pd.DataFrame(out)


def build_factors(prefix: str) -> pd.DataFrame:
    """品种级5因子 (日频, index=trade_date)."""
    df = load_broker_daily(prefix)
    if df.empty:
        return pd.DataFrame()

    # 每日排名: 按volume降序
    df = df.sort_values(["trade_date", "volume"], ascending=[True, False])
    df["rank"] = df.groupby("trade_date").cumcount() + 1
    df["is_cr5"] = df["rank"] <= 5

    # 市场总持仓 (多头总量=空头总量)
    daily = df.groupby("trade_date").agg(
        total_long=("long", "sum"), total_short=("short", "sum"),
        n_brokers=("broker", "nunique"),
    )
    daily["total_oi"] = (daily["total_long"] + daily["total_short"]) / 2

    # CR5聚合
    cr5 = df[df["is_cr5"]].groupby("trade_date").agg(
        cr5_net=("net_long", "sum"), cr5_oi=("long", "sum"),
        cr5_volume=("volume", "sum"),
    )
    tail = df[~df["is_cr5"]].groupby("trade_date").agg(tail_net=("net_long", "sum"))
    # 前5多头/前5空头经纪 (按多/空持仓分别排名)
    top_l = df.sort_values(["trade_date", "long"], ascending=[True, False]).groupby("trade_date").head(5)  # noqa: E501
    top_s = df.sort_values(["trade_date", "short"], ascending=[True, False]).groupby("trade_date").head(5)  # noqa: E501
    hedge = top_l.groupby("trade_date")["long"].sum().to_frame("top5_long")
    hedge["top5_short"] = top_s.groupby("trade_date")["short"].sum()

    panel = daily.join(cr5).join(tail).join(hedge).dropna(subset=["total_oi"])
    panel["cr5_net_pct"] = panel["cr5_net"] / panel["total_oi"]

    factors = pd.DataFrame(index=panel.index)
    m52 = panel["cr5_net_pct"].rolling(Z_WIN, min_periods=26).mean()
    s52 = panel["cr5_net_pct"].rolling(Z_WIN, min_periods=26).std().replace(0, np.nan)
    # F1: 头部净持仓占比 z-score
    factors["F1_cr5_level"] = (panel["cr5_net_pct"] - m52) / s52
    # F2: 头部调仓5日变化 z-score
    chg = panel["cr5_net_pct"].diff(CHG_WIN)
    m26 = chg.rolling(130, min_periods=26).mean()
    s26 = chg.rolling(130, min_periods=26).std().replace(0, np.nan)
    factors["F2_cr5_chg"] = (chg - m26) / s26
    # F3: CR5市场份额 (控盘度)
    factors["F3_cr5_share"] = panel["cr5_volume"] / df.groupby("trade_date")["volume"].sum()
    # F4: 多空分歧度 (前5多 vs 前5空 对冲比)
    factors["F4_hedge_ratio"] = panel["top5_long"] / panel["top5_short"].replace(0, np.nan)
    # F5: 散户-机构方向差
    factors["F5_tail_head"] = (panel["tail_net"] - panel["cr5_net"]) / panel["total_oi"]
    return factors


def zscore(s: pd.Series, win: int = 252) -> pd.Series:
    m = s.rolling(win, min_periods=60).mean()
    sd = s.rolling(win, min_periods=60).std().replace(0, np.nan)
    return (s - m) / sd


def quintile_spread(f: pd.Series, ret5: pd.Series, mask: pd.Series) -> float | None:
    fv, rv = f[mask & f.notna() & ret5.notna()], ret5[mask & f.notna() & ret5.notna()]
    if len(fv) < 150:
        return None
    q = pd.qcut(fv.rank(method="first"), 5, labels=False)
    return (rv[q == 4].mean() - rv[q == 0].mean()) * 252 / 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()

    cn_close = load_cn_close(PRODUCTS)
    print(f"国内价格: {cn_close.index.min().date()}~{cn_close.index.max().date()}")

    all_ic, all_factors = [], {}
    for prefix in PRODUCTS:
        print(f"\n=== {prefix}: 因子构建 ===")
        f = build_factors(prefix)
        if f.empty:
            print("  ❌ 无数据")
            continue
        # F3/F4是水平值 → z-score标准化 (跨期可比)
        for col in ["F3_cr5_share", "F4_hedge_ratio", "F5_tail_head"]:
            f[col] = zscore(f[col])
        f = f.reindex(cn_close.index).ffill()
        all_factors[prefix] = f
        print(f"  因子: {list(f.columns)}, {f.index.min().date()}~{f.index.max().date()}")

        ret5 = cn_close[prefix].pct_change(5, fill_method=None).shift(-5)
        is_mask = pd.Series(f.index < IS_END, index=f.index)

        print(f"--- {prefix} IC + 五分位年化价差 (未来5日) ---")
        for col in f.columns:
            fv = f[col]
            valid = fv.notna() & ret5.notna()
            if valid.sum() < 300:
                continue
            ic = fv[valid].corr(ret5[valid])
            roll_ic = fv[valid].rolling(52).corr(ret5[valid]).dropna()
            t = (roll_ic.mean() / (roll_ic.std() / np.sqrt(len(roll_ic)))
                 if len(roll_ic) > 10 else np.nan)
            s_is = quintile_spread(fv, ret5, is_mask)
            s_oos = quintile_spread(fv, ret5, ~is_mask)
            agree = (s_is is not None and s_oos is not None
                     and np.sign(s_is) == np.sign(s_oos) and abs(s_is) > 0.05)
            all_ic.append({"product": prefix, "factor": col, "ic": ic,
                           "roll_ic": roll_ic.mean(), "t_stat": t,
                           "is_spread": s_is, "oos_spread": s_oos,
                           "agree": agree})
            print(f"  {col:<16} IC={ic:+.3f} rollIC={roll_ic.mean():+.3f} t={t:+.1f} "
                  f"IS={s_is if s_is is None else f'{s_is:+.1%}'} "
                  f"OOS={s_oos if s_oos is None else f'{s_oos:+.1%}'} "
                  f"{'✓一致' if agree else ''}")

    ic_df = pd.DataFrame(all_ic)
    if ic_df.empty:
        return
    ic_df.to_csv(REPORT_DIR / "broker_holding_ic.csv", index=False)

    n_sig = (ic_df["is_spread"].abs() > 0.05).sum()
    n_agree = ic_df["agree"].sum()
    print(f"\n=== IS/OOS一致性: {n_agree}/{n_sig} (IS显著对中) ===")
    if n_sig > 0 and n_agree / n_sig < 0.6:
        print("❌ 一致率<60% → 止损归档")
        return
    if args.ic_only:
        return

    # Step5: IS定方向回测
    print("\n=== Step 5: IS定方向回测 ===")
    cost_table = json.loads((REPORT_DIR / "cost_model.json").read_text())
    cost_bps = {p: (cost_table.get(p, {}).get("slippage_bps", 5.0)
                    + cost_table.get(p, {}).get("commission_bps", 0.1))
                for p in PRODUCTS}

    signal = pd.DataFrame(0.0, index=cn_close.index, columns=cn_close.columns)
    for _, row in ic_df.iterrows():
        if not row["agree"] or abs(row["is_spread"]) <= 0.05:
            continue
        f = all_factors[row["product"]][row["factor"]]
        direction = np.sign(row["is_spread"])
        sig = (np.sign(f) * direction * (f.abs() > 1.0).astype(float)).fillna(0)
        signal[row["product"]] += sig
    signal = signal.clip(-1, 1)

    ret = cn_close.pct_change(fill_method=None)
    vol = ret.rolling(20).std() * np.sqrt(252)
    scale = (0.08 / vol).clip(upper=1.0).fillna(0)
    pos = signal * scale
    turnover = pos.diff().abs()
    cost = turnover * pd.Series(cost_bps).reindex(pos.columns).fillna(5.0) / 10000
    pnl = (pos.shift(1) * ret - cost).sum(axis=1).dropna()

    nav = (1 + pnl).cumprod()
    ann = nav.iloc[-1] ** (252 / len(pnl)) - 1
    dd = (nav / nav.cummax() - 1).min()
    sh = ann / (pnl.std() * np.sqrt(252))
    oos_pnl = pnl[pnl.index >= IS_END]

    def st(r):
        nav_r = (1 + r).cumprod()
        a = nav_r.iloc[-1] ** (252 / len(r)) - 1
        d = (nav_r / nav_r.cummax() - 1).min()
        return a / (r.std() * np.sqrt(252)), a, d

    sh_o, a_o, d_o = st(oos_pnl)
    print(f"  全期:   Sharpe={sh:.3f} AnnRet={ann:+.2%} MaxDD={dd:.2%}")
    print(f"  OOS期:  Sharpe={sh_o:.3f} AnnRet={a_o:+.2%} MaxDD={d_o:.2%}")
    print(f"  年化换手: {turnover.sum(axis=1).mean() * 252:.1f}x")

    pd.DataFrame({"nav": nav}).to_csv(REPORT_DIR / "broker_holding_nav.csv")
    print(f"已保存: {REPORT_DIR}/broker_holding_*.csv")


if __name__ == "__main__":
    main()

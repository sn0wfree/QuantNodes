# coding=utf-8
"""cta_carry_eval.py — 横截面 carry 子策略 (§25).

信号: carry_i = (P_near - P_next)/P_next × 365/(days_near - days_next)
    backwardation (near>next, carry>0) → 做多; contango → 做空
组合: 日度横截面排序, 多 top 1/3 / 空 bottom 1/3 / 中 1/3 = 0, 等权
    vol_target 8%, cost 3bp, pit_shift=1 (T收盘信号→T+1执行)
执行收益: smoothed 主力连续 close (信号为品种选择, 经典近似)
过滤: rank1 成交量>1000手, days_to_expiry 有效, 价格>0

门槛 (预设): OOS Sharpe≥0.15 + IS/OOS同向 + r(主CTA)<0.3
    多空对称 → 无 long-only 贝塔问题, 但仍验证与主CTA正交性
输出: reports/cta/carry_eval.csv + carry_navs.parquet
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
MIN_VOL = 1000
TIER = 3  # top/bottom 分位


def load_carry() -> pd.DataFrame:
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute("""
        SELECT trade_date, product, rank, close, days_to_expiry, volume
        FROM term_structure_daily
        WHERE rank <= 2 AND close > 0 AND days_to_expiry IS NOT NULL
          AND days_to_expiry > 0 AND volume > 1000
        ORDER BY trade_date
    """).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    p1 = df[df["rank"] == 1].set_index(["trade_date", "product"])
    p2 = df[df["rank"] == 2].set_index(["trade_date", "product"])
    common = p1.index.intersection(p2.index)
    p1, p2 = p1.loc[common], p2.loc[common]
    dt = (p1["days_to_expiry"] - p2["days_to_expiry"]).clip(lower=1)
    carry = (p1["close"] - p2["close"]) / p2["close"] * (365 / dt)
    out = carry.rename("carry").reset_index()
    return out.pivot_table(index="trade_date", columns="product", values="carry")


def load_near_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """rank1 (近月) close + symbol 面板 — carry持仓收益应含展期, 用近月自身."""
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute("""
        SELECT trade_date, product, symbol, close
        FROM term_structure_daily
        WHERE rank = 1 AND close > 0
        ORDER BY trade_date
    """).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.pivot_table(index="trade_date", columns="product", values="close")
    sym = df.pivot_table(index="trade_date", columns="product", values="symbol",
                         aggfunc="first")
    return close, sym


def near_returns(close: pd.DataFrame, sym: pd.DataFrame) -> pd.DataFrame:
    """近月合约日收益; 换月日 (symbol变化) 置0 剔除拼接伪影."""
    ret = close.pct_change(fill_method=None)
    roll_day = sym != sym.shift(1)
    ret[roll_day] = 0.0
    return ret


def main() -> None:
    carry = load_carry()
    print(f"carry 面板: {carry.shape[1]}品种 x {carry.shape[0]}天 "
          f"({carry.index.min().date()}~{carry.index.max().date()})")

    near_close, near_sym = load_near_panel()
    products = [p for p in carry.columns if p in near_close.columns]
    print(f"近月面板交集: {len(products)}品种")
    carry = carry[products]

    # 横截面排序信号: 每日 top/bottom 1/3, ±1/n_active (资金等分, 防杠杆爆炸)
    ranks = carry.rank(axis=1, pct=True)
    sig = pd.DataFrame(0.0, index=carry.index, columns=carry.columns)
    sig[ranks >= 1 - 1 / TIER] = 1.0
    sig[ranks <= 1 / TIER] = -1.0
    sig[carry.isna()] = 0.0
    n_active = (sig != 0).sum(axis=1).clip(lower=1)
    sig = sig.div(n_active, axis=0)
    print(f"日均多空腿: {(n_active).mean():.0f} 品种 (单腿)")

    # 持仓收益: 近月合约自身 (含展期), 换月日剔除
    ret1 = near_returns(
        near_close[products].reindex(carry.index.union(near_close.index)).sort_index(),
        near_sym[products].reindex(carry.index.union(near_close.index)).sort_index(),
    )
    sig = sig.reindex(ret1.index).fillna(0)
    ret1 = ret1.reindex(sig.index)
    res = vol_target_backtest(sig, ret1, cost_bps=COST, target_vol=TV, pit_shift=1,
                              ret_df=ret1)
    r = res["nav"].pct_change(fill_method=None).fillna(0)
    r.to_frame("ret").to_parquet(f"{REPORT_DIR}/carry_navs.parquet")

    def stats(x: pd.Series) -> dict:
        x = x.dropna()
        nav = (1 + x).cumprod()
        ann = nav.iloc[-1] ** (252 / len(x)) - 1
        vol = x.std() * np.sqrt(252)
        dd = (nav / nav.cummax() - 1).min()

        def _s(seg):
            if len(seg) > 50 and seg.std() > 0:
                return seg.mean() / seg.std() * np.sqrt(252)
            return np.nan
        return {"full": ann / vol if vol > 0 else np.nan, "ar": ann, "dd": dd,
                "is": _s(x[x.index < IS_END]), "oos": _s(x[x.index >= IS_END])}

    st = stats(r)
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_r = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)
    common = r.index.intersection(main_r.index)
    r_main = r[common].corr(main_r[common])

    # 多空分腿诊断: 多头腿/空头腿收益 (2018-2025)
    n_ret = n_active.reindex(ret1.index)
    long_ret = (sig.clip(lower=0) * ret1).sum(axis=1) / n_ret
    short_ret = (-sig.clip(upper=0) * ret1).sum(axis=1) / n_ret

    print(f"\ncarry 全样本: Sharpe {st['full']:+.3f} AR {st['ar']:+.2%} DD {st['dd']:+.2%} "
          f"| IS {st['is']:+.3f} | OOS {st['oos']:+.3f}")
    print(f"多腿均值日 {long_ret.mean():+.4%} 空腿 {short_ret.mean():+.4%} "
          f"(多-空 {long_ret.mean()-short_ret.mean():+.4%})")
    print(f"r(主CTA) = {r_main:+.3f}")

    ok = (st["oos"] >= 0.15 and (st["is"] > 0) == (st["oos"] > 0)
          and abs(r_main) < 0.3)
    print(f"门槛: OOS {st['oos']:+.3f}≥0.15 | IS/OOS同向 {((st['is'] > 0) == (st['oos'] > 0))} | "
          f"r(主CTA)<0.3 → {'PASS' if ok else 'FAIL'}")

    rows = [{"shp_full": round(st["full"], 3), "ar": round(st["ar"], 4),
             "dd": round(st["dd"], 4), "shp_is": round(st["is"], 3),
             "shp_oos": round(st["oos"], 3), "r_main": round(r_main, 3),
             "pass": ok}]
    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/carry_eval.csv", index=False)
    print("落盘: carry_eval.csv + carry_navs.parquet")


if __name__ == "__main__":
    main()

# coding=utf-8
"""cta_sub_sensitivity_audit.py — P3: 子策略参数敏感性抽查.

审计对象 (干净信号三兄弟):
    jjm_coke: log(J/JM) z{win} 反转, 阈值|z|>{th}   定版: z120, 1.0
    ZN库存:   LME锌库存 z{win} 极端, 阈值|z|>{th}    定版: z52, 1.0
    T10y:     OI {n}日变化 sign                      定版: 5

判定: 各邻域参数 OOS Sharpe 无刀尖效应 (同号且量级相近) → 稳健.
输出: reports/cta/sub_sensitivity_audit.csv
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
CMF_DB = "data/cmf/fundamental.duckdb"
IS_END = "2022-01-01"

CON = duckdb.connect(CTA_DB, read_only=True)
CON2 = duckdb.connect(CMF_DB, read_only=True)


def load_smoothed_full(tbl: str) -> pd.DataFrame:
    df = CON.execute(
        f"SELECT trade_date, open, close, open_interest FROM smoothed_{tbl}_daily "
        f"WHERE close IS NOT NULL AND open IS NOT NULL ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.drop_duplicates("trade_date", keep="last").set_index("trade_date")


def stats(r: pd.Series) -> tuple[float, float]:
    r = r.dropna()
    is_r, oos_r = r[r.index < IS_END], r[r.index >= IS_END]
    def _s(x):
        return x.mean() / x.std() * np.sqrt(252) if len(x) > 50 and x.std() > 0 else np.nan
    return _s(is_r), _s(oos_r)


def run_bt(sig: pd.DataFrame, close: pd.DataFrame, cost: float, tv: float,
           pit: int) -> tuple[float, float]:
    res = vol_target_backtest(sig, close, cost_bps={sig.columns[0]: cost},
                              target_vol=tv, pit_shift=pit)
    return stats(res["nav"].pct_change(fill_method=None).fillna(0))


def jjm_variant(win: int, th: float) -> tuple[float, float]:
    d1, d2 = load_smoothed_full("j"), load_smoothed_full("jm")
    common = d1.index.intersection(d2.index)
    c1, c2 = d1["close"].reindex(common), d2["close"].reindex(common)
    spread = np.log(c1 / c2)
    roll = spread.rolling(win, min_periods=win // 2)
    z = (spread - roll.mean()) / roll.std()
    sig = pd.DataFrame(
        {"jjm": (-np.sign(z) * (z.abs() > th).astype(float)).fillna(0).values},
        index=common)
    return run_bt(sig, c1.to_frame("jjm"), 2.0, 0.10, 1)


def zn_variant(win: int, th: float) -> tuple[float, float]:
    inv = CON2.execute(
        "SELECT trade_date, value FROM inventory_daily "
        "WHERE source='LME铜库存' AND value_col='锌-库存' ORDER BY trade_date").df()
    inv["trade_date"] = pd.to_datetime(inv["trade_date"])
    inv = inv.set_index("trade_date")["value"].sort_index()
    close = load_smoothed_full("zn")["close"]
    common = inv.dropna().index.intersection(close.dropna().index)
    inv_c, close_c = inv[common], close[common]
    m = inv_c.rolling(win, min_periods=win // 2).mean()
    sd = inv_c.rolling(win, min_periods=win // 2).std().replace(0, np.nan)
    z = (inv_c - m) / sd
    sig = pd.Series(0.0, index=z.index)
    sig[z > th] = 1.0
    sig[z < -th] = -1.0
    return run_bt(sig.to_frame("ZN"), close_c.to_frame("ZN"), 10.0, 0.08, 1)


def t10y_variant(n: int) -> tuple[float, float]:
    df = load_smoothed_full("t")
    oin = df["open_interest"].pct_change(n, fill_method=None)
    sig = pd.DataFrame({"T10y": np.sign(oin).fillna(0).values}, index=df.index)
    return run_bt(sig, df[["close"]].rename(columns={"close": "T10y"}), 2.0, 0.10, 1)


def main() -> None:
    grid = {
        "jjm_coke": [("z90_t1.0", lambda: jjm_variant(90, 1.0)),
                     ("z120_t1.0*", lambda: jjm_variant(120, 1.0)),
                     ("z150_t1.0", lambda: jjm_variant(150, 1.0)),
                     ("z120_t0.8", lambda: jjm_variant(120, 0.8)),
                     ("z120_t1.2", lambda: jjm_variant(120, 1.2))],
        "ZN库存":   [("z40_t1.0", lambda: zn_variant(40, 1.0)),
                     ("z52_t1.0*", lambda: zn_variant(52, 1.0)),
                     ("z65_t1.0", lambda: zn_variant(65, 1.0)),
                     ("z52_t0.8", lambda: zn_variant(52, 0.8)),
                     ("z52_t1.2", lambda: zn_variant(52, 1.2))],
        "T10y":     [("oi3", lambda: t10y_variant(3)),
                     ("oi5*", lambda: t10y_variant(5)),
                     ("oi10", lambda: t10y_variant(10))],
    }
    rows = []
    for name, variants in grid.items():
        print(f"\n=== {name} (IS Shp | OOS Shp) ===")
        for label, fn in variants:
            is_s, oos_s = fn()
            star = "*" if label.endswith("*") else " "
            print(f"  {label:14s}{star} IS {is_s:+.3f} | OOS {oos_s:+.3f}")
            rows.append({"strategy": name, "variant": label.rstrip("*"),
                         "is_sharpe": round(is_s, 3), "oos_sharpe": round(oos_s, 3)})
    pd.DataFrame(rows).to_csv("reports/cta/sub_sensitivity_audit.csv", index=False)
    print("\n落盘: reports/cta/sub_sensitivity_audit.csv")


if __name__ == "__main__":
    main()

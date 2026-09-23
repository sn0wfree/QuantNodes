#!/usr/bin/env python3
# coding=utf-8
"""提取期货 carry 信号 (期限结构/展期收益) 从 ClickHouse daily_quote (§5.48).

数据源: daily_quote.v_<PRODUCT>_daily (合约级: Symbol/TradeDate/Close/OpenInterest)
carry 定义: 近月 vs 次近月 年化价差
  carry = (P_near / P_far - 1) × 365 / (到期日差)
  正 (backwardation) → 看多现货持有收益; 负 (contango) → 看空

输出: data/allweather/carry_daily.parquet (产品 × 日期的 carry 信号)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/allweather/carry_daily.parquet"

# 面板 fut_<prod> → ClickHouse 视图名
PANEL_FUTS = {
    "fut_t": "T", "fut_tf": "TF", "fut_ts": "TS", "fut_if": "IF",
    "fut_ic": "IC", "fut_ih": "IH", "fut_au": "AU", "fut_ag": "AG",
    "fut_cu": "CU", "fut_al": "AL", "fut_zn": "ZN", "fut_ni": "NI",
    "fut_rb": "RB", "fut_hc": "HC", "fut_i": "I", "fut_j": "J",
    "fut_jm": "JM", "fut_ta": "TA", "fut_ma": "MA", "fut_pp": "PP",
    "fut_l": "L", "fut_bu": "BU", "fut_sc": "SC", "fut_m": "M",
    "fut_y": "Y", "fut_p": "P", "fut_cf": "CF", "fut_sr": "SR",
    "fut_c": "C",
}

PURE = re.compile(r"^([A-Za-z]{1,2})(\d{3,4})$")


def _expiry(sym: str) -> pd.Timestamp | None:
    """合约代码 → 到期年月 (YYYY-MM 的 15 日近似)."""
    m = PURE.match(sym)
    if not m:
        return None                      # 期权/异常代码
    digits = m.group(2)
    if len(digits) == 4:
        yy, mm = int(digits[:2]), int(digits[2:])
        year = 2000 + yy
    else:
        yy, mm = int(digits[0]), int(digits[1:])
        year = 2000 + yy
    if not (1 <= mm <= 12):
        return None
    return pd.Timestamp(year=year, month=mm, day=15)


def _query(sql: str) -> pd.DataFrame:
    import subprocess
    from io import StringIO
    r = subprocess.run(
        ["curl", "-s", "-G",
         "http://default:Imsn0wfree@0.0.0.0:8123/",
         "--data-urlencode", f"query={sql} FORMAT TSVWithNames"],
        capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(r.stderr.decode()[:200] or "empty response")
    return pd.read_csv(StringIO(r.stdout.decode()), sep="\t")


def fetch_product(view: str) -> pd.DataFrame:
    return _query(
        f"SELECT TradeDate, Symbol, Close, OpenInterest "
        f"FROM daily_quote.v_{view}_daily WHERE Close > 0 AND OpenInterest > 0")


def compute_carry(df: pd.DataFrame, min_oi: float = 0) -> pd.Series:
    """单产品的 carry 时序 (近月 vs 次近月年化价差; 向量化)."""
    df = df.copy()
    df["exp"] = df["Symbol"].map(_expiry)
    df = df.dropna(subset=["exp"]).sort_values(["TradeDate", "exp"])
    df["rnk"] = df.groupby("TradeDate")["exp"].rank(method="first")
    near = df[df["rnk"] == 1].set_index("TradeDate")
    far = df[df["rnk"] == 2].set_index("TradeDate")
    j = near.index.intersection(far.index)
    days = (far.loc[j, "exp"] - near.loc[j, "exp"]).dt.days
    ok = (days > 0) & (far.loc[j, "Close"] > 0)
    s = (near.loc[j, "Close"] / far.loc[j, "Close"] - 1.0) * 365.0 / days
    s = s[ok].sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache_dir = REPO / "data/allweather/carry_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(item):
        col, view = item
        cp = cache_dir / f"{col}.parquet"
        if cp.exists():
            s = pd.read_parquet(cp).iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return col, s
        df = fetch_product(view)
        s = compute_carry(df)
        pd.DataFrame({col: s}).to_parquet(cp)
        return col, s

    frames = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, kv): kv[0] for kv in PANEL_FUTS.items()}
        for f in as_completed(futs):
            col = futs[f]
            try:
                c, s = f.result()
                if len(s):
                    frames[c] = s
                    print(f"  {c}: {len(s)} 天 "
                          f"({s.index[0].date()} → {s.index[-1].date()}), "
                          f"carry 均值 {s.mean()*100:+.2f}%", flush=True)
            except Exception as e:
                print(f"  {col}: ✗ {type(e).__name__}: {str(e)[:60]}",
                      flush=True)

    carry = pd.DataFrame(frames).sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    carry.to_parquet(OUT)
    print(f"\n已保存: {OUT} ({carry.shape})")


if __name__ == "__main__":
    main()

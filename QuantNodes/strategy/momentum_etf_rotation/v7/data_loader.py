# coding=utf-8
"""v7 数据加载: 复用现有 Excel + parquet 数据, 不重建基础设施.

来源 (用户决策: 直接复用数据, 不写代码):
  - 9 周频宏观因子: ~/Public/高频宏观因子/高频宏观因子跟踪_output_2026-06-01.xlsx
  - 5 宽基 ETF:    data/real/etf_nav_2018-01-01_2026-06-30.parquet

输出:
  - data/high_freq_macro/v9_factors_weekly.parquet       (9 因子净值)
  - data/high_freq_macro/v9_factors_weekly_returns.parquet (周对数收益)
  - data/high_freq_macro/v9_etf_daily.parquet            (5 ETF 日净值)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
HF_DIR = REPO / "data" / "high_freq_macro"
HF_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = Path.home() / "Public" / "高频宏观因子"
FACTOR_FILE = SRC_DIR / "高频宏观因子跟踪_output_2026-06-01.xlsx"
ETF_PANEL = REPO / "data" / "real" / "etf_nav_2018-01-01_2026-06-30.parquet"

# 9 宏观因子 (来自 reference output)
FACTOR_COLS = [
    "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
    "无风险收益率", "信用利差因子", "期限利差因子_债",
    "期限利差因子_股", "期限利差因子_加权", "宏观汇率因子",
]

# 5 宽基 ETF 池 (用户决策: 沪深300/中证500/创业板 + 恒生 + 国债)
ETF_POOL = ("510300", "510500", "159915", "510900", "511260")


def load_macro_factors() -> pd.DataFrame:
    """加载 9 周频宏观因子净值, 索引=周日期."""
    cache = HF_DIR / "v9_factors_weekly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if not FACTOR_FILE.exists():
        raise FileNotFoundError(f"Factor file not found: {FACTOR_FILE}")

    wb = openpyxl.load_workbook(FACTOR_FILE, data_only=True, read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").sort_index()
    out = df[FACTOR_COLS].astype(float)
    out.to_parquet(cache)
    return out


def load_factor_returns() -> pd.DataFrame:
    """加载 9 因子对数收益, 索引=周日期."""
    cache = HF_DIR / "v9_factors_weekly_returns.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    nav = load_macro_factors()
    out = np.log(nav / nav.shift(1)).dropna()
    out.to_parquet(cache)
    return out


def load_etf_panel(codes: tuple = ETF_POOL, start: str = "2018-01-01") -> pd.DataFrame:
    """加载 5 ETF 日净值, 索引=日日期. 严格截取到 start 之后.

    Returns:
        DataFrame (T, 5) 含 510300/510500/159915/510900/511260.
    """
    cache = HF_DIR / "v9_etf_daily.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.loc[start:]

    df = pd.read_parquet(ETF_PANEL)
    out = df[list(codes)].astype(float).sort_index().loc[start:]
    out.to_parquet(cache)
    return out


__all__ = [
    "FACTOR_COLS",
    "ETF_POOL",
    "load_macro_factors",
    "load_factor_returns",
    "load_etf_panel",
    "HF_DIR",
]

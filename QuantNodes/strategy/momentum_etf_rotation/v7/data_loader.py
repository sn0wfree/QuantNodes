# coding=utf-8
"""v7.3 v2 数据加载: 9 宏观因子 + 13 INDICES (faithful to source).

[关键 v2 决策]
v7.3 v1 用了 5 个 ETF (510300/510500/159915/510900/511260), 但与 source 实现不一致.
source v2 notebook cell 99/102/104 用的池子是 13 个 **指数 (level-1)**:
  沪深300 + 中证500 + 中证1000 + 恒生指数 + 4 中债 + 4 商品 (含 农产品 / 工业品 / 原油 / 黄金)
source 跑出 29.07% 年化 (12 年样本), v7.3 v1 仅 0.01-0.42%. 因 ETF vs 指数 数据结构
和价格行为不同, 改用 INDICES 一比一复刻 source 实现.

[数据来源] (用户决策: 复用源 Excel)
  - 9 周频宏观因子:  ~/Public/高频宏观因子/高频宏观因子跟踪_output_2026-06-01.xlsx
  - 13 indices:     ~/Public/高频宏观因子/Factor Minicking组合-高频宏观因子20260601.xlsx

[输出]
  - data/high_freq_macro/v9_factors_weekly.parquet (9 因子净值)
  - data/high_freq_macro/v9_factors_weekly_returns.parquet (周对数收益)
  - data/high_freq_macro/v9_indices_daily.parquet (13 INDICES 日对数收益)
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
INDEX_FILE = SRC_DIR / "Factor Minicking组合-高频宏观因子20260601.xlsx"

# 9 宏观因子
FACTOR_COLS = [
    "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
    "无风险收益率", "信用利差因子", "期限利差因子_债",
    "期限利差因子_股", "期限利差因子_加权", "宏观汇率因子",
]

# 13 INDICES (一比一复刻 source cell 99/102/104)
# 注意: 源 cell 99 用 '南华农产品指数', cell 102 用 '南华工业品指数',
# 我们用 v2 笔记本的两者并集 (与最新版 Excel 一致)
INDEX_COLS = [
    "沪深300指数",
    "中证500指数",
    "中证1000",
    "恒生指数",
    "中债10年期国债指数",
    "中债3-5年期国债指数",
    # v1 notebook 有 '中债1-3年期国债财富指数' (109); v2 没有, 我们用既有 6 个
    "中债国开行债券总指数",
    "中债企业债总指数",
    "南华综合指数",
    "南华工业品指数",       # source cell 102
    "南华农产品指数",       # source cell 99
    "期货结算价(连续):布伦特原油",
    "收盘价:沪金指数",
]


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


def load_index_panel(start: str = "2008-01-01") -> pd.DataFrame:
    """加载 13 INDICES 日对数收益, 索引=日日期.

    Returns:
        DataFrame (T, 13) 含 INDEX_COLS 13 个指数日对数收益.
    """
    cache = HF_DIR / "v9_indices_daily.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.loc[start:]

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")
    wb = openpyxl.load_workbook(INDEX_FILE, data_only=True, read_only=True)
    ws = wb["主要指数"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[1]  # 第二行
    data = rows[8:]   # 第 9 行起为数据
    df = pd.DataFrame(data, columns=[str(c) if c else f"col_{i}"
                                      for i, c in enumerate(header)])
    df["指标名称"] = pd.to_datetime(df["指标名称"], errors="coerce")
    df = df.set_index("指标名称").sort_index()
    df.index.name = "dt"

    sub = df[INDEX_COLS].apply(pd.to_numeric, errors="coerce")
    # 对数收益
    rets = np.log(sub / sub.shift(1)).dropna(how="all")
    rets = rets.loc[start:]
    rets.to_parquet(cache)
    return rets


def load_index_prices(start: str = "2008-01-01") -> pd.DataFrame:
    """加载 13 INDICES 日价格 (用于回测持仓 NAV 计算).

    Returns:
        DataFrame (T, 13) 含 INDEX_COLS 13 个指数日价格.
    """
    cache = HF_DIR / "v9_indices_daily_prices.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.loc[start:]

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index file not found: {INDEX_FILE}")
    wb = openpyxl.load_workbook(INDEX_FILE, data_only=True, read_only=True)
    ws = wb["主要指数"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[1]
    data = rows[8:]
    df = pd.DataFrame(data, columns=[str(c) if c else f"col_{i}"
                                      for i, c in enumerate(header)])
    df["指标名称"] = pd.to_datetime(df["指标名称"], errors="coerce")
    df = df.set_index("指标名称").sort_index()
    df.index.name = "dt"

    sub = df[INDEX_COLS].apply(pd.to_numeric, errors="coerce")
    sub = sub.loc[start:]
    sub.to_parquet(cache)
    return sub


# 兼容 v7.3 v1 接口, 标记 deprecation
def load_etf_panel(*args, **kwargs):
    raise NotImplementedError(
        "load_etf_panel 已弃用. v7.3 v2 用 13 INDICES (load_index_panel / load_index_prices)."
    )


__all__ = [
    "FACTOR_COLS",
    "INDEX_COLS",
    "load_macro_factors",
    "load_factor_returns",
    "load_index_panel",
    "load_index_prices",
    "HF_DIR",
]

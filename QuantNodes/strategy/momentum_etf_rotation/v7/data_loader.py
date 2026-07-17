# coding=utf-8
"""v7 数据加载: 9 宏观因子 + 13 INDICES / 56 EXPANDED (ETF+bond).

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
REAL_DIR = REPO / "data" / "real"

SRC_DIR = Path.home() / "Public" / "高频宏观因子"
FACTOR_FILE = SRC_DIR / "高频宏观因子跟踪_output_2026-06-01.xlsx"
INDEX_FILE = SRC_DIR / "Factor Minicking组合-高频宏观因子20260601.xlsx"

# 8 宏观因子 (移除 期限利差因子_加权，无价值)
FACTOR_COLS = [
    "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
    "无风险收益率", "信用利差因子", "期限利差因子_债",
    "期限利差因子_股", "宏观汇率因子",
]

# 13 INDICES (一比一复刻 source cell 99/102/104)
# 关键 (Stage 3 优化 a 2026-07-13): 替换 '南华综合指数' -> '中债1-3年国债财富指数'
# 源 cell 99 main_idx_cols 含 '中债1_3年期国债财富指数', 源 cell 73 写法:
#   main_idx['1-3 年国债财富指数'] = idx1['1-3 年国债财富指数']
# 我们从 sheet='指数' 列 11 加载该列.
INDEX_COLS = [
    "沪深300指数",
    "中证500指数",
    "中证1000",
    "恒生指数",
    "中债10年期国债指数",
    "中债3-5年期国债指数",
    "中债1-3年国债财富指数",     # <-- 新增 (源 cell 99)
    "中债国开行债券总指数",
    "中债企业债总指数",
    "南华工业品指数",            # source cell 102
    "南华农产品指数",            # source cell 99
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

    [关键修复 2026-07-11]
    原始 Excel 数据有 142 个 NaN 缺口 (跨节假日 / 周末), 直接 dropna 会丢掉
    跨缺口的真实价格变化, 导致 (1+r).cumprod() 与 log(end/start) 不一致.
    修复: 先 resample 业务日 + bfill, 让 log returns 跨缺口的累积正确.

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

    # 主要指数 sheet 拿 12 个指数 (除 中债1-3年国债财富指数)
    main_cols = [c for c in INDEX_COLS if c in df.columns]
    sub = df[main_cols].apply(pd.to_numeric, errors="coerce")

    # 从 sheet='指数' 列 11 拿 中债1-3年国债财富指数 (源 cell 73)
    if "中债1-3年国债财富指数" in INDEX_COLS and "中债1-3年国债财富指数" not in df.columns:
        ws_idx = wb["指数"]
        idx_rows = list(ws_idx.iter_rows(values_only=True))
        idx_header = idx_rows[0]
        # 找列名
        col_name = "1-3 年国债财富指数"
        if col_name in idx_header:
            idx_data = idx_rows[1:]
            idx_df = pd.DataFrame(idx_data, columns=[str(c) if c else f"col_{i}"
                                                       for i, c in enumerate(idx_header)])
            idx_df = idx_df[["dt", col_name]].copy()
            idx_df["dt"] = pd.to_datetime(idx_df["dt"], errors="coerce")
            idx_df = idx_df.dropna(subset=["dt"])
            # 去重 (指数 sheet 在 2029-12 月有 3-5 个重复, 取 first)
            idx_df = idx_df[~idx_df["dt"].duplicated(keep="first")]
            idx_df = idx_df.set_index("dt").sort_index()
            idx_df.index.name = "dt"
            idx_series = idx_df[col_name].apply(pd.to_numeric, errors="coerce")
            # 对齐到 sub 的日期
            idx_series = idx_series.reindex(sub.index)
            sub["中债1-3年国债财富指数"] = idx_series

    # 关键修复: 业务日重采样 + bfill, 解决跨假日的 NaN 缺口
    bday_idx = pd.bdate_range(start=sub.dropna().index[0], end=sub.index[-1])
    sub = sub.reindex(bday_idx).bfill()
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


def load_benchmark_price(benchmark: str = "沪深300指数") -> pd.Series:
    """加载 benchmark 指数日价格 (用于 v7 trend filter 信号, 默认沪深300).

    [Stage 4 v2 新增 2026-07-13]
    v7_macro_baseline_v2_tf 在每个调仓日检查 benchmark 价格 < 200 日 MA,
    触发则减仓到 50% + 50% 中债10年.

    Returns:
        pd.Series 索引=业务日, 值=benchmark 日价格.
    """
    cache = HF_DIR / f"v9_benchmark_{benchmark.replace('指数','')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).squeeze("columns")

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

    if benchmark not in df.columns:
        raise ValueError(f"benchmark {benchmark!r} not in 主要指数 sheet")

    s = df[benchmark].apply(pd.to_numeric, errors="coerce")
    bday_idx = pd.bdate_range(start=s.dropna().index[0], end=s.index[-1])
    s = s.reindex(bday_idx).bfill()
    s.name = benchmark
    s.to_frame().to_parquet(cache)
    return s


# ── EXPANDED POOL: 51 ETFs + 5 bond indices = 56 assets ──────────────────

# 51 ETF codes (DEFAULT_POOL 44 - 511260 bond + SmartBeta 8 unique)
# SmartBeta unique: 510880, 512890, 512260, 515900, 512040, 159786, 515080, 515100
EXPANDED_ETF_COLS: list[str] = [
    # A_BROAD (6)
    "510300", "510500", "510050", "159915", "588000", "159901",
    # A_SECTOR (20)
    "512760", "512480", "515030", "515790", "512690", "512170", "512010",
    "515050", "159928", "512880", "512000", "512800", "515220", "512200",
    "512400", "512660", "512980", "515880", "159996", "512120",
    # HK (5)
    "510900", "159920", "513010", "513050", "159740",
    # COMMODITY (6)
    "518880", "518800", "159985", "161226", "159981", "159766",
    # OVERSEAS (6)
    "513100", "513300", "513500", "513520", "513880", "159941",
    # SmartBeta (8 unique, not in DEFAULT_POOL)
    "510880", "512890", "512260", "515900", "512040", "159786", "515080", "515100",
]

# 5 bond indices (from v7 INDEX_COLS)
EXPANDED_BOND_INDICES: list[str] = [
    "中债10年期国债指数",
    "中债3-5年期国债指数",
    "中债1-3年国债财富指数",
    "中债国开行债券总指数",
    "中债企业债总指数",
]

# Combined
EXPANDED_COLS: list[str] = EXPANDED_ETF_COLS + EXPANDED_BOND_INDICES

# TF classification
EQUITY_ETF_COLS: list[str] = [
    # A_BROAD (6)
    "510300", "510500", "510050", "159915", "588000", "159901",
    # A_SECTOR (20)
    "512760", "512480", "515030", "515790", "512690", "512170", "512010",
    "515050", "159928", "512880", "512000", "512800", "515220", "512200",
    "512400", "512660", "512980", "515880", "159996", "512120",
    # HK (5)
    "510900", "159920", "513010", "513050", "159740",
    # OVERSEAS (6)
    "513100", "513300", "513500", "513520", "513880", "159941",
    # SmartBeta (8)
    "510880", "512890", "512260", "515900", "512040", "159786", "515080", "515100",
]

COMMODITY_ETF_COLS: list[str] = [
    "518880", "518800", "159985", "161226", "159981", "159766",
]


def load_expanded_panel(start: str = "2018-01-01") -> pd.DataFrame:
    """加载 56 assets 日对数收益 (51 ETFs + 5 bond indices).

    ETFs: data/real/etf_nav_*.parquet (价格水平) → 日对数收益
    Bond: data/high_freq_macro/v9_indices_daily.parquet → 日对数收益

    Returns:
        DataFrame (T, 56) 含 EXPANDED_COLS 56 个资产日对数收益.
    """
    cache = HF_DIR / "v56_expanded_daily.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return df.loc[start:]

    # ── ETF NAV → daily log returns ──
    main_nav = pd.read_parquet(REAL_DIR / "etf_nav_2018-01-01_2026-06-30.parquet")
    smartbeta_nav = pd.read_parquet(REAL_DIR / "etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")

    # Merge (dedup shared columns: 510300, 510500, 159915, 588000)
    shared = [c for c in smartbeta_nav.columns if c in main_nav.columns]
    smartbeta_unique = [c for c in smartbeta_nav.columns if c not in main_nav.columns]
    etf_nav = pd.concat([main_nav, smartbeta_nav[smartbeta_unique]], axis=1)

    # Daily log returns
    etf_rets = np.log(etf_nav / etf_nav.shift(1)).dropna(how="all")

    # ── Bond indices (already daily log returns) ──
    index_daily = pd.read_parquet(HF_DIR / "v9_indices_daily.parquet")
    bond_rets = index_daily[EXPANDED_BOND_INDICES]

    # ── Align dates (business day) ──
    common_start = max(etf_rets.dropna(how="all").index[0],
                       bond_rets.dropna(how="all").index[0])
    common_end = min(etf_rets.index[-1], bond_rets.index[-1])
    bday_idx = pd.bdate_range(start=common_start, end=common_end)

    etf_rets = etf_rets.reindex(bday_idx)
    bond_rets = bond_rets.reindex(bday_idx)

    # Forward-fill bond indices (skip weekends/holidays, not price gaps)
    bond_rets = bond_rets.ffill()

    # Combine
    expanded = pd.concat([etf_rets, bond_rets], axis=1)
    expanded.index.name = "dt"

    # Select only desired columns (drop 511260 bond ETF, keep bond indices)
    want = [c for c in EXPANDED_COLS if c in expanded.columns]
    expanded = expanded[want]

    # Cache
    expanded.to_parquet(cache)

    return expanded.loc[start:]


# 兼容 v7.3 v1 接口, 标记 deprecation
def load_etf_panel(*args, **kwargs):
    raise NotImplementedError(
        "load_etf_panel 已弃用. v7.3 v2 用 13 INDICES (load_index_panel / load_index_prices)."
    )


__all__ = [
    "FACTOR_COLS",
    "INDEX_COLS",
    "EXPANDED_COLS",
    "EXPANDED_ETF_COLS",
    "EXPANDED_BOND_INDICES",
    "EQUITY_ETF_COLS",
    "COMMODITY_ETF_COLS",
    "load_macro_factors",
    "load_factor_returns",
    "load_index_panel",
    "load_index_prices",
    "load_expanded_panel",
    "HF_DIR",
]

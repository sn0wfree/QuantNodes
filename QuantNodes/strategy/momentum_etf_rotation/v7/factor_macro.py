"""
v7.0 宏观因子 fetcher (PIT-aware, Stage 30 POC).

[设计要点] 区分 数据时间 (obs_date) 和 发布时间 (release_date):
- obs_date: 数据描述的时期 (e.g., 2024-01-31 = 1月份数据)
- release_date: 投资者实际可获得的日期 = obs_date + release_lag_days
- 回测时 T 日, 只能用 release_date <= T 的数据 → 防 look-ahead

[Stage 30 POC 范围]
- 5 宏观因子: PMI / CPI / M2 / CN10Y / US10Y
- 数据源: iFinD MCP (同花顺) edb 服务
- 本地缓存: data/ifind_cache/macro/{NAME}.parquet
- 字段: obs_date / release_date / value / name / unit

[业界标准发布滞后] (国家统计局/央行/交易所)
- PMI:        1 天 (次月 1 日 09:00 国家统计局)
- CPI/PPI:    10 天 (次月 9-10 日 09:30 国家统计局)
- M2:         12 天 (次月 10-15 日 央行)
- CN10Y:      0 天 (T+0 实时, 交易所/中债登)
- US10Y:      0 天 (T+0 实时, 彭博/同花顺)

[PIT 关键约束]
1. 月度数据 obs_date 统一为月末, release_date = obs_date + lag_days
2. 日度数据 obs_date = release_date (T+0)
3. HMM 5 状态机在 t 日输入 = T 日 PIT 后的最新可用值
4. 严禁使用 obs_date <= T 的原始数据 (这是最常见 look-ahead 错误)
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import urllib3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

IFIND_PATH = Path("/home/ll/Public/ifind-finance-data-1.1.0")
MCP_CONFIG_PATH = IFIND_PATH / "mcp_config.json"


def _i_find_call(server_type: str, tool_name: str, params: dict) -> dict:
    """iFinD MCP 调用, 兼容从任意 CWD 调用 (强制 chdir 到 IFIND_PATH).

    call.py 第 5 行用相对路径读 mcp_config.json, 必须先 chdir.
    """
    old_cwd = Path.cwd()
    try:
        os.chdir(IFIND_PATH)
        if str(IFIND_PATH) not in sys.path:
            sys.path.insert(0, str(IFIND_PATH))
        if "call" in sys.modules and hasattr(sys.modules["call"], "call"):
            call_mod = importlib.reload(sys.modules["call"])
        else:
            call_mod = importlib.import_module("call")
        return call_mod.call(server_type, tool_name, params)
    finally:
        os.chdir(old_cwd)


CACHE_DIR = Path("data/ifind_cache/macro")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# === 标准发布滞后 (天) ===
RELEASE_LAG_DAYS: dict[str, int] = {
    "PMI":   1,    # 月初 1 日 09:00
    "CPI":   10,   # 月初 9-10 日 09:30
    "M2":    12,   # 月初 10-15 日
    "CN10Y": 0,    # T+0 实时
    "US10Y": 0,    # T+0 实时
}


# === iFinD 查询模板 ===
IFIND_QUERIES: dict[str, str] = {
    "PMI":   "官方制造业PMI 2018-01至2026-06",                # PMI 是 diffusion index, 无 YoY
    "CPI":   "CPI 同比 2018-01至2026-06 居民消费价格",         # YoY %
    "M2":    "M2 同比 2018-01至2026-06 货币供应量",             # YoY %
    "CN10Y": "中国10年期国债收益率 2018-01-01至2026-06-30 日度",  # %
    "US10Y": "美国10年期国债收益率 2018-01-01至2026-06-30 日度",  # %
}


@dataclass
class MacroMeta:
    """单个宏观因子的元信息."""
    name: str
    unit: str
    freq: str             # "M" 月度 | "D" 日度
    release_lag_days: int
    column: str           # iFinD 返回的列名 (除"日期"外)
    query: str


META: dict[str, MacroMeta] = {
    "PMI":   MacroMeta("PMI",   "%",  "M", 1,  "制造业PMI",               IFIND_QUERIES["PMI"]),
    "CPI":   MacroMeta("CPI",   "%",  "M", 10, "CPI:当月同比",            IFIND_QUERIES["CPI"]),
    "M2":    MacroMeta("M2",    "%",  "M", 12, "M2(货币和准货币):同比",    IFIND_QUERIES["M2"]),
    "CN10Y": MacroMeta("CN10Y", "%",  "D", 0,  "中债国债到期收益率:10年",  IFIND_QUERIES["CN10Y"]),
    "US10Y": MacroMeta("US10Y", "%",  "D", 0,  "美国:国债收益率:10年",     IFIND_QUERIES["US10Y"]),
}


def _parse_ifind_response(name: str, data: dict[str, Any]) -> pd.DataFrame:
    """解析 iFinD MCP 返回, 统一为 obs_date / value DataFrame."""
    if "result" not in data:
        raise RuntimeError(f"iFinD {name}: 无 result 字段: {data}")
    for item in data["result"].get("content", []):
        if "text" not in item:
            continue
        text = item["text"]
        j = json.loads(text)
        datas = j.get("data", {}).get("datas", [])
        if not datas:
            continue
        rows = datas[0].get("data", {}).get("data", [])
        if not rows:
            continue
        # rows 是 [[date_str, value], ...] 格式
        df = pd.DataFrame(rows, columns=["obs_date", "value"])
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("obs_date").reset_index(drop=True)
        return df
    raise RuntimeError(f"iFinD {name}: 解析失败, 数据格式未知")


def _add_release_date(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """添加 release_date = obs_date + release_lag_days."""
    lag = RELEASE_LAG_DAYS[name]
    df = df.copy()
    df["release_date"] = df["obs_date"] + timedelta(days=lag)
    return df


def fetch_macro_factor(name: str, use_cache: bool = True) -> pd.DataFrame:
    """拉取 1 个宏观因子, 返回 obs_date/release_date/value DataFrame.

    Args:
        name: PMI / CPI / M2 / CN10Y / US10Y
        use_cache: True 命中 data/ifind_cache/macro/{name}.parquet 则不调 API

    Returns:
        DataFrame: [obs_date, value, release_date]
    """
    if name not in META:
        raise ValueError(f"未知宏观因子: {name}, 候选: {list(META.keys())}")

    cache_path = CACHE_DIR / f"{name}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[cache hit] {name} ({len(df)} rows)")
        return df

    meta = META[name]
    print(f"[iFinD] fetching {name}: {meta.query[:60]}...")
    r = _i_find_call("edb", "get_edb_data", {"query": meta.query})
    if not r.get("ok"):
        raise RuntimeError(f"iFinD {name} 失败: {r.get('error') or r.get('raw')}")
    df = _parse_ifind_response(name, r["data"])
    df = _add_release_date(df, name)
    df.to_parquet(cache_path)
    print(f"[saved] {name}: {len(df)} rows → {cache_path}")
    return df


def fetch_all_macro(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """拉取全部 5 宏观因子."""
    result = {}
    for name in META.keys():
        result[name] = fetch_macro_factor(name, use_cache=use_cache)
    return result


# === PIT 查询函数 (回测用) ===

def get_pit_value(series: pd.DataFrame, T: pd.Timestamp) -> float | None:
    """T 日时点可见的最新宏观值.

    Args:
        series: 来自 fetch_macro_factor 的 DataFrame
        T: 当前回测时间 (T 日收盘后做决策)

    Returns:
        最近一个 release_date <= T 的 value; 无则 None
    """
    released = series[series["release_date"] <= T]
    if released.empty:
        return None
    return float(released.iloc[-1]["value"])


def get_pit_series(series: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """批量 PIT 查询, 给定回测日期序列, 返回每日 PIT 后的最新值.

    Args:
        series: 来自 fetch_macro_factor 的 DataFrame
        dates: 回测日期序列 (T+0 ... T+N)

    Returns:
        pd.Series, index=dates, values=PIT value
        早期无数据时 NaN
    """
    import numpy as np

    # 排序后用 searchsorted 找每个 date 的最近 release_date
    sorted_series = series.sort_values("release_date").reset_index(drop=True)
    rel_dates = sorted_series["release_date"].values
    values = sorted_series["value"].values

    result = []
    for d in dates:
        d_py = d.to_pydatetime() if hasattr(d, "to_pydatetime") else d
        # 找 release_date <= d_py 的最大索引
        idx = np.searchsorted(rel_dates, d_py, side="right") - 1
        if idx < 0:
            result.append(float("nan"))
        else:
            result.append(float(values[idx]))
    return pd.Series(result, index=dates)


__all__ = [
    "RELEASE_LAG_DAYS",
    "IFIND_QUERIES",
    "META",
    "fetch_macro_factor",
    "fetch_all_macro",
    "get_pit_value",
    "get_pit_series",
]

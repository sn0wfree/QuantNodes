# coding=utf-8
"""ETF 净值数据加载模块 (Stage 7 重建).

数据源:
    data/real/etf_nav_{start}_{end}.parquet   # 主面板 (44 列 × ~1820 行)
    data/real/per_etf/{code}.parquet          # per-ETF 缓存 (失败可重拉单支)

主要函数:
    load_etf_nav_panel()  → 加载完整 ETF 净值面板 (44 ETFs)
    load_bond_etf_nav()   → 加载单只国债 ETF (511260)

设计原则:
    - 优先用主面板 parquet, 不存在则拼接 per_etf/*.parquet
    - ffill(limit=5) 与 CICC 原报告实现一致 (见 GAP_ANALYSIS.md §1)
    - 自动附加 511260 (国债 ETF) 如果不在面板中
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# 数据根目录: <project_root>/data/real
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "real"


def _resolve_data_dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


def _panel_path(start: str, end: str, data_dir: Path) -> Path:
    """主面板 parquet 路径."""
    return data_dir / f"etf_nav_{start}_{end}.parquet"


def _per_etf_path(code: str, data_dir: Path) -> Path:
    """per-ETF 缓存路径."""
    return data_dir / "per_etf" / f"{code}.parquet"


def load_etf_nav_panel(
    start: str = "2018-01-01",
    end: str = "2025-07-06",
    data_dir: Path | None = None,
    codes: Iterable[str] | None = None,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """加载 ETF 净值面板.

    Args:
        start: 开始日期 (YYYY-MM-DD), 同时影响主面板文件名匹配
        end:   结束日期 (YYYY-MM-DD)
        data_dir: 数据根目录 (默认 data/real)
        codes: 要加载的子集 codes; None = 全部可用 ETF
        ffill_limit: ffill 限制 (默认 5, 与 CICC 实现一致)

    Returns:
        DataFrame: index=DatetimeIndex (工作日), columns=ETF codes

    Raises:
        FileNotFoundError: 找不到主面板且 per_etf 缓存为空
    """
    data_dir = _resolve_data_dir(data_dir)
    panel_p = _panel_path(start, end, data_dir)

    if panel_p.exists():
        df = pd.read_parquet(panel_p)
    else:
        # Fallback: 从 per_etf 缓存拼接
        df = _assemble_from_per_etf(data_dir, codes)

    if codes is not None:
        codes = list(codes)
        available = [c for c in codes if c in df.columns]
        df = df[available]

    # 自动附加 511260 (国债 ETF) 如果不在面板中
    bond_code = "511260"
    if bond_code not in df.columns:
        bond_series = load_bond_etf_nav(bond_code, data_dir)
        if not bond_series.empty:
            df[bond_code] = bond_series

    df = df.sort_index()
    df = df.ffill(limit=ffill_limit)
    return df


def load_bond_etf_nav(
    code: str = "511260",
    data_dir: Path | None = None,
) -> pd.Series:
    """加载单只国债 ETF 净值.

    Args:
        code: ETF code (默认 511260 国泰 10 年期国债 ETF)
        data_dir: 数据根目录 (默认 data/real)

    Returns:
        pd.Series: index=DatetimeIndex, values=close prices
        (若数据不存在则返回空 Series)
    """
    data_dir = _resolve_data_dir(data_dir)
    per_path = _per_etf_path(code, data_dir)

    if per_path.exists():
        df = pd.read_parquet(per_path)
        if "close" in df.columns:
            return df["close"].rename(code)

    # Fallback: 尝试从主面板提取
    for panel_p in sorted(data_dir.glob("etf_nav_*.parquet")):
        try:
            df = pd.read_parquet(panel_p)
            if code in df.columns:
                return df[code].rename(code)
        except Exception:
            continue

    return pd.Series(dtype=float, name=code)


def _assemble_from_per_etf(
    data_dir: Path,
    codes: Iterable[str] | None,
) -> pd.DataFrame:
    """从 per_etf/*.parquet 拼接主面板."""
    per_dir = data_dir / "per_etf"
    if not per_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {per_dir}")

    parquet_files = sorted(per_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"per_etf 缓存为空: {per_dir}")

    series_map = {}
    for path in parquet_files:
        code = path.stem
        if codes is not None and code not in set(codes):
            continue
        try:
            df = pd.read_parquet(path)
            if "close" in df.columns and not df.empty:
                series_map[code] = df["close"].rename(code)
        except Exception:
            continue

    if not series_map:
        raise FileNotFoundError(f"per_etf 无可用数据: {per_dir}")

    panel = pd.concat(series_map.values(), axis=1).sort_index()
    return panel


def list_available_etfs(data_dir: Path | None = None) -> list[str]:
    """列出所有可用的 ETF codes (从主面板或 per_etf 缓存)."""
    data_dir = _resolve_data_dir(data_dir)
    panel_p = sorted(data_dir.glob("etf_nav_*.parquet"))
    if panel_p:
        try:
            df = pd.read_parquet(panel_p[-1])
            return list(df.columns)
        except Exception:
            pass

    per_dir = data_dir / "per_etf"
    if per_dir.exists():
        return sorted(p.stem for p in per_dir.glob("*.parquet"))
    return []


def get_fetch_status(data_dir: Path | None = None) -> dict:
    """读取 fetch_log.json, 返回 ETF 拉取状态."""
    import json

    data_dir = _resolve_data_dir(data_dir)
    log_path = data_dir / "fetch_log.json"
    if not log_path.exists():
        return {"fetched": {}, "failed": [], "fetched_count": 0, "failed_count": 0}
    with open(log_path) as f:
        return json.load(f)


__all__ = [
    "DEFAULT_DATA_DIR",
    "load_etf_nav_panel",
    "load_bond_etf_nav",
    "list_available_etfs",
    "get_fetch_status",
]

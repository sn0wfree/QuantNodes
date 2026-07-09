# coding=utf-8
"""统一 ETF 池数据加载器.

合并:
- 44 只主池 NAV (etf_nav_*.parquet)
- 8 只 SmartBeta NAV (etf_nav_smartbeta_*.parquet)
- 44 只 OHLCV 前复权 (etf_ohlcv_*_adjusted.parquet)
- 额外科创/创业板 ETF

输出: 52 只 ETF 统一 close 面板 + OHLCV 面板
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

REPO = Path("/home/ll/Public/QuantNodes")
DATA_DIR = REPO / "data" / "real"

# 主池 44 只 (from common/universe.py)
MAIN_44 = [
    "159740", "159766", "159901", "159915", "159920", "159928", "159941",
    "159981", "159985", "159996", "161226", "510050", "510300", "510500",
    "510900", "511260", "512000", "512010", "512120", "512170", "512200",
    "512400", "512480", "512660", "512690", "512760", "512800", "512880",
    "512980", "513010", "513050", "513100", "513300", "513500", "513520",
    "513880", "515030", "515050", "515220", "515790", "515880", "518800",
    "518880", "588000",
]

# SmartBeta 额外 8 只 (only in smartbeta panel)
SMARTBETA_8 = [
    "510880",  # 华泰柏瑞红利
    "512890",  # 红利低波
    "512260",  # 300低波
    "515900",  # 中证质量
    "512040",  # 国泰价值
    "159786",  # 现金流
    "515080",  # 中信红利
    "515100",  # 红利低波100
]

# 额外科创/创业板 ETF (用户要求)
EXTRA_STAR_CHINEXT = [
    # 科创板
    "588050",  # 科创50ETF (华夏)
    "588100",  # 科创信息ETF
    "588160",  # 科创芯片ETF
    # 创业板
    "159952",  # 创业50ETF (华夏)
    "159974",  # 创业板动量ETF
]

# 全部 52 只 = 44 + 8 SmartBeta
ALL_52 = MAIN_44 + SMARTBETA_8


@dataclass
class UnifiedData:
    """统一数据容器."""
    close_52: pd.DataFrame       # 52 只 close 面板 (date × code)
    close_60: pd.DataFrame       # 60 只 close 面板 (52 + 8 extra)
    ohlcv_44: pd.DataFrame       # 44 只 OHLCV 面板 (MultiIndex columns)
    ohlcv_60: pd.DataFrame       # 60 只 OHLCV 面板
    date_range: tuple[str, str]  # (start, end)
    n_days: int


def load_unified_data(
    start: str = "2018-01-01",
    end: str = "2026-06-30",
) -> UnifiedData:
    """加载并合并统一数据."""
    print("[unified] 加载主池 NAV (44 只) ...")
    nav_main = pd.read_parquet(DATA_DIR / "etf_nav_2018-01-01_2026-06-30.parquet")

    print("[unified] 加载 SmartBeta NAV (12 只) ...")
    nav_sb = pd.read_parquet(DATA_DIR / "etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")

    print("[unified] 加载 OHLCV 前复权 (44 只) ...")
    ohlcv_raw = pd.read_parquet(DATA_DIR / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    ohlcv_44 = ohlcv_raw.loc[start:end]
    close_from_ohlcv = ohlcv_44.xs("close", axis=1, level=1)

    # === 构建 52 只 close 面板 ===
    # 主池 44 只: 优先用 OHLCV 的 close (前复权), 回退到 NAV
    codes_52 = list(set(MAIN_44 + SMARTBETA_8))
    close_52 = pd.DataFrame(index=nav_main.index, dtype=float)

    for code in codes_52:
        if code in close_from_ohlcv.columns:
            # OHLCV 前复权 close (已修正拆合股)
            close_52[code] = close_from_ohlcv[code].reindex(nav_main.index)
        elif code in nav_main.columns:
            close_52[code] = nav_main[code]
        elif code in nav_sb.columns:
            close_52[code] = nav_sb[code]
        else:
            print(f"  [WARN] {code} 无数据, 跳过")

    close_52 = close_52.dropna(how="all")
    close_52 = close_52.loc[start:end]

    # === 构建 60 只 close 面板 (含额外科创/创业板) ===
    codes_60_extra = [c for c in EXTRA_STAR_CHINEXT if c in nav_main.columns or c in nav_sb.columns]
    close_extra = pd.DataFrame(index=nav_main.index, dtype=float)
    for code in codes_60_extra:
        if code in nav_main.columns:
            close_extra[code] = nav_main[code]
        elif code in nav_sb.columns:
            close_extra[code] = nav_sb[code]

    close_60 = pd.concat([close_52, close_extra], axis=1)
    close_60 = close_60.dropna(how="all").loc[start:end]

    # === OHLCV 60 只 ===
    # 对于额外的 ETF, 从 close 面板构造 (只有 close, 其他字段 NaN)
    ohlcv_60 = ohlcv_44.copy()
    for code in codes_60_extra:
        if code not in ohlcv_44.columns.get_level_values(0):
            if code in close_60.columns:
                for field in ["open", "high", "low"]:
                    ohlcv_60[(code, field)] = np.nan
                ohlcv_60[(code, "close")] = close_60[code].reindex(ohlcv_60.index)
                ohlcv_60[(code, "volume")] = np.nan

    n_days = len(close_52)
    print(f"[unified] 52 只 close: {close_52.shape}, 日期 {close_52.index[0]} ~ {close_52.index[-1]}")
    print(f"[unified] 60 只 close: {close_60.shape}")
    print(f"[unified] OHLCV: {ohlcv_44.shape}")

    return UnifiedData(
        close_52=close_52,
        close_60=close_60,
        ohlcv_44=ohlcv_44,
        ohlcv_60=ohlcv_60,
        date_range=(start, end),
        n_days=n_days,
    )

# coding=utf-8
"""AllWeather — 38 资产统一面板加载.

数据源:
    - 期货 29: DuckDB continuous_main_{product}_daily → close 序列
    - ETF 9: parquet (主面板 + smartbeta 合并) → NAV 序列

输出: data/allweather/panel_v1.parquet
    - columns: 38 资产 (futures_{P} 前缀 + etf_{code} 前缀, 避免重名)
    - index: 交易日 (DatetimeIndex, 无时区)
    - values: close/NAV 价格

设计要点:
    - 期货和 ETF 用前缀区分, 防止重名 (例如 IF 期货与某 ETF)
    - 起始日取交集, 池内资产按其可用日期渐入池 (HRP 天然支持)
    - 终点取最早停止更新日
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import pandas as pd

from .config import (
    DUCKDB_PATH,
    ETF_CODES,
    ETF_NAV_PATH,
    ETF_SMARTBETA_PATH,
    END_DATE,
    FUTURES_PRODUCTS,
    OUTPUT_DIR,
    PANEL_OUTPUT,
    START_DATE,
)

warnings.filterwarnings("ignore")


# 列名加前缀避免冲突
def _fcol(p: str) -> str:
    return f"fut_{p.lower()}"


def _ecol(c: str) -> str:
    return f"etf_{c}"


def load_futures_panel(products: list[str], duckdb_path: Path) -> pd.DataFrame:
    """从 DuckDB continuous_main_*_daily 加载 29 期货 close 序列.

    Args:
        products: 期货产品代码列表, e.g. ['AU','AG','CU',...]
        duckdb_path: DuckDB 文件路径

    Returns:
        DataFrame (T x 29), columns=fut_{product}, values=close
    """
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        frames: dict[str, pd.Series] = {}
        for p in products:
            table = f"continuous_main_{p.lower()}_daily"
            sql = (
                f"SELECT trade_date, close FROM {table} "
                f"WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}' "
                f"ORDER BY trade_date"
            )
            df = con.execute(sql).df()
            if df.empty:
                warnings.warn(f"[{p}] DuckDB 表为空, 跳过", stacklevel=2)
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            s = df.set_index("trade_date")["close"].rename(_fcol(p))
            frames[_fcol(p)] = s
        panel = pd.concat(frames.values(), axis=1)
        panel.index.name = "date"
        return panel
    finally:
        con.close()


def load_etf_panel(
    codes: list[str],
    nav_path: Path,
    smartbeta_path: Path,
) -> pd.DataFrame:
    """从 parquet 加载 9 只 ETF NAV.

    512890 在主面板不在, 从 smartbeta 面板补;
    其余 8 只从主面板取.

    Args:
        codes: ETF 代码列表
        nav_path: 主 NAV 面板路径
        smartbeta_path: smartbeta 面板路径

    Returns:
        DataFrame (T x 9), columns=etf_{code}, values=NAV
    """
    nav_main = pd.read_parquet(nav_path)
    nav_smart = pd.read_parquet(smartbeta_path)

    # 索引已是日期
    if not isinstance(nav_main.index, pd.DatetimeIndex):
        nav_main.index = pd.to_datetime(nav_main.index)
    if not isinstance(nav_smart.index, pd.DatetimeIndex):
        nav_smart.index = pd.to_datetime(nav_smart.index)

    nav_main = nav_main.loc[START_DATE:END_DATE]
    nav_smart = nav_smart.loc[START_DATE:END_DATE]

    frames: dict[str, pd.Series] = {}
    for c in codes:
        if c in nav_main.columns:
            s = nav_main[c].rename(_ecol(c))
            frames[_ecol(c)] = s
        elif c in nav_smart.columns:
            s = nav_smart[c].rename(_ecol(c))
            frames[_ecol(c)] = s
        else:
            warnings.warn(f"[{c}] 两表均无, 跳过", stacklevel=2)

    if not frames:
        raise RuntimeError("未加载到任何 ETF 数据")

    panel = pd.concat(frames.values(), axis=1)
    panel.index.name = "date"
    return panel


def build_unified_panel(
    futures_products: list[str] = FUTURES_PRODUCTS,
    etf_codes: list[str] = ETF_CODES,
    duckdb_path: Path = DUCKDB_PATH,
    etf_nav_path: Path = ETF_NAV_PATH,
    etf_smartbeta_path: Path = ETF_SMARTBETA_PATH,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """构建 38 资产统一面板.

    Returns:
        DataFrame (T x 38), 期货前缀 fut_ + ETF 前缀 etf_
    """
    fut_panel = load_futures_panel(futures_products, duckdb_path)
    etf_panel = load_etf_panel(etf_codes, etf_nav_path, etf_smartbeta_path)

    panel = pd.concat([fut_panel, etf_panel], axis=1)
    panel = panel.loc[start:end]
    panel = panel.sort_index()
    return panel


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_unified_panel()
    panel.to_parquet(PANEL_OUTPUT)

    # 完整性报告
    n_assets = panel.shape[1]
    n_days = panel.shape[0]
    coverage = panel.notna().mean()
    print(f"[AllWeather panel] shape={panel.shape}")
    print(f"  资产数: {n_assets} (期望 38)")
    print(f"  交易日数: {n_days}")
    print(f"  日期范围: {panel.index.min().date()} ~ {panel.index.max().date()}")
    print(f"  整体非空率: {coverage.mean():.2%}")
    print(f"  落盘: {PANEL_OUTPUT}")
    print()
    print("各资产起始日 (HRP 渐入池支持):")
    for c in panel.columns:
        first = panel[c].first_valid_index()
        if first is not None:
            n_valid = panel[c].notna().sum()
            pct = n_valid / n_days
            print(f"  {c:<20} 起始={first.date()}  非空={n_valid}/{n_days} ({pct:.1%})")


if __name__ == "__main__":
    main()
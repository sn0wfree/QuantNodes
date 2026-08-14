# coding=utf-8
"""对冲数据: 月差/Backwardation + VRP 面板 (docs/99/100).

月差: 从 inter_contract_spread 加载 spread_1_2 + 分位.
VRP:  从 vrp_daily 加载 IV−HV + 分位 + 品种分类.

接口:
    load_basis_panel(products, start, end)      → (date × product) 原始 spread
    basis_percentile_panel(panel, window)       → (date × product) [0,1]
    load_vrp_panel(products, start, end)        → (date × product) 原始 VRP
    vrp_percentile_panel(panel, window)         → (date × product) [0,1]
    load_vrp_detail(products, start, end)       → DataFrame (所有列)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .iv_panel import add_percentile

SPREAD_TABLE = "inter_contract_spread"
VRP_TABLE = "vrp_daily"


def load_basis_panel(
    products: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """从 inter_contract_spread 加载月差面板 (date × product), 值 = spread_1_2."""
    import duckdb

    path = duckdb_path or str(Path.home() / "Public" / "DataCache"
                              / "futures_options_daily.duckdb")
    conds = ["spread_1_2 IS NOT NULL"]
    if products:
        plist = ", ".join(f"'{p.upper()}'" for p in products)
        conds.append(f"product IN ({plist})")
    if start:
        conds.append(f"trade_date >= DATE '{start}'")
    if end:
        conds.append(f"trade_date <= DATE '{end}'")
    where = " WHERE " + " AND ".join(conds)
    con = duckdb.connect(path, read_only=True)
    try:
        df = con.execute(
            f"SELECT trade_date, product, spread_1_2 FROM {SPREAD_TABLE}{where}"
        ).df()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    panel = df.pivot_table(index="trade_date", columns="product",
                           values="spread_1_2").sort_index()
    return panel


def basis_percentile_panel(panel: pd.DataFrame, window: int = 252,
                           min_periods: int = 60) -> pd.DataFrame:
    """月差面板 → rolling 252 天 pct-rank 分位 [0,1].

    高分位 = 深贴水 (Backwardation) → 强化保护; 低分位 = 深升水 → 弱化.
    """
    return add_percentile(panel, window=window, min_periods=min_periods)


# ── VRP 面板 ──────────────────────────────────────────────────────────

def load_vrp_panel(
    products: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """从 vrp_daily 加载 VRP 面板 (date × product), 值 = vrp (IV − HV_20)."""
    import duckdb

    path = duckdb_path or str(Path.home() / "Public" / "DataCache"
                              / "futures_options_daily.duckdb")
    conds = ["vrp IS NOT NULL"]
    if products:
        plist = ", ".join(f"'{p.upper()}'" for p in products)
        conds.append(f"product IN ({plist})")
    if start:
        conds.append(f"trade_date >= DATE '{start}'")
    if end:
        conds.append(f"trade_date <= DATE '{end}'")
    where = " WHERE " + " AND ".join(conds)
    con = duckdb.connect(path, read_only=True)
    try:
        df = con.execute(
            f"SELECT trade_date, product, vrp FROM {VRP_TABLE}{where}"
        ).df()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot_table(index="trade_date", columns="product",
                          values="vrp").sort_index()


def vrp_percentile_panel(panel: pd.DataFrame, window: int = 252,
                         min_periods: int = 60) -> pd.DataFrame:
    """VRP 面板 → rolling 252 天 pct-rank 分位 [0,1].

    高分位 = VRP 高 (IV 远高于 HV) → 适合卖出.
    """
    return add_percentile(panel, window=window, min_periods=min_periods)


def load_vrp_detail(
    products: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """从 vrp_daily 加载完整明细 (含 vrp_percentile, product_class 等).

    返回: DataFrame (trade_date × product 列含 vrp/vrp_pct/product_class).
    """
    import duckdb

    path = duckdb_path or str(Path.home() / "Public" / "DataCache"
                              / "futures_options_daily.duckdb")
    conds = ["vrp IS NOT NULL"]
    if products:
        plist = ", ".join(f"'{p.upper()}'" for p in products)
        conds.append(f"product IN ({plist})")
    if start:
        conds.append(f"trade_date >= DATE '{start}'")
    if end:
        conds.append(f"trade_date <= DATE '{end}'")
    where = " WHERE " + " AND ".join(conds)
    con = duckdb.connect(path, read_only=True)
    try:
        df = con.execute(
            f"SELECT * FROM {VRP_TABLE}{where} ORDER BY product, trade_date"
        ).df()
    finally:
        con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


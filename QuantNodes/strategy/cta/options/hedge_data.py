# coding=utf-8
"""对冲数据: 月差/Backwardation 择时信号 (docs/99 §3.3 方案 B).

从 inter_contract_spread 加载逐品种 spread_1_2 (近月 − 次月),
正值 = Backwardation (贴水, 市场脆弱), 负值 = Contango (升水, 市场安全).
构造 rolling 分位面板 (复用 iv_panel.add_percentile) 作为择时信号.

接口:
    load_basis_panel(products, start, end) -> DataFrame (date × product) 原始 spread
    basis_percentile_panel(panel, window)  -> DataFrame (date × product) [0,1]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .iv_panel import add_percentile

SPREAD_TABLE = "inter_contract_spread"


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


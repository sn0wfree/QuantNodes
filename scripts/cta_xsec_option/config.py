# coding=utf-8
"""cta_xsec_option 参数集中配置 (唯一参数源)."""
from __future__ import annotations

import duckdb
import pandas as pd

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"

UNIVERSE = [
    "M", "C", "CF", "SR", "RM", "Y", "P", "OI", "PK", "SM",
    "I", "RB", "SI",
    "TA", "MA", "SA", "V", "EB", "EG", "RU",
    "FG", "SH", "LC", "AG",
    "CU", "AL", "AU",
]

MOM_WINDOWS = (4, 13, 26, 52)

TIER = 3
IS_END = "2022-01-01"
OOS_START = IS_END
TV = 0.08
COST_BPS = 3.0
MIN_GROSS = 0.05
MIN_PRODUCTS = 10
MAX_TURNOVER = 40.0
END = "2025-12-31"

DATA_START = "2018-01-01"


def load_smoothed_panel(con=None, start: str = DATA_START) -> pd.DataFrame:
    """品种池 smoothed close 面板 (品种×日)."""
    own = con is None
    if own:
        con = duckdb.connect(CTA_DB, read_only=True)
    try:
        frames = {}
        for p in UNIVERSE:
            t = f"smoothed_{p.lower()}_daily"
            d = con.execute(
                f"SELECT trade_date, close FROM {t} "
                f"WHERE close IS NOT NULL AND close > 0 ORDER BY trade_date"
            ).df()
            if d.empty:
                continue
            d["trade_date"] = pd.to_datetime(d["trade_date"])
            frames[p] = (
                d.drop_duplicates("trade_date", keep="last")
                .set_index("trade_date")["close"]
            )
        panel = pd.DataFrame(frames)
        panel = panel[panel.index >= start]
        return panel.sort_index()
    finally:
        if own:
            con.close()

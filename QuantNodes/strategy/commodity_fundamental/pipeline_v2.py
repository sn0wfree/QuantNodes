# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/pipeline.py

单品种信号合成流水线: KF → Lasso → 四象限 → 估值 → 拥挤度 → 最终仓位序列.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .kalman_filter import weekly_kalman_center
from .sign_constrained_lasso import (
    walk_forward_lasso, rolling_phy_center, compute_valuation_gap, rolling_zscore,
)
from .inventory_quadrant import inventory_state_for_product
from .valuation_gap import industry_momentum, apply_valuation_layer
from .crowding_simplified import crowding_4w_vol, apply_crowding_override


DB = Path("data/cmf/fundamental.duckdb")


def _close(symbol: str, con: duckdb.DuckDBPyConnection) -> pd.Series:
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc", "M0": "futures_main_close_m"}[symbol]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].rename(symbol)


def _lme_copper(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM inventory_daily WHERE value_col='铜-库存' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("copper_inv")


def _eia_cum(con) -> pd.Series:
    """EIA 库存变动累加 → 库存水平 (近似, 月度累加, 旧实现)."""
    df = con.execute("SELECT trade_date, value FROM inventory_daily WHERE value_col='eia_chg_kbd' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["value"].fillna(0)
    return s.cumsum().rename("eia_cum")


def _eia_weekly_stocks(con) -> pd.Series:
    """EIA WCRSTUS1 — 美国商业原油库存水平 (周度, 1982+, 千桶).

    替代原 _eia_cum 月度累加 (误差大).
    """
    df = con.execute("SELECT trade_date, eia_crude_stocks FROM eia_weekly_inventory ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["eia_crude_stocks"].rename("eia_stocks")


def _dxy(con) -> pd.Series:
    """ICE DXY (合成, akshare currency_boc_safe, 2016+)."""
    df = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col='dxy' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("dxy")


def _tips10y(con) -> pd.Series:
    """FRED DFII10 — 10年期TIPS实际利率 (日度, 2003+)."""
    df = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col='tips10y' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("tips10y")


def _rig(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='us_oil_rig' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("rig")


def _hog(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='hog_index' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("hog")


def _pmi(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='pmi_mfg' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("pmi")


def build_signal(symbol: str, con: duckdb.Connection | None = None) -> pd.DataFrame:
    """单品种三层信号合成 → 返回 DataFrame 含仓位 + 各中间状态.

    Columns: M, p_phy, gap, d_inv, d_price, quadrant_pos, industry_dz, crowded, position
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    close = _close(symbol, con)
    weekly_close = close.resample("W-FRI").last()
    mu_market = weekly_kalman_center(close, q=0.015)

    if symbol == "CU0":
        inv_raw = _lme_copper(con)
        inv_weekly = inv_raw.resample("W-FRI").last()
        inv_dir = -1   # 铜库存↓利多
        industry_raw = _rig(con)
    elif symbol == "SC0":
        inv_raw = _eia_weekly_stocks(con)
        inv_weekly = inv_raw.resample("W-FRI").last()
        inv_dir = -1   # 原油库存↓利多
        industry_raw = _rig(con)
    elif symbol == "M0":
        # 豆粕: akshare库存只3个月, 用价格本身 + OI作为代理
        inv_raw = weekly_close.dropna()
        inv_weekly = inv_raw
        inv_dir = +1   # 价格上涨≈供给紧张(进口驱动)
        industry_raw = _hog(con)
    else:
        raise ValueError(symbol)

    industry_weekly = industry_raw.resample("W-FRI").last()

    inv_weekly_index = inv_weekly.dropna().index
    industry_weekly_index = industry_weekly.dropna().index
    pmi_weekly_index = _pmi(con).dropna().index
    dxy_weekly_index = _dxy(con).resample("W-FRI").last().dropna().index if symbol in ("CU0", "SC0") else mu_market.index
    tips_weekly_index = _tips10y(con).resample("W-FRI").last().dropna().index if symbol in ("CU0", "SC0") else mu_market.index

    earliest_inv = inv_weekly_index.min() if len(inv_weekly_index) else mu_market.index.min()
    earliest_ind = industry_weekly_index.min() if len(industry_weekly_index) else mu_market.index.min()
    earliest = max(mu_market.index.min(), earliest_inv, earliest_ind)
    common = mu_market.index[mu_market.index >= earliest]

    mu_market = mu_market.loc[common]
    inv_aligned = inv_weekly.reindex(common).ffill()
    price_for_quadrant = weekly_close.reindex(common)
    industry_aligned = industry_weekly.reindex(common).ffill()

    feats = {}
    if symbol in ("CU0", "SC0"):
        feats["inv_neg"] = (-rolling_zscore(inv_aligned, 104)).reindex(common)
    feats["rig_neg"] = (-rolling_zscore(industry_aligned, 104)).reindex(common)
    pmi = _pmi(con).resample("W-FRI").last().reindex(common).ffill()
    feats["pmi"] = rolling_zscore(pmi, 104).reindex(common)

    if symbol in ("CU0", "SC0", "M0"):
        dxy_weekly = _dxy(con).resample("W-FRI").last().reindex(common).ffill()
        tips_weekly = _tips10y(con).resample("W-FRI").last().reindex(common).ffill()
        feats["dxy_neg"] = (-rolling_zscore(dxy_weekly, 104)).reindex(common)
        feats["tips_neg"] = (-rolling_zscore(tips_weekly, 104)).reindex(common)

    if symbol == "M0":
        feats["demand"] = rolling_zscore(industry_aligned, 104).reindex(common)
        feats["demand_4wchg"] = rolling_zscore(industry_aligned, 104).diff(4).reindex(common)

    X = pd.concat(feats, axis=1)
    X = X.dropna(how="any")
    common2 = X.index.intersection(mu_market.index)
    X = X.loc[common2]
    y = pd.Series(np.log(mu_market).diff().reindex(common2)).dropna()
    common3 = X.index.intersection(y.index)
    X = X.loc[common3]
    y = y.loc[common3]
    mu_market = mu_market.reindex(common3)

    coefs, oos = walk_forward_lasso(X, y, refit_weeks=26, min_train_weeks=156)
    p_phy = rolling_phy_center(mu_market, oos["oos_pred"] if len(oos) else pd.Series(dtype=float), window=13)
    gap = compute_valuation_gap(mu_market, p_phy)

    inv_state = inventory_state_for_product(
        inv_aligned.reindex(common3).ffill(),
        price_for_quadrant.reindex(common3),
        inv_dir=inv_dir,
    )

    industry_dz = industry_momentum(industry_aligned.reindex(common3).ffill(), z_window=104, delta_window=4)

    base_pos = apply_valuation_layer(
        inv_state["quadrant_pos"],
        gap.reindex(common3),
        industry_dz.reindex(common3),
    )

    weekly_ret = mu_market.pct_change()
    crowding = crowding_4w_vol(weekly_ret, trigger_threshold=0.80, release_threshold=0.50)

    final_pos = apply_crowding_override(base_pos, crowding, crowded_position=0.1)

    return pd.DataFrame({
        "M": mu_market,
        "p_phy": p_phy.reindex(mu_market.index),
        "gap": gap.reindex(mu_market.index),
        "d_inv": inv_state["d_inv"].reindex(mu_market.index),
        "d_price": inv_state["d_price"].reindex(mu_market.index),
        "quadrant_pos": inv_state["quadrant_pos"].reindex(mu_market.index),
        "industry_dz": industry_dz.reindex(mu_market.index),
        "crowded": crowding.reindex(mu_market.index),
        "position": final_pos.reindex(mu_market.index),
    })


__all__ = ["build_signal"]
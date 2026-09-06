# coding: utf-8
"""Group C (商品特定) 变量构造 — P2 实装.

论文 37 变量分三族. 本项目可构造 12 类 (跨品种 → 720+ 列):
  库存族: mom12 / basis / rbasis / basis_term_slope / inv_yoy / inv_to_consume
  对冲族: dOI / dOI_skew  (hp12 缺 CFTC 覆盖)
  流动性族: bmom / iliq / atv / turnover

输出: features_panel.parquet (MultiIndex columns: (product, feature))
PIT 纪律: 见 loader.py (行情天然 PIT OK, 库存按记账日, 宏观滞后 1 期在 features_m.py)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import loader
from .config import (GROUP_C_FEATURES, TARGET_N_PRODUCTS)

LOG = logging.getLogger(__name__)

# ── 库存数据映射 (有库存的品种 + 对应 value_col) ─────────────────────────────
INVENTORY_MAP = {  # product -> value_col (基于 inventory_daily 实际 7 个 value_col)
    "CU": "铜-库存",
    "ZN": "锌-库存",
    "NI": "镍-库存",
    "AL": "铝-库存",
}

# term_structure_daily 中 days_to_expiry 缺值时使用 fallback (天)
FALLBACK_DAYS = {1: 30, 2: 90, 3: 150, 4: 210}


def select_top_products(start: str, end: str, top_n: int = TARGET_N_PRODUCTS) -> list[str]:
    """按 2018-2025 月均成交量选 top_n 品种 (沿用 §20 经验).

    仅保留存在 smoothed_{p}_daily 表的产品 (排除 HO/MO/IO 等期权品种).
    Returns:
        品种代码列表 (大写).
    """
    con = loader.get_market_conn()
    try:
        spec = con.execute("SELECT product FROM contract_specs").fetchall()
        products = [s[0].upper() for s in spec]
        existing = set(
            r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'smoothed_%_daily'"
            ).fetchall()
        )
        sums = []
        for p in products:
            tbl = f"smoothed_{p.lower()}_daily"
            if tbl not in existing:
                continue
            r = con.execute(
                f"SELECT AVG(volume) FROM {tbl} "
                f"WHERE trade_date BETWEEN '{start}' AND '{end}'"
            ).fetchone()
            sums.append((p, r[0] or 0))
        sums.sort(key=lambda x: -x[1])
        out = [p for p, _ in sums[:top_n]]
        LOG.info(f"top {top_n} 产品池 (排除无合约的期权品种): {out}")
        return out
    finally:
        con.close()


def _mom12(close_me: pd.Series) -> pd.Series:
    """12 月动量 (月收益 over 12 月). NaN for 前 12 月."""
    return close_me.pct_change(periods=12)


def _basis(spot_basis: pd.DataFrame, product: str) -> tuple[pd.Series, pd.Series]:
    """现货-近月基差 (basis) 与相对基差 (rbasis)."""
    s = spot_basis[spot_basis["symbol"] == product].copy()
    if s.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    s["trade_date"] = pd.to_datetime(s["trade_date"]).dt.date
    s.set_index("trade_date", inplace=True)
    s.index = pd.DatetimeIndex(s.index)
    me = s.groupby(s.index.to_period("M").to_timestamp("M")).last()
    return me["basis"], me["rbasis"]


def _basis_term_slope(product: str, start: str, end: str) -> pd.Series:
    """期限结构斜率 (rank1 - rank2) / days_diff. 月末截面.

    数据源: term_structure_daily (rank=1 近月, rank=2 次近月).
    """
    con = loader.get_market_conn()
    try:
        r = con.execute(
            f"SELECT trade_date, rank, close, days_to_expiry "
            f"FROM term_structure_daily "
            f"WHERE product='{product}' AND rank<=2 "
            f"AND trade_date BETWEEN '{start}' AND '{end}' "
            f"ORDER BY trade_date, rank"
        ).fetchall()
        if not r:
            return pd.Series(dtype=float)
        df = pd.DataFrame(r, columns=["trade_date", "rank", "close", "dte"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        wide = df.pivot(index="trade_date", columns="rank", values=["close", "dte"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide.index = pd.DatetimeIndex(wide.index)
        me = wide.groupby(wide.index.to_period("M").to_timestamp("M")).last()
        c1 = me.get("close_1")
        c2 = me.get("close_2")
        d1 = me.get("dte_1").fillna(FALLBACK_DAYS[1])
        d2 = me.get("dte_2").fillna(FALLBACK_DAYS[2])
        dd = (d2 - d1).replace(0, np.nan)
        return (c1 - c2) / dd
    finally:
        con.close()


def _inv_yoy(inv_me: pd.Series) -> pd.Series:
    """库存同比 (本月末 / 12 月前 - 1)."""
    if inv_me.empty:
        return pd.Series(dtype=float)
    return inv_me.pct_change(periods=12)


def _inv_to_consume(inv_me: pd.Series) -> pd.Series:
    """库存消费比近似 (本月末库存 / 12 月平均消费率).

    简化: 用 inv_yoy 的倒数近似 (YoY↑→消费/产量不足 → 库存累积).
    更稳健替代: 仅当有"消费/产量"列时计算; 此处简化为 inv_yoy 的反向.
    """
    yoy = _inv_yoy(inv_me)
    return 1 / (1 + yoy.fillna(0))


def _dOI(oi_me: pd.Series) -> pd.Series:
    """OI 月变化."""
    return oi_me.diff()


def _dOI_skew(oi_daily: pd.Series, lookback: int = 252) -> pd.Series:
    """OI 月变动偏度 (12 月 rolling)."""
    daily_diff = oi_daily.diff()
    skew = daily_diff.rolling(lookback).skew()
    skew.index = pd.DatetimeIndex(skew.index)
    return skew.groupby(skew.index.to_period("M").to_timestamp("M")).last()


def _bmom(vol_daily: pd.Series, ret_daily: pd.Series) -> pd.Series:
    """量价协动 (volume × sign(ret)). 月度 sum."""
    s = vol_daily * np.sign(ret_daily)
    s = s.fillna(0)
    return s.resample("M").sum()


def _iliq(ret_daily: pd.Series, vol_daily: pd.Series) -> pd.Series:
    """Amihud 比率 |ret|/volume. 月度均值."""
    s = (ret_daily.abs() / vol_daily.replace(0, np.nan))
    return s.resample("M").mean()


def _atv(vol_daily: pd.Series, lookback: int = 60) -> pd.Series:
    """平均成交量 (60 日滚动). 月末."""
    return vol_daily.rolling(lookback).mean().resample("M").last()


def _turnover(vol_daily: pd.Series, oi_daily: pd.Series) -> pd.Series:
    """换手率 volume/OI. 月度均值."""
    s = vol_daily / oi_daily.replace(0, np.nan)
    return s.resample("M").mean()


def _load_product_oi_volume(products: list[str], start: str, end: str) -> pd.DataFrame:
    """加载产品日频 OI + volume 平表 (用于流动性族 + dOI)."""
    con = loader.get_market_conn()
    try:
        frames = []
        for p in products:
            tbl = f"smoothed_{p.lower()}_daily"
            r = con.execute(
                f"SELECT trade_date, open_interest, volume "
                f"FROM {tbl} WHERE trade_date BETWEEN '{start}' AND '{end}'"
            ).fetchall()
            if not r:
                continue
            df = pd.DataFrame(r, columns=["trade_date", f"{p}_oi", f"{p}_vol"])
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            frames.append(df.set_index("trade_date"))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1).sort_index()
        out.index = pd.DatetimeIndex(out.index)
        return out
    finally:
        con.close()


def _load_spot_basis(start: str, end: str) -> pd.DataFrame:
    """加载 spot_basis_daily (全品种)."""
    con = loader.get_market_conn()
    try:
        r = con.execute(
            f"SELECT trade_date, symbol, near_basis, near_basis_rate "
            f"FROM spot_basis_daily "
            f"WHERE trade_date BETWEEN '{start}' AND '{end}'"
        ).fetchall()
        if not r:
            return pd.DataFrame()
        df = pd.DataFrame(r, columns=["trade_date", "symbol", "basis", "rbasis"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        return df
    finally:
        con.close()


def build_group_c(products: list[str], start: str, end: str) -> pd.DataFrame:
    """构造 Group C 12 类变量 → 60 商品月频面板.

    Returns:
        DataFrame: columns=MultiIndex (product, feature), index=MonthEnd.
    """
    LOG.info(f"P2 Group C: {len(products)} 产品, {start} ~ {end}")
    close_daily = loader.load_smoothed_close(products, start, end)
    if close_daily.empty:
        raise RuntimeError("smoothed close 加载为空, 检查产品池 / 数据库")
    oi_vol = _load_product_oi_volume(products, start, end)
    spot_basis = _load_spot_basis(start, end)

    # 月末 close / 收益 / OI / volume
    close_me = loader.month_end_panel(close_daily)
    ret_daily = close_daily.pct_change()
    oi_me_dict, vol_me_dict = {}, {}
    for p in products:
        if f"{p}_oi" in oi_vol.columns:
            col_oi = f"{p}_oi"
            oi_me_dict[p] = loader.month_end_panel(
                oi_vol[[col_oi]].rename(columns={col_oi: p}))[p]
            col_vol = f"{p}_vol"
            vol_me_dict[p] = loader.month_end_panel(
                oi_vol[[col_vol]].rename(columns={col_vol: p}))[p]

    columns = {}
    for p in products:
        c_me = close_me[p] if p in close_me.columns else pd.Series(dtype=float)
        r_d = ret_daily[p] if p in ret_daily.columns else pd.Series(dtype=float)
        oi_d = oi_vol[f"{p}_oi"] if f"{p}_oi" in oi_vol.columns else pd.Series(dtype=float)
        vol_d = oi_vol[f"{p}_vol"] if f"{p}_vol" in oi_vol.columns else pd.Series(dtype=float)

        # 库存族
        if "mom12" in GROUP_C_FEATURES:
            columns[(p, "mom12")] = _mom12(c_me)
        if "basis" in GROUP_C_FEATURES or "rbasis" in GROUP_C_FEATURES:
            b, rb = _basis(spot_basis, p)
            if "basis" in GROUP_C_FEATURES:
                columns[(p, "basis")] = b
            if "rbasis" in GROUP_C_FEATURES:
                columns[(p, "rbasis")] = rb
        if "basis_term_slope" in GROUP_C_FEATURES:
            columns[(p, "basis_term_slope")] = _basis_term_slope(p, start, end)
        if "inv_yoy" in GROUP_C_FEATURES or "inv_to_consume" in GROUP_C_FEATURES:
            if p in INVENTORY_MAP:
                vc = INVENTORY_MAP[p]
                inv_me = loader.load_inventory_monthly(vc, start, end)
                if "inv_yoy" in GROUP_C_FEATURES:
                    columns[(p, "inv_yoy")] = _inv_yoy(inv_me)
                if "inv_to_consume" in GROUP_C_FEATURES:
                    columns[(p, "inv_to_consume")] = _inv_to_consume(inv_me)
            else:
                columns[(p, "inv_yoy")] = pd.Series(dtype=float)
                columns[(p, "inv_to_consume")] = pd.Series(dtype=float)

        # 对冲族 (dOI / dOI_skew)
        if "dOI" in GROUP_C_FEATURES:
            columns[(p, "dOI")] = _dOI(oi_me_dict.get(p, pd.Series(dtype=float)))
        if "dOI_skew" in GROUP_C_FEATURES:
            columns[(p, "dOI_skew")] = _dOI_skew(oi_d)

        # 流动性族
        if "bmom" in GROUP_C_FEATURES:
            columns[(p, "bmom")] = _bmom(vol_d, r_d)
        if "iliq" in GROUP_C_FEATURES:
            columns[(p, "iliq")] = _iliq(r_d, vol_d)
        if "atv" in GROUP_C_FEATURES:
            columns[(p, "atv")] = _atv(vol_d)
        if "turnover" in GROUP_C_FEATURES:
            columns[(p, "turnover")] = _turnover(vol_d, oi_d)

    panel = pd.DataFrame(columns)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["product", "feature"])
    panel = panel.sort_index(axis=1).sort_index()
    return panel

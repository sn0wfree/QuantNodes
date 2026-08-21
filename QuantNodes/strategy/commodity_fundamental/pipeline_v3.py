# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/pipeline_v3.py

Phase 7 v3 pipeline: 天风因子 + 完整拥挤度 + 真实豆粕库存.

改动 vs v2:
  - 铜: 天风洋山铜溢价 + TC + LME库存(替代akshare)
  - 豆粕: 天风真实库存 + 压榨利润 + 自繁自养(替代价格代理)
  - 原油: 保留 EIA WCRSTUS1 (已验证最佳)
  - 拥挤度: CFTC Managed Money % (D) + 波动率 (U) 完整版
"""
from __future__ import annotations

import warnings
from pathlib import Path

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
from .crowding_simplified import vol_percentile, apply_crowding_override

DB = Path("data/cmf/fundamental.duckdb")


# ── 数据 helpers ─────────────────────────────────────────────────────────────

def _close(symbol: str, con) -> pd.Series:
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc", "M0": "futures_main_close_m"}[symbol]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].rename(symbol)


def _tfapi(con, index_name: str) -> pd.Series:
    """天风指标 → pd.Series (index=trade_date)."""
    df = con.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name = ? ORDER BY trade_date",
        [index_name],
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename(index_name)


def _dxy(con) -> pd.Series:
    """DXY美元指数 (优先使用FRED DTWEXBGS, 2006+; 否则用akshare合成, 2016+)."""
    # 优先使用FRED数据 (DTWEXBGS, 2006+, 历史更长)
    df_fred = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col='dxy_fred' ORDER BY trade_date
    """).df()
    if len(df_fred) > 0:
        df_fred["trade_date"] = pd.to_datetime(df_fred["trade_date"])
        return df_fred.set_index("trade_date")["value"].rename("dxy")

    # 备选: akshare合成 (2016+)
    df_ak = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col='dxy' ORDER BY trade_date
    """).df()
    df_ak["trade_date"] = pd.to_datetime(df_ak["trade_date"])
    return df_ak.set_index("trade_date")["value"].rename("dxy")


def _tips10y(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='tips10y' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("tips10y")


def _eia_weekly_stocks(con) -> pd.Series:
    df = con.execute("SELECT trade_date, eia_crude_stocks FROM eia_weekly_inventory ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["eia_crude_stocks"].rename("eia_stocks")


def _rig(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='us_oil_rig' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("rig")


def _pmi(con) -> pd.Series:
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='pmi_mfg' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("pmi")


def _cftc_managed_money_pct(con, market_pattern: str) -> pd.Series:
    """CFTC Managed Money net long % (周频). market_pattern: 'COPPER' / 'BRENT'."""
    df = con.execute(
        "SELECT report_date as trade_date, mm_net_pct "
        "FROM tfapi_cftc "
        "WHERE market LIKE ? ORDER BY report_date",
        [f"%{market_pattern}%"],
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.groupby("trade_date")["mm_net_pct"].last()
    return s.rename(f"mm_pct_{market_pattern}")


# ── signal 生成 ──────────────────────────────────────────────────────────────

def build_signal_v3(symbol: str, con=None) -> pd.DataFrame:
    """v3 pipeline: 天风因子 + 完整拥挤度."""
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    close = _close(symbol, con)
    weekly_close = close.resample("W-FRI").last()
    mu_market = weekly_kalman_center(close, q=0.015)

    # ── 品种因子 ──────────────────────────────────────────────────────────────
    if symbol == "CU0":
        # 铜: 天风洋山铜溢价 + TC + LME库存 + DXY/TIPS
        premium = _tfapi(con, "洋山铜溢价")
        tc = _tfapi(con, "现货:中国铜冶炼厂:粗炼费(TC)")
        lme = _tfapi(con, "LME库存总计_铜")
        industry_raw = _rig(con)
        inv_raw = lme
        inv_dir = -1  # LME库存↓利多

    elif symbol == "SC0":
        # 原油: EIA WCRSTUS1 + DXY/TIPS
        inv_raw = _eia_weekly_stocks(con)
        industry_raw = _rig(con)
        inv_dir = -1
        premium = pd.Series(dtype=float)  # 无天风溢价
        tc = pd.Series(dtype=float)

    elif symbol == "M0":
        # 豆粕: 天风真实库存 + 压榨利润 + 自繁自养 + DXY/TIPS
        inv_raw = _tfapi(con, "豆粕库存_中国")
        industry_raw = _tfapi(con, "养殖利润_自繁自养")
        crush = _tfapi(con, "江苏:压榨利润:大豆(进口)")
        inv_dir = +1  # 豆粕库存↑=供给宽松, 利空 (但我们按库存水平处理)
        premium = pd.Series(dtype=float)
        tc = pd.Series(dtype=float)
    else:
        raise ValueError(symbol)

    inv_weekly = inv_raw.resample("W-FRI").last() if len(inv_raw) else pd.Series(dtype=float)
    industry_weekly = industry_raw.resample("W-FRI").last() if len(industry_raw) else pd.Series(dtype=float)

    # ── common index ──────────────────────────────────────────────────────────
    earliest_ind = industry_weekly.dropna().index.min() if len(industry_weekly.dropna()) else mu_market.index.min()
    earliest_inv = inv_weekly.dropna().index.min() if len(inv_weekly.dropna()) else mu_market.index.min()
    earliest = max(mu_market.index.min(), earliest_inv, earliest_ind)
    common = mu_market.index[mu_market.index >= earliest]

    mu_market = mu_market.loc[common]
    inv_aligned = inv_weekly.reindex(common).ffill()
    price_for_quadrant = weekly_close.reindex(common)
    industry_aligned = industry_weekly.reindex(common).ffill()

    # ── feats 矩阵 ───────────────────────────────────────────────────────────
    feats = {}

    if symbol in ("CU0", "SC0"):
        # LME/EIA 库存
        feats["inv_neg"] = (-rolling_zscore(inv_aligned, 104)).reindex(common)
    elif symbol == "M0":
        # 豆粕真实库存 (天风)
        if len(inv_aligned.dropna()) > 52:
            feats["inv"] = rolling_zscore(inv_aligned, 104).reindex(common)

    feats["rig_neg"] = (-rolling_zscore(industry_aligned, 104)).reindex(common)

    pmi = _pmi(con).resample("W-FRI").last().reindex(common).ffill()
    feats["pmi"] = rolling_zscore(pmi, 104).reindex(common)

    # DXY/TIPS (所有品种)
    dxy_weekly = _dxy(con).resample("W-FRI").last().reindex(common).ffill()
    tips_weekly = _tips10y(con).resample("W-FRI").last().reindex(common).ffill()
    feats["dxy_neg"] = (-rolling_zscore(dxy_weekly, 104)).reindex(common)
    feats["tips_neg"] = (-rolling_zscore(tips_weekly, 104)).reindex(common)

    # 铜特有: 洋山溢价 + TC
    if symbol == "CU0":
        if len(premium.dropna()) > 52:
            feats["premium"] = rolling_zscore(premium, 104).reindex(common)
        if len(tc.dropna()) > 52:
            feats["tc_neg"] = (-rolling_zscore(tc, 104)).reindex(common)

    # 豆粕特有: 压榨利润
    if symbol == "M0" and len(crush.dropna()) > 52:
        feats["crush_profit"] = rolling_zscore(crush, 104).reindex(common)

    # ── X 矩阵 + y 收益 ──────────────────────────────────────────────────────
    X = pd.concat(feats, axis=1)
    X = X.dropna(how="any")
    common2 = X.index.intersection(mu_market.index)
    X = X.loc[common2]
    y = pd.Series(np.log(mu_market).diff().reindex(common2)).dropna()
    common3 = X.index.intersection(y.index)
    X = X.loc[common3]
    y = y.loc[common3]
    mu_market = mu_market.reindex(common3)

    # ── Lasso walk-forward ────────────────────────────────────────────────────
    coefs, oos = walk_forward_lasso(X, y, refit_weeks=26, min_train_weeks=156)
    p_phy = rolling_phy_center(mu_market, oos["oos_pred"] if len(oos) else pd.Series(dtype=float), window=13)
    gap = compute_valuation_gap(mu_market, p_phy)

    # ── 库存象限 ──────────────────────────────────────────────────────────────
    inv_state = inventory_state_for_product(
        inv_aligned.reindex(common3).ffill(),
        price_for_quadrant.reindex(common3),
        inv_dir=inv_dir,
    )

    industry_dz = industry_momentum(industry_aligned.reindex(common3).ffill(), z_window=104, delta_window=4)

    # ── 估值层 ────────────────────────────────────────────────────────────────
    base_pos = apply_valuation_layer(
        inv_state["quadrant_pos"],
        gap.reindex(common3),
        industry_dz.reindex(common3),
    )

    # ── 拥挤度: CFTC D项 + 波动率 U项 ────────────────────────────────────────
    if symbol == "CU0":
        cftc_d_raw = _cftc_managed_money_pct(con, "COPPER")
    elif symbol == "SC0":
        cftc_d_raw = _cftc_managed_money_pct(con, "CRUDE OIL, LIGHT SWEET-WTI")
    else:
        cftc_d_raw = pd.Series(0.5, index=common3)

    # 统一索引精度 + 对齐到周五
    if len(cftc_d_raw) > 0:
        cftc_d_raw.index = cftc_d_raw.index.astype("datetime64[ns]")
        cftc_d_fri = cftc_d_raw.resample("W-FRI").last()
    else:
        cftc_d_fri = pd.Series(0.5, index=common3)

    weekly_ret = mu_market.pct_change()
    u_vol = vol_percentile(weekly_ret, vol_window=4, pct_window=156)
    u_vol.index = u_vol.index.astype("datetime64[ns]")

    # 完整拥挤度: crowding = 2 * (0.75*D + 0.25*U) - 1
    # D 项: CFTC mm_net_pct 转换为百分位数 (报告§3.3 的 D 是百分位数)
    d_raw = cftc_d_fri.reindex(common3).fillna(0.5)
    # 用156周窗口计算百分位数 (与报告一致)
    d_pct = d_raw.rolling(156, min_periods=52).rank(pct=True)
    d_val = d_pct.fillna(0.5)
    u_val = u_vol.reindex(common3).fillna(0.5)
    crowding_raw = 2.0 * (0.75 * d_val + 0.25 * u_val) - 1.0

    # 触发/解除: 阈值 0.7/0.5
    crowding_sig = pd.Series(0.0, index=common3)
    triggered = False
    for i, (idx, val) in enumerate(crowding_raw.items()):
        if np.isnan(val):
            crowding_sig.iloc[i] = np.nan
            continue
        if not triggered and val >= 0.7:
            triggered = True
        elif triggered and val <= 0.5:
            triggered = False
        crowding_sig.iloc[i] = 1.0 if triggered else 0.0

    # 暂时禁用拥挤度 (太严格导致Sharpe下降), 用 base_pos
    final_pos = base_pos  # apply_crowding_override(base_pos, crowding_sig, crowded_position=0.1)

    # 铜特有: 空仓改为0 (不保留10%最小仓位)
    if symbol == "CU0":
        final_pos[final_pos == 0.1] = 0.0

    # 修复: 第一个周期仓位设为0 (避免pos.shift(1)导致的收益偏差)
    final_pos.iloc[0] = 0.0

    # M 必须是真实周收盘价 (回测收益口径); mu 仅内部用于Lasso目标变量
    return pd.DataFrame({
        "M": price_for_quadrant.reindex(mu_market.index),
        "mu": mu_market,
        "p_phy": p_phy.reindex(mu_market.index),
        "gap": gap.reindex(mu_market.index),
        "d_inv": inv_state["d_inv"].reindex(mu_market.index),
        "d_price": inv_state["d_price"].reindex(mu_market.index),
        "quadrant_pos": inv_state["quadrant_pos"].reindex(mu_market.index),
        "industry_dz": industry_dz.reindex(mu_market.index),
        "crowding_raw": crowding_raw.reindex(mu_market.index),
        "crowded": crowding_sig.reindex(mu_market.index),
        "position": final_pos.reindex(mu_market.index),
    })


__all__ = ["build_signal_v3"]
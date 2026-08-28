"""scripts/cmf/option_utils.py — 期权数据加载和工具函数 (共享模块)."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import calendar
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/cmf/fundamental.duckdb")
OPTION_DB = Path.home() / "Public" / "DataCache" / "futures_options_daily.duckdb"


def load_options():
    """加载SHFE黄金期权数据 (2020-2025)."""
    con = duckdb.connect(str(OPTION_DB), read_only=True)
    opt = con.execute("""
        SELECT trade_date, underlying, option_type, strike_price,
               close, volume, open_interest
        FROM shfe_au_options_daily
        WHERE trade_date >= '2020-01-01' AND trade_date <= '2025-12-31'
        ORDER BY trade_date, underlying, option_type, strike_price
    """).df()
    con.close()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"])
    return opt


def calc_metrics(pos, close, start, end, cost_bps=5.0):
    """计算策略核心指标 (年化, Sharpe, MaxDD, Calmar, 换手)."""
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * cost_bps / 10000
    sr = (sr - cost).dropna()
    if len(sr) == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    start_price = c.iloc[0]
    nav = start_price * (1 + sr).cumprod()
    total_ret = nav.iloc[-1] / start_price - 1
    n_y = len(c) / 252
    ar = (1 + total_ret) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    to = sub.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "turnover": to,
            "calmar": ar / abs(dd) if dd != 0 else 0}


def calc_2026(pos, close, start="2026-01-01", end="2026-08-25"):
    """计算2026年表现."""
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    sr = (sub.shift(1) * c.pct_change()).dropna()
    cost = sub.diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()
    return (1 + sr).prod() - 1 if len(sr) > 0 else 0


def next_month_date(date):
    """计算下个月同一天 (处理月末日期)."""
    if date.month == 12:
        next_month = 1
        next_year = date.year + 1
    else:
        next_month = date.month + 1
        next_year = date.year
    last_day = calendar.monthrange(next_year, next_month)[1]
    return pd.Timestamp(year=next_year, month=next_month,
                       day=min(date.day, last_day))


def select_otm_option(options_df, trade_date, option_type, spot_price,
                     otm_pct, min_volume=50):
    """选择OTM期权 (基于行权价和流动性)."""
    daily_opts = options_df[
        (options_df["trade_date"] == trade_date) &
        (options_df["option_type"] == option_type) &
        (options_df["volume"] >= min_volume)
    ]
    if daily_opts.empty:
        return None
    if option_type == "P":
        # Put: 行权价 < spot, 选最接近spot*(1-otm)的
        target = spot_price * (1 - otm_pct)
        valid = daily_opts[daily_opts["strike_price"] <= target]
        if valid.empty:
            return None
        return valid.loc[valid["strike_price"].idxmax()]
    else:
        # Call: 行权价 > spot, 选最接近spot*(1+otm)的
        target = spot_price * (1 + otm_pct)
        valid = daily_opts[daily_opts["strike_price"] >= target]
        if valid.empty:
            return None
        return valid.loc[valid["strike_price"].idxmin()]


def get_current_premium(options_df, trade_date, option_type, spot_price,
                       otm_pct, min_volume=50):
    """获取当前期权价格 (盯市)."""
    return select_otm_option(options_df, trade_date, option_type,
                           spot_price, otm_pct, min_volume)

# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_real_rate.py

黄金实际利率择时策略

核心逻辑:
    实际利率下降 → 黄金持有成本降低 → 做多黄金
    - TIPS收益率: 10年期通胀保值债券收益率 (实际利率代理)
    - RIR: 美联储模型实际利率
    - 月度数据前向填充至日度/周度

策略变体:
    - tips_ma: TIPS < MA(TIPS, N周) → 做多
    - tips_level: TIPS < 阈值 (如0.5%) → 做多
    - rir_negative: RIR < 0 → 做多 (零利率环境)
    - combined: 多个信号投票决定仓位
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.set_index("trade_date")["close"].resample("W-FRI").last().dropna()
    return close


def load_indicator_w(value_col: str, source: str = "gold_fred") -> pd.Series:
    """加载月度/日度指标, 前向填充到周度 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"SELECT trade_date, value FROM fundamentals_daily "
        f"WHERE source='{source}' AND value_col='{value_col}' ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["value"].resample("W-FRI").last().ffill().dropna()
    return s


def compute_real_rate_signal(
    tips_window: int = 26,
    use_rir_filter: bool = False,
    use_level_filter: bool = False,
    tips_level_thresh: float = 0.5,
    variant: str = "tips_ma",
) -> pd.DataFrame:
    """计算实际利率策略信号.

    参数:
        tips_window: TIPS均线窗口 (周)
        use_rir_filter: 是否使用RIR负值过滤
        use_level_filter: 是否使用TIPS水平过滤
        tips_level_thresh: TIPS水平阈值 (%)
        variant: 策略变体
            - "tips_ma": TIPS < MA → 做多
            - "tips_level": TIPS < 阈值 → 做多
            - "rir_negative": RIR < 0 → 做多
            - "combined": 多数投票

    返回: DataFrame with columns:
        - close: 金价
        - tips: TIPS值
        - rir: RIR值
        - tips_ma: TIPS均线
        - signal: 原始信号 (0/1)
        - position: 仓位 (0/1)
    """
    close = load_gold_price_w()
    tips = load_indicator_w("tips")
    rir = load_indicator_w("rir")

    common = close.index.intersection(tips.index)
    close = close.reindex(common)
    tips = tips.reindex(common)
    rir = rir.reindex(common)

    tips_ma = tips.rolling(tips_window).mean()

    if variant == "tips_ma":
        signal = (tips < tips_ma).astype(float)
    elif variant == "tips_level":
        signal = (tips < tips_level_thresh).astype(float)
    elif variant == "rir_negative":
        signal = (rir < 0).astype(float)
    elif variant == "combined":
        s1 = (tips < tips_ma).astype(float)
        s2 = (tips < tips_level_thresh).astype(float)
        s3 = (rir < 0).astype(float)
        avg = (s1 + s2 + s3) / 3
        signal = (avg >= 0.5).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if use_rir_filter and variant != "rir_negative":
        rir_ok = (rir < 1.0).astype(float)
        signal = signal * rir_ok

    position = signal.fillna(0).clip(0, 1)

    return pd.DataFrame({
        "close": close,
        "tips": tips,
        "rir": rir,
        "tips_ma": tips_ma,
        "signal": signal,
        "position": position,
    })


def backtest_real_rate(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测实际利率策略."""
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "avg_pos": 0, "pct_active": 0, "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()
    active = strat_ret[pos.shift(1).reindex(strat_ret.index).fillna(0) > 0]

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "win_rate_active": (active > 0).mean() if len(active) > 0 else 0,
        "avg_pos": pos.mean(),
        "pct_active": (pos > 0).mean(),
        "nav": nav,
    }


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    tips_window: int = 26,
    variant: str = "tips_ma",
):
    """获取周频收益用于组合优化."""
    signal = compute_real_rate_signal(
        tips_window=tips_window, variant=variant
    )
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "load_gold_price_w",
    "load_indicator_w",
    "compute_real_rate_signal",
    "backtest_real_rate",
    "get_weekly_returns",
]
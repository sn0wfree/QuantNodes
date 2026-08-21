# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_model_v6_3.py

铜模型v6.3: 多因子体系 + 波动率目标化

架构:
    1. 因子层: 12个独立因子 (copper_factors.py)
    2. 组内合成: 同组因子加权平均 (copper_factor_groups.py)
    3. 跨组合成: 组间加权平均
    4. 风控层: 波动率目标化

与v6.2对比:
    - v6.2: 3因子投票 (库存/TC/期限结构), Sharpe 0.68
    - v6.3: 12因子多组体系, 预期Sharpe 0.75+
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .copper_factors import compute_all_factors, _load_weekly_close
from .copper_factor_groups import (
    combine_all_groups,
    DEFAULT_GROUP_WEIGHTS,
    DEFAULT_FACTOR_WEIGHTS,
)

DB = Path("data/cmf/fundamental.duckdb")


def build_signal_v6_3(
    symbol: str = "CU0",
    con=None,
    group_weights: dict = None,
    factor_weights: dict = None,
    target_vol: float = 0.12,
) -> pd.DataFrame:
    """v6.3: 多因子合成 + 波动率目标化.

    参数:
        symbol: 期货合约代码
        con: 数据库连接
        group_weights: 组间权重 {group_name: weight}
        factor_weights: 组内因子权重 {group_name: {factor_name: weight}}
        target_vol: 目标年化波动率 (默认0.12)

    返回: DataFrame with columns:
        - M: 周频收盘价
        - position_raw: 原始合成仓位
        - position_sized: 波动率目标化后仓位
        - factor_scores: 各组得分 (可选)
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    close_w = _load_weekly_close(con, symbol)
    close_w = close_w.dropna()

    # 计算所有因子
    factors = compute_all_factors(con)

    # 对齐 index
    common = close_w.index.intersection(factors.index)
    close_w = close_w.reindex(common)
    factors = factors.reindex(common)

    # 跨组合成
    position_raw = combine_all_groups(factors, group_weights, factor_weights)

    # 波动率目标化
    ret_w = close_w.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    position_sized = (position_raw * scale).clip(upper=1.0)

    # 构建输出
    out = pd.DataFrame({
        "M": close_w,
        "position_raw": position_raw,
        "position": position_sized,  # 最终仓位
    })

    return out


def backtest_v6_3(signal: pd.DataFrame, close: pd.Series = None,
                  cost_bps: float = 5.0) -> dict:
    """v6.3回测.

    参数:
        signal: build_signal_v6_3输出
        close: 收盘价 (默认用signal['M'])
        cost_bps: 交易成本 (默认5bps)

    返回: 回测结果字典
    """
    if close is None:
        close = signal["M"]

    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "total_ret": 0, "ann_ret": 0, "ann_vol": 0,
                "sharpe": 0, "max_dd": 0, "calmar": 0, "win_rate": 0,
                "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {"n_weeks": len(strat_ret), "total_ret": total_ret, "ann_ret": ann_ret,
            "ann_vol": ann_vol, "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            "max_dd": max_dd, "calmar": ann_ret / abs(max_dd) if max_dd != 0 else 0,
            "win_rate": (strat_ret > 0).mean(), "nav": nav}


__all__ = ["build_signal_v6_3", "backtest_v6_3"]

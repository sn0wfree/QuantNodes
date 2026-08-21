# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_economy.py

美债期限差经济状态模型 (优化版)

核心逻辑 (基于验证研究):
  1. 2Y变化方向是关键区分因素
     - 2Y↑ (熊陡): +45.2% (最强信号!)
     - 2Y↓ (牛陡): +3.5%
  2. 简化3状态模型 (避免噪声状态)
     - 扩张: 2Y↑+10Y↑
     - 衰退: 2Y↓+10Y↓
  3. 期限差领先1周 (IC=0.0718)
  4. 2Y水平过滤 (高位+17.0%)

数据:
  - DGS10 (10Y收益率)
  - DGS2 (2Y收益率)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_treasury_data(con) -> tuple[pd.Series, pd.Series, pd.Series]:
    """加载美债数据: 10Y, 2Y, 期限差."""
    from fredapi import Fred
    import json

    with open("data/cmf/.keys.json") as f:
        keys = json.load(f)
    fred = Fred(api_key=keys.get("fred_api_key", ""))

    dgs10 = fred.get_series("DGS10", observation_start="2000-01-01")
    dgs2 = fred.get_series("DGS2", observation_start="2000-01-01")

    df_10y = pd.DataFrame({"trade_date": dgs10.index, "dgs10": dgs10.values})
    df_2y = pd.DataFrame({"trade_date": dgs2.index, "dgs2": dgs2.values})
    df_10y["trade_date"] = pd.to_datetime(df_10y["trade_date"])
    df_2y["trade_date"] = pd.to_datetime(df_2y["trade_date"])

    df = df_10y.merge(df_2y, on="trade_date", how="inner")
    df = df.dropna()
    df["spread"] = df["dgs10"] - df["dgs2"]
    df = df.set_index("trade_date")

    spread_w = df["spread"].resample("W-FRI").last()
    dgs10_w = df["dgs10"].resample("W-FRI").last()
    dgs2_w = df["dgs2"].resample("W-FRI").last()

    return dgs10_w, dgs2_w, spread_w


def apply_min_duration_constraint(score: pd.Series, min_weeks: int = 4) -> pd.Series:
    """最小持续时间约束: 经济状态必须持续至少 min_weeks 周.

    规则: 如果状态变化间隔 < min_weeks, 则保持前一个状态
    作用: 减少噪声, 降低交易频率
    """
    constrained = score.copy()
    prev_score = None
    last_change_idx = 0

    for i in range(len(constrained)):
        if i == 0:
            prev_score = constrained.iloc[i]
            continue

        if constrained.iloc[i] != prev_score:
            if i - last_change_idx < min_weeks:
                constrained.iloc[i] = prev_score
            else:
                last_change_idx = i
                prev_score = constrained.iloc[i]

    return constrained


def compute_economy_state(dgs10: pd.Series, dgs2: pd.Series,
                          spread: pd.Series, window: int = 20,
                          min_duration: int = 1) -> pd.DataFrame:
    """计算经济状态 (基于验证研究的优化版).

    核心发现:
      - 2Y变化方向是关键区分因素 (2Y↑年化+45.2%)
      - 2Y水平可作为过滤器 (高位年化+17.0%)

    约束:
      - min_duration: 最小持续时间约束 (默认4周)

    返回: DataFrame with columns [economy_state, economy_score, dgs2_level]
    """
    # 4周变化
    dgs10_change = dgs10.diff(4)
    dgs2_change = dgs2.diff(4)

    # 20周均线 (用于水平过滤)
    dgs2_ma = dgs2.rolling(window).mean()

    # 经济状态判断 (简化3状态)
    # 1. 经济扩张: 2Y↑ + 10Y↑ → +1.0 (年化+18.7%)
    expansion = (dgs2_change > 0) & (dgs10_change > 0)
    # 2. 经济衰退: 2Y↓ + 10Y↓ → -1.0 (年化-0.3%)
    recession = (dgs2_change < 0) & (dgs10_change < 0)
    # 3. 中性: 其他情况 → 0.0

    # 初始分数
    initial_score = pd.Series(0.0, index=dgs10.index, name="economy_score_initial")
    initial_score[expansion] = 1.0
    initial_score[recession] = -1.0

    # 应用最小持续时间约束
    score = apply_min_duration_constraint(initial_score, min_weeks=min_duration)

    # 分配状态
    state = pd.Series("neutral", index=dgs10.index, name="economy_state")
    state[score == 1.0] = "expansion"
    state[score == -1.0] = "recession"

    # 2Y水平过滤 (高位时铜价+17.0%)
    dgs2_level = pd.Series(0, index=dgs10.index, name="dgs2_level")
    dgs2_level[dgs2 > dgs2_ma] = 1  # 高位
    dgs2_level[dgs2 < dgs2_ma] = -1  # 低位

    result = pd.DataFrame({
        "dgs10": dgs10,
        "dgs2": dgs2,
        "spread": spread,
        "dgs10_change": dgs10_change,
        "dgs2_change": dgs2_change,
        "economy_state": state,
        "economy_score": score,
        "dgs2_level": dgs2_level,
    })

    return result


def get_economy_signal(con, min_duration: int = None) -> tuple[pd.Series, pd.Series]:
    """获取经济状态信号.

    参数:
        min_duration: 最小持续时间约束 (周)

    返回: (economy_score, dgs2_level)
    """
    import inspect
    dgs10, dgs2, spread = load_treasury_data(con)
    if min_duration is None:
        # 从compute_economy_state默认参数中读取
        defaults = compute_economy_state.__defaults__
        if defaults is None:
            min_duration = 1
        else:
            # 默认参数顺序是 (window, min_duration)
            min_duration = defaults[1] if len(defaults) > 1 else 1
    state = compute_economy_state(dgs10, dgs2, spread, min_duration=min_duration)
    return state["economy_score"], state["dgs2_level"]


__all__ = ["load_treasury_data", "compute_economy_state", "get_economy_signal", "apply_min_duration_constraint", "min_duration_supported"]

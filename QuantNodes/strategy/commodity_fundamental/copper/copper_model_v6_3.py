# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_model_v6_3.py

铜模型v6.3: 多因子等权 + bull_enhance条件性加仓 + 波动率目标化

架构:
    1. 因子层: 12个独立因子 (copper_factors.py)
    2. 合成层: 等权加权合成仓位
    3. bull_enhance层: 条件性加仓 (牛市+多因子确认)
    4. 风控层: 波动率目标化

与v6.2对比:
    - v6.2: 3因子投票 (库存/TC/期限结构), Sharpe 0.50, MaxDD -22.9%
    - v6.3: 12因子等权 + bull_enhance, Sharpe 0.75, MaxDD -9.7%
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .copper_factors import (
    compute_all_factors,
    compute_lme_z,
    compute_carry,
    compute_cu_crude,
    _load_weekly_close,
)

DB = Path("data/cmf/fundamental.duckdb")


def build_signal_v6_3(
    symbol: str = "CU0",
    con=None,
    target_vol: float = 0.12,
    bull_mom_weeks: int = 26,
    bull_mom_thresh: float = 0.10,
    enhance_threshold: int = 2,
) -> pd.DataFrame:
    """v6.3: 多因子等权 + bull_enhance + 波动率目标化.

    参数:
        symbol: 期货合约代码
        con: 数据库连接
        target_vol: 目标年化波动率 (默认0.12)
        bull_mom_weeks: 牛市判定动量窗口 (默认26周)
        bull_mom_thresh: 牛市判定动量阈值 (默认10%)
        enhance_threshold: 增强条件所需因子数 (默认2)

    返回: DataFrame with columns:
        - M: 周频收盘价
        - position_raw: 等权合成原始仓位
        - position_enhanced: bull_enhance后仓位
        - position: 波动率目标化后最终仓位
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    close_w = _load_weekly_close(con, symbol)
    close_w = close_w.dropna()

    # 计算所有因子
    factors = compute_all_factors(con)

    # 对齐index
    common = close_w.index.intersection(factors.index)
    close_w = close_w.reindex(common)
    factors = factors.reindex(common)

    # 1. 等权合成
    position_raw = factors.mean(axis=1).clip(0, 1)

    # 2. bull_enhance (条件性加仓)
    mom26 = close_w.pct_change(bull_mom_weeks)
    bull = mom26 > bull_mom_thresh

    # 增强条件: 至少2个因子确认
    lme_z = compute_lme_z(con, close_w)
    carry = compute_carry(con, close_w)
    cu_crude_pos = compute_cu_crude(con, close_w)

    factor_score = (
        (lme_z < -0.5).astype(int) +      # 库存低
        (carry > 0.7).astype(int) +        # 强backwardation
        (cu_crude_pos > 0.6).astype(int)   # 铜油比强
    )
    enhance = bull & (factor_score >= enhance_threshold)

    position_enhanced = position_raw.copy()
    position_enhanced[enhance] = np.maximum(position_enhanced[enhance], 0.9)
    position_enhanced.iloc[0] = 0.0

    # 3. 波动率目标化
    ret_w = close_w.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    position_sized = (position_enhanced * scale).clip(upper=1.0)
    position_sized.iloc[0] = 0.0

    # 构建输出
    out = pd.DataFrame({
        "M": close_w,
        "position_raw": position_raw,
        "position_enhanced": position_enhanced,
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

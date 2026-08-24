# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_inventory_momentum.py

铜库存动量信号

基于LME铜库存变化率生成交易信号:
    - 库存下降 → 做多 (供给收紧)
    - 库存上升 → 观望 (供给宽松)

数据: inventory_daily (LME铜库存, 2014+)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_lme_inventory(con=None) -> pd.Series:
    """加载LME铜库存数据.

    返回: 日频库存序列 (index=trade_date)
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, value AS inventory "
        "FROM inventory_daily "
        "WHERE value_col='铜-库存' "
        "ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["inventory"]


def compute_inventory_momentum(
    con=None,
    change_window: int = 4,
    z_window: int = 52,
    z_threshold: float = 0.5,
    use_extreme: bool = True,
    extreme_percentile: float = 0.25,
) -> pd.DataFrame:
    """计算库存动量信号 (含极值判断).

    参数:
        con: 数据库连接
        change_window: 库存变化窗口 (周)
        z_window: z-score窗口
        z_threshold: z-score阈值
        use_extreme: 是否使用极值判断
        extreme_percentile: 极值分位数 (25% = 低库存)

    信号逻辑:
        1. 库存变化: 4周变化率 < 0 → 做多信号
        2. 库存极值: 库存低于历史25分位 → 信号增强
        3. 组合: 两者加权

    返回: DataFrame with columns:
        - inventory: 原始库存
        - inv_change: 库存变化率
        - inv_zscore: 库存z-score (负值=下降=利好)
        - inv_level_pct: 库存水平分位数
        - position: 仓位 [0, 1]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载日频库存, 转周频
    inv = load_lme_inventory(con)
    inv_weekly = inv.resample("W-FRI").last()

    # 库存变化率
    inv_change = inv_weekly.pct_change(change_window)

    # 库存z-score (负值=下降=利好)
    inv_zscore = (inv_change - inv_change.rolling(z_window).mean()) / inv_change.rolling(z_window).std()

    # 库存水平分位数 (低分位=低库存=利好)
    inv_level_pct = inv_weekly.rolling(min(104, len(inv_weekly)//2)).rank(pct=True)  # 2年滚动分位数

    # 生成仓位
    if use_extreme:
        # 极值判断: 库存低 + 变化下降 → 强信号
        # 库存高 + 变化上升 → 弱信号

        # 变化信号: z-score < -threshold → 1.0, > threshold → 0.0
        change_signal = pd.Series(0.5, index=inv_weekly.index)
        change_signal[inv_zscore < -z_threshold] = 1.0
        change_signal[inv_zscore > z_threshold] = 0.0
        mask = (inv_zscore >= -z_threshold) & (inv_zscore <= z_threshold)
        change_signal[mask] = 0.5 - inv_zscore[mask] / (2 * z_threshold)
        change_signal = change_signal.clip(0, 1).fillna(0.5)

        # 极值信号: 库存低分位 → 1.0, 高分位 → 0.0
        level_signal = pd.Series(0.5, index=inv_weekly.index)
        level_signal[inv_level_pct < extreme_percentile] = 1.0      # 库存极低
        level_signal[inv_level_pct > (1 - extreme_percentile)] = 0.0  # 库存极高
        mask = (inv_level_pct >= extreme_percentile) & (inv_level_pct <= (1 - extreme_percentile))
        level_signal[mask] = 1.0 - inv_level_pct[mask]  # 线性插值
        level_signal = level_signal.clip(0, 1).fillna(0.5)

        # 组合: 70%变化信号 + 30%极值信号
        position = (0.7 * change_signal + 0.3 * level_signal).clip(0, 1)
    else:
        # 纯变化信号
        position = pd.Series(0.5, index=inv_weekly.index)
        position[inv_zscore < -z_threshold] = 1.0
        position[inv_zscore > z_threshold] = 0.0
        mask = (inv_zscore >= -z_threshold) & (inv_zscore <= z_threshold)
        position[mask] = 0.5 - inv_zscore[mask] / (2 * z_threshold)
        position = position.clip(0, 1).fillna(0.5)

    return pd.DataFrame({
        "inventory": inv_weekly,
        "inv_change": inv_change,
        "inv_zscore": inv_zscore,
        "inv_level_pct": inv_level_pct,
        "position": position,
    })


def compute_inventory_momentum_v2(
    con=None,
    change_window: int = 4,
    z_threshold: float = 0.3,
    confirm_threshold: int = 2,
) -> pd.DataFrame:
    """计算库存动量信号v2 - 三因子确认.

    三因子:
        1. 铜库存变化: 4周变化率 < 0 = 利好
        2. 铜注册仓单变化: 4周变化率 < 0 = 利好
        3. 铝库存变化: 4周变化率 < 0 = 利好 (跨品种确认)

    信号逻辑:
        三个因子投票, 至少confirm_threshold个确认 → 做多

    参数:
        con: 数据库连接
        change_window: 库存变化窗口 (周)
        z_threshold: z-score阈值 (未使用, 保留兼容)
        confirm_threshold: 确认因子数 (默认2)

    返回: DataFrame with columns:
        - inv_cu: 铜库存
        - inv_cu_change: 铜库存变化率
        - inv_warrant_change: 注册仓单变化率
        - inv_al_change: 铝库存变化率
        - confirm_count: 确认因子数
        - position: 仓位 [0, 1]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载库存数据
    df_inv = pd.read_sql(
        "SELECT trade_date, value_col, value FROM inventory_daily "
        "WHERE value_col IN ('铜-库存', '铜-注册仓单', '铝-库存') "
        "ORDER BY trade_date",
        con,
    )
    df_inv["trade_date"] = pd.to_datetime(df_inv["trade_date"])

    # 转为周频
    inv_cu = df_inv[df_inv["value_col"] == "铜-库存"].set_index("trade_date")["value"]
    inv_cu_w = inv_cu.resample("W-FRI").last()

    inv_warrant = df_inv[df_inv["value_col"] == "铜-注册仓单"].set_index("trade_date")["value"]
    inv_warrant_w = inv_warrant.resample("W-FRI").last()

    inv_al = df_inv[df_inv["value_col"] == "铝-库存"].set_index("trade_date")["value"]
    inv_al_w = inv_al.resample("W-FRI").last()

    # 计算变化率
    inv_cu_change = inv_cu_w.pct_change(change_window)
    inv_warrant_change = inv_warrant_w.pct_change(change_window)
    inv_al_change = inv_al_w.pct_change(change_window)

    # 三因子投票
    confirm_count = (
        (inv_cu_change < 0).astype(int) +
        (inv_warrant_change < 0).astype(int) +
        (inv_al_change < 0).astype(int)
    )

    # 生成仓位
    position = (confirm_count >= confirm_threshold).astype(float)
    position = position.reindex(inv_cu_w.index).fillna(0)

    return pd.DataFrame({
        "inv_cu": inv_cu_w,
        "inv_cu_change": inv_cu_change,
        "inv_warrant_change": inv_warrant_change,
        "inv_al_change": inv_al_change,
        "confirm_count": confirm_count,
        "position": position,
    })


def backtest_inventory_momentum(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测库存动量信号v1."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    common = signal.index.intersection(close_w.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close_w.pct_change().reindex(common)

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "ann_ret": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


def backtest_inventory_momentum_v2(signal: pd.DataFrame, cost_bps: float = 5.0,
                                   dd_threshold: float = -0.12,
                                   target_vol: float = 0.12) -> dict:
    """回测库存动量信号v2 (含止损+波动率目标化).

    参数:
        signal: compute_inventory_momentum_v2输出
        cost_bps: 交易成本
        dd_threshold: 回撤止损阈值 (默认-12%)
        target_vol: 目标波动率 (默认12%)
    """
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    common = signal.index.intersection(close_w.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close_w.pct_change().reindex(common)

    # 波动率目标化
    vol_4w = ret.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    pos = (pos * scale).clip(0, 1)

    # 回测
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    # 止损: 回撤超阈值时清仓
    nav = (1 + strat_ret).cumprod()
    for i in range(1, len(nav)):
        current_dd = nav.iloc[i] / nav.iloc[:i+1].max() - 1
        if current_dd < dd_threshold:
            # 清仓
            strat_ret.iloc[i] = 0

    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "ann_ret": 0, "ann_vol": 0, "sharpe": 0,
                "max_dd": 0, "win_rate": 0, "nav": pd.Series(dtype=float)}

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }
    """回测库存动量信号.

    参数:
        signal: compute_inventory_momentum输出
        cost_bps: 交易成本

    返回: 回测结果字典
    """
    # 加载价格
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    # 对齐
    common = signal.index.intersection(close_w.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close_w.pct_change().reindex(common)

    # 回测
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "ann_ret": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


__all__ = [
    "load_lme_inventory",
    "compute_inventory_momentum",
    "compute_inventory_momentum_v2",
    "backtest_inventory_momentum",
    "backtest_inventory_momentum_v2",
]

# coding=utf-8
"""v8 Jump Model 集成到 v7.14 TV-PR 框架.

三种集成方案:
  A. 市场状态叠加: 保持 v7.14 选股, 用 Jump Model 检测熊市降仓
  B. 仓位调节: 用 Jump Model 的 Bear% 直接调整仓位比例
  C. 混合信号: Jump Model 检测状态 + v7.14 选股 + 组合信号

OOS 测试: 2022-02-17 ~ 2026-06-30 (4.2 年)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_DIR = REPO / "data" / "high_freq_macro"

from .jump_model import jump_model_rolling, JUMP_PENALTY_MAP, TRAIN_WINDOW_MAP, RETRAIN_EVERY_MAP
from .signal_composer import (
    ASSET_CLASSES,
    compute_composite_signal,
    compute_enhanced_signal,
    apply_min_duration,
)


# ============================================================
# 方案 A: 市场状态叠加 (推荐)
# ============================================================
def regime_overlay_weights(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    retrain_every: int = 30,
    min_duration: int = 60,
    reduce_ratio: float = 0.5,
) -> pd.DataFrame:
    """方案 A: 市场状态叠加.

    逻辑:
      1. 保持 v7.14 的周频选股权重
      2. 用 Jump Model 检测市场状态
      3. 当 ≥2 个大类资产 bearish 时, 降仓 reduce_ratio
      4. 当债券也 bearish 时, 进一步降仓

    Parameters:
        weekly_weights: (T_weekly, N) 周频目标权重
        daily_returns: (T_daily, N_etf) 日频 ETF 收益
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口
        retrain_every: 重估频率
        min_duration: 最小避险持续天数
        reduce_ratio: 降仓比例 (0.5 = 降仓 50%)

    Returns:
        adjusted_weights: (T_weekly, N) 调整后的周频权重
    """
    # 1. 对每个资产运行 Jump Model
    asset_signals = {}
    for col in daily_returns.columns:
        returns = daily_returns[col].dropna()
        if len(returns) < train_window:
            continue
        states = jump_model_rolling(
            returns, jump_penalty, train_window, retrain_every
        )
        asset_signals[col] = states

    # 2. 计算复合信号
    composite = compute_composite_signal(asset_signals)

    # 3. 计算增强信号 (需要债券信号)
    bond_assets = [k for k, v in ASSET_CLASSES.items() if v == 'bond' and k in asset_signals]
    if bond_assets:
        bond_signal = asset_signals[bond_assets[0]]
    else:
        bond_signal = pd.Series(0, index=composite.index)

    enhanced = compute_enhanced_signal(composite, bond_signal)

    # 4. 应用最小持续期
    enhanced = apply_min_duration(enhanced, min_duration)

    # 5. 调整权重
    adjusted_weights = weekly_weights.copy()

    # 将日频信号转换为周频 (取每周最后一个交易日的信号)
    weekly_signal = enhanced.resample('W').last().dropna()

    # 对齐日期
    common_dates = adjusted_weights.index.intersection(weekly_signal.index)

    for date in common_dates:
        signal = weekly_signal.loc[date]
        if signal > 0:
            # 降仓 (不归一化, 保持现金比例)
            adjusted_weights.loc[date] = adjusted_weights.loc[date] * (1 - signal)

    return adjusted_weights


# ============================================================
# 方案 B: 仓位调节
# ============================================================
def position_sizing_weights(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    retrain_every: int = 30,
    min_duration: int = 60,
    bear_threshold: float = 0.3,
) -> pd.DataFrame:
    """方案 B: 仓位调节.

    逻辑:
      1. 保持 v7.14 的周频选股权重
      2. 用 Jump Model 计算每个资产的 Bear%
      3. 当 Bear% > bear_threshold 时, 按比例降低该资产权重

    Parameters:
        weekly_weights: (T_weekly, N) 周频目标权重
        daily_returns: (T_daily, N_etf) 日频 ETF 收益
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口
        retrain_every: 重估频率
        min_duration: 最小避险持续天数
        bear_threshold: Bear% 阈值

    Returns:
        adjusted_weights: (T_weekly, N) 调整后的周频权重
    """
    # 1. 对每个资产运行 Jump Model, 计算 Bear%
    asset_bear_pct = {}
    for col in daily_returns.columns:
        returns = daily_returns[col].dropna()
        if len(returns) < train_window:
            continue
        states = jump_model_rolling(
            returns, jump_penalty, train_window, retrain_every
        )
        # 计算滚动 Bear%
        bear_pct = states.rolling(window=60, min_periods=1).mean()
        asset_bear_pct[col] = bear_pct

    # 2. 调整权重
    adjusted_weights = weekly_weights.copy()

    for date in adjusted_weights.index:
        for col in adjusted_weights.columns:
            if col in asset_bear_pct:
                # 找到该日期的 Bear%
                bear_series = asset_bear_pct[col]
                # 使用 ffill 找到最近的 bear%
                if date in bear_series.index:
                    bear_pct = bear_series.loc[date]
                else:
                    # 找到 date 之前的最近值
                    before = bear_series[bear_series.index <= date]
                    if len(before) > 0:
                        bear_pct = before.iloc[-1]
                    else:
                        bear_pct = 0.0

                if bear_pct > bear_threshold:
                    # 按比例降低权重 (不归一化, 保持现金比例)
                    reduce_factor = 1 - (bear_pct - bear_threshold) / (1 - bear_threshold)
                    adjusted_weights.loc[date, col] *= reduce_factor

    return adjusted_weights


# ============================================================
# 方案 C: 混合信号
# ============================================================
def hybrid_signal_weights(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    retrain_every: int = 30,
    min_duration: int = 60,
    regime_weight: float = 0.5,
) -> pd.DataFrame:
    """方案 C: 混合信号.

    逻辑:
      1. 保持 v7.14 的周频选股权重
      2. 用 Jump Model 计算市场状态信号
      3. 用 v7.14 的因子得分计算因子信号
      4. 组合两个信号: final_weight = regime_weight * regime_signal + (1 - regime_weight) * factor_signal

    Parameters:
        weekly_weights: (T_weekly, N) 周频目标权重
        daily_returns: (T_daily, N_etf) 日频 ETF 收益
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口
        retrain_every: 重估频率
        min_duration: 最小避险持续天数
        regime_weight: 市场状态信号权重

    Returns:
        adjusted_weights: (T_weekly, N) 调整后的周频权重
    """
    # 1. 对每个资产运行 Jump Model
    asset_signals = {}
    for col in daily_returns.columns:
        returns = daily_returns[col].dropna()
        if len(returns) < train_window:
            continue
        states = jump_model_rolling(
            returns, jump_penalty, train_window, retrain_every
        )
        asset_signals[col] = states

    # 2. 计算复合信号
    composite = compute_composite_signal(asset_signals)

    # 3. 应用最小持续期
    composite = apply_min_duration(composite, min_duration)

    # 4. 将日频信号转换为周频
    weekly_regime_signal = composite.resample('W').last().dropna()

    # 5. 调整权重
    adjusted_weights = weekly_weights.copy()

    # 对齐日期
    common_dates = adjusted_weights.index.intersection(weekly_regime_signal.index)

    for date in common_dates:
        regime_signal = weekly_regime_signal.loc[date]

        if regime_signal > 0:
            # 市场状态信号: 降仓
            regime_adjustment = 1 - regime_signal

            # 因子信号: 保持原始权重
            factor_weights = adjusted_weights.loc[date]

            # 混合信号 (不归一化, 保持现金比例)
            adjusted_weights.loc[date] = (
                regime_weight * factor_weights * regime_adjustment +
                (1 - regime_weight) * factor_weights
            )

    return adjusted_weights


# ============================================================
# 完整回测函数
# ============================================================
def backtest_v8_integration(
    version: str = "v7.14",
    integration_method: str = "regime_overlay",
    **params,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """v8 Jump Model 集成回测.

    Parameters:
        version: v7 版本号
        integration_method: 集成方式 ("regime_overlay", "position_sizing", "hybrid")
        **params: 其他参数

    Returns:
        nav: 日频 NAV
        weekly_weights: 周频权重
        adjusted_weights: 调整后的权重
    """
    from ..v7.adapters import get_strategy, load_v7_14_data_uniform
    from ..v7.macro_substrategy_v7_6 import (
        V7_6Config, construct_portfolio_components,
    )
    from ..v7.tvpr_estimator import expanding_window_tvpr

    # 1. 加载数据
    X, Y, codes = load_v7_14_data_uniform()

    # 2. 配置
    cfg = V7_6Config(
        top_n=params.get("top_n", 10),
        vol_window=params.get("vol_window", 26),
        max_weight=params.get("max_weight", 0.25),
        lambda_tv=params.get("lambda_tv", 0.15),
        lambda_l1=params.get("lambda_l1", 0.05),
        step=params.get("step", 13),
        method="expanding",
        cost_enabled=False,
    )

    # 3. TV-PR beta 估计
    beta = expanding_window_tvpr(
        Y, X,
        cfg.lambda_tv, cfg.lambda_l1,
        min_history=cfg.min_history,
        step=cfg.step,
    )

    # 4. 构造组合
    shares, prices, weekly_weights = construct_portfolio_components(
        Y, X, beta, cfg,
    )

    # 5. 加载日频收益 (用于 Jump Model)
    daily_returns = pd.read_parquet(HF_DIR / "v7_6_daily_etf_returns.parquet")

    # 6. 应用集成方案
    if integration_method == "regime_overlay":
        adjusted_weights = regime_overlay_weights(
            weekly_weights, daily_returns,
            jump_penalty=params.get("jump_penalty", 50.0),
            train_window=params.get("train_window", 1000),
            retrain_every=params.get("retrain_every", 30),
            min_duration=params.get("min_duration", 60),
            reduce_ratio=params.get("reduce_ratio", 0.5),
        )
    elif integration_method == "position_sizing":
        adjusted_weights = position_sizing_weights(
            weekly_weights, daily_returns,
            jump_penalty=params.get("jump_penalty", 50.0),
            train_window=params.get("train_window", 1000),
            retrain_every=params.get("retrain_every", 30),
            min_duration=params.get("min_duration", 60),
            bear_threshold=params.get("bear_threshold", 0.3),
        )
    elif integration_method == "hybrid":
        adjusted_weights = hybrid_signal_weights(
            weekly_weights, daily_returns,
            jump_penalty=params.get("jump_penalty", 50.0),
            train_window=params.get("train_window", 1000),
            retrain_every=params.get("retrain_every", 30),
            min_duration=params.get("min_duration", 60),
            regime_weight=params.get("regime_weight", 0.5),
        )
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")

    # 7. 计算日频 NAV
    # 用日频收益 × 调整后权重累积 NAV
    # 权重 < 1 的部分隐含为现金 (0% 收益)
    nav = _compute_daily_nav_from_weights(adjusted_weights, daily_returns)

    return nav, weekly_weights, adjusted_weights


def _compute_daily_nav_from_weights(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cost_bp: float = 0.0,
) -> pd.Series:
    """从周频权重和日频收益计算日频 NAV.

    逻辑:
      - 每个交易日找到最近的已生效的周频权重
      - 组合日收益 = sum(weight_i × return_i)
      - 权重 < 1 的部分隐含为现金 (0% 收益)
      - NAV 累积: nav[t] = nav[t-1] * (1 + daily_port_return - cost)
      - 成本模型: 调仓日扣除 turnover × cost_bp / 10000

    Parameters:
        weekly_weights: (T_weekly, N) 周频权重 (DatetimeIndex)
        daily_returns: (T_daily, N) 日频收益 (DatetimeIndex)
        cost_bp: 单边交易成本 (bp), 默认 0

    Returns:
        nav: (T_daily,) 日频 NAV, 起点=1.0
    """
    # 对齐资产列
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建映射: 每个交易日 → 最近的已生效的周频权重
    # 周频权重在周五(t)生成, 生效于下周一(t+1)
    date_to_weights = {}
    for i, wd in enumerate(weekly_dates):
        # 找到 wd 之后的第一个交易日 (生效日)
        after = all_dates[all_dates > wd]
        if len(after) == 0:
            continue
        start = after[0]
        # 生效到下一个调仓日之前
        if i + 1 < len(weekly_dates):
            next_wd = weekly_dates[i + 1]
            before_next = all_dates[all_dates <= next_wd]
            if len(before_next) == 0:
                continue
            end = before_next[-1]
        else:
            end = all_dates[-1]

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_weights[d] = weekly_weights.loc[wd]

    # 计算日频 NAV
    # 中国假期处理: ETF 数据全 NaN 当日, 视为不交易日 (跳过, 与 v7_6 数据行为一致)
    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_weights.get(d)
        if w is not None:
            row = daily_returns.loc[d]
            # 中国假期判断: ETF 收益全 NaN
            if row[common_codes].isna().all():
                # 跳过该日, NAV 不变 (与 v7_6 数据一致)
                nav.iloc[i] = nav.iloc[i - 1]
            else:
                ret = row.fillna(0.0)
                port_ret = float((w * ret).sum())
                # 成本扣除: 调仓日扣除 turnover × cost_bp / 10000 (乘法模型)
                cost_factor = 1.0
                if cost_bp > 0:
                    turnover = float((w - prev_w).abs().sum())
                    cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
                nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
                prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]

    return nav


def smooth_weekly_weights(
    weights_df: pd.DataFrame,
    alpha: float = 0.7,
    min_trade_threshold: float = 0.02,
) -> pd.DataFrame:
    """对周频权重做后处理平滑, 降低换手率.

    逻辑:
      1. EMA 平滑: blended = alpha * new_w + (1-alpha) * prev_w
      2. 最小调仓阈值: |blended - prev_w| < threshold 时忽略
      3. 归一化: 确保权重和 ≤ 1

    Parameters:
        weights_df: (T_weekly, N) 周频权重
        alpha: EMA 系数 (1.0 = 不平滑, 0.0 = 完全平滑)
        min_trade_threshold: 最小调仓阈值 (per asset)

    Returns:
        smoothed: (T_weekly, N) 平滑后的权重
    """
    smoothed = weights_df.copy()
    for t in range(1, len(smoothed)):
        prev_w = smoothed.iloc[t - 1]
        new_w = weights_df.iloc[t]
        # EMA 平滑
        blended = alpha * new_w + (1 - alpha) * prev_w
        # 最小调仓阈值: 忽略微小变化
        diff = blended - prev_w
        diff[diff.abs() < min_trade_threshold] = 0.0
        smoothed.iloc[t] = prev_w + diff
    # 归一化: 确保权重和 ≤ 1
    row_sums = smoothed.sum(axis=1)
    mask = row_sums > 1.0
    smoothed.loc[mask] = smoothed.loc[mask].div(row_sums[mask], axis=0)
    return smoothed


# ============================================================
# 统一接口
# ============================================================
def get_integration_method(method: str) -> callable:
    """获取集成方法."""
    methods = {
        "regime_overlay": regime_overlay_weights,
        "position_sizing": position_sizing_weights,
        "hybrid": hybrid_signal_weights,
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}, available: {list(methods.keys())}")
    return methods[method]


__all__ = [
    "regime_overlay_weights",
    "position_sizing_weights",
    "hybrid_signal_weights",
    "backtest_v8_integration",
    "get_integration_method",
]

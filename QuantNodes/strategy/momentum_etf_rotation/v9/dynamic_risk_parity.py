# coding=utf-8
"""v9 6 因子动态风险平价 — V×A×C×T×R.

完整公式:
    w_i^{final}(t) = w_i^{base}(t) × V_i(t) × A_i(t) × C_i(t) × T_i(t) × R(t)
    归一化: w_i^{final} = w_i^{final} / Σ w_j^{final}

6 个因子:
    V(t): 周期速度 (相位一阶导数) → [0.7, 1.3]
    A(t): 周期加速度 (相位二阶导数) → [0.8, 1.2]
    C(t): 多周期叠加 (短+中+长共振) → [0.6, 1.4]
    T(t): 趋势因子 (动量信号) → [0.7, 1.3]
    R(t): 相关性体制 (危机降仓) → [0.5, 1.0]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_risk_parity_base(returns_df, lookback=260):
    """计算风险平价基础权重.

    参数:
        returns_df: 日收益率 DataFrame
        lookback: 滚动窗口 (日)

    返回:
        weights: DataFrame, 风险平价权重
    """
    vol = returns_df.rolling(lookback).std()
    inv_vol = 1.0 / vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights.fillna(0)


def compute_velocity_adjustment(phases, lookback=26):
    """周期速度调整.

    参数:
        phases: 相位 DataFrame (每个资产的相位)
        lookback: 速度计算窗口

    返回:
        adjustment: 调整系数 DataFrame ∈ [0.7, 1.3]
    """
    velocity = np.gradient(phases, axis=0)

    # 修正相位循环 (0→2π→0)
    velocity = np.where(velocity > np.pi, velocity - 2 * np.pi, velocity)
    velocity = np.where(velocity < -np.pi, velocity + 2 * np.pi, velocity)

    # 平滑
    vel_smooth = pd.DataFrame(velocity, index=phases.index, columns=phases.columns)
    vel_smooth = vel_smooth.rolling(lookback, min_periods=1).mean()

    # tanh 映射 → [0.7, 1.3]
    adjustment = 1.0 + np.tanh(vel_smooth / np.pi) * 0.3
    return adjustment.clip(0.7, 1.3)


def compute_acceleration_adjustment(phases, lookback=26):
    """周期加速度调整.

    参数:
        phases: 相位 DataFrame
        lookback: 加速度计算窗口

    返回:
        adjustment: 调整系数 DataFrame ∈ [0.8, 1.2]
    """
    velocity = np.gradient(phases, axis=0)
    velocity = np.where(velocity > np.pi, velocity - 2 * np.pi, velocity)
    velocity = np.where(velocity < -np.pi, velocity + 2 * np.pi, velocity)

    acceleration = np.gradient(velocity, axis=0)

    acc_smooth = pd.DataFrame(acceleration, index=phases.index, columns=phases.columns)
    acc_smooth = acc_smooth.rolling(lookback, min_periods=1).mean()

    # tanh 映射 → [0.8, 1.2]
    adjustment = 1.0 + np.tanh(acc_smooth / np.pi) * 0.2
    return adjustment.clip(0.8, 1.2)


def compute_cycle叠加_adjustment(phases_short, phases_medium, phases_long):
    """多周期叠加调整.

    参数:
        phases_short: 短周期相位 (Kitchin)
        phases_medium: 中周期相位 (Juglar)
        phases_long: 长周期相位

    返回:
        adjustment: 调整系数 DataFrame ∈ [0.6, 1.4]
    """
    # 三个周期的余弦值之和 / 3 → [-1, 1]
    cos_short = np.cos(phases_short)
    cos_medium = np.cos(phases_medium)
    cos_long = np.cos(phases_long)

    # 加权平均 (短周期权重更高)
    combined = 0.4 * cos_short + 0.35 * cos_medium + 0.25 * cos_long

    # 归一化到 [0, 1]
    normalized = (combined + 1) / 2

    # 映射到 [0.6, 1.4]
    adjustment = 0.6 + normalized * 0.8
    return adjustment.clip(0.6, 1.4)


def compute_correlation_adjustment(returns_df, lookback=260):
    """相关性体制调整.

    参数:
        returns_df: 日收益率 DataFrame
        lookback: 滚动窗口

    返回:
        adjustment: 调整系数 Series (index=日期) ∈ [0.5, 1.0]
    """
    # 滚动相关性 → 每个日期的平均相关系数
    avg_corr_list = []
    dates = returns_df.index[lookback:]

    for i in range(lookback, len(returns_df)):
        window = returns_df.iloc[i-lookback:i]
        corr = window.corr()
        # 上三角均值 (排除对角线)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        avg_corr = corr.values[mask].mean()
        avg_corr_list.append(avg_corr)

    avg_corr = pd.Series(avg_corr_list, index=dates)

    # 相关性越高 → 调整系数越小
    adjustment = 1.0 - (avg_corr - 0.3) * 0.5
    return adjustment.clip(0.5, 1.0)


def compute_dynamic_risk_parity(
    returns_df,
    macro_cycles,
    asset_cycles,
    lookback=260,
):
    """6 因子动态风险平价主入口.

    参数:
        returns_df: 日收益率 DataFrame (13资产)
        macro_cycles: 宏观周期提取结果 (来自 cycle_extractor)
        asset_cycles: 资产周期提取结果 (来自 cycle_extractor)
        lookback: 风险平价窗口

    返回:
        final_weights: DataFrame, 最终权重 (每日)
        components: dict, 各因子贡献
    """
    # 1. 风险平价基础权重
    base_weights = compute_risk_parity_base(returns_df, lookback)

    # 2. 周期速度调整 (用资产周期的 Kitchin 相位)
    # 对齐到资产周期的日期
    asset_idx = asset_cycles[list(asset_cycles.keys())[0]]['cycles']['kitchin']['phase'].index
    base_aligned = base_weights.reindex(asset_idx, method='ffill').fillna(0)

    # 提取相位矩阵
    phases_df = pd.DataFrame(index=asset_idx, columns=returns_df.columns, dtype=float)
    for col in returns_df.columns:
        if col in asset_cycles:
            phases_df[col] = asset_cycles[col]['cycles']['kitchin']['phase'].reindex(asset_idx)

    phases_df = phases_df.fillna(0).astype(float)

    velocity_adj = compute_velocity_adjustment(phases_df)
    acceleration_adj = compute_acceleration_adjustment(phases_df)

    # 3. 多周期叠加调整
    phases_short = phases_df
    phases_medium = pd.DataFrame(index=asset_idx, columns=returns_df.columns, dtype=float)
    phases_long = pd.DataFrame(index=asset_idx, columns=returns_df.columns, dtype=float)

    for col in returns_df.columns:
        if col in asset_cycles:
            phases_medium[col] = asset_cycles[col]['cycles']['juglar']['phase'].reindex(asset_idx)
            phases_long[col] = asset_cycles[col]['cycles']['long_term']['phase'].reindex(asset_idx)

    phases_medium = phases_medium.fillna(0).astype(float)
    phases_long = phases_long.fillna(0).astype(float)

    cycle_adj = compute_cycle叠加_adjustment(phases_short, phases_medium, phases_long)

    # 4. 相关性体制调整
    corr_adj = compute_correlation_adjustment(returns_df, lookback)
    corr_adj = corr_adj.reindex(asset_idx, method='ffill').fillna(1.0)

    # 5. 最终权重 = 基础 × V × A × C × R
    # (趋势因子 T 在 backtest 中根据每日数据单独计算)
    final_weights = base_aligned * velocity_adj * acceleration_adj * cycle_adj
    final_weights = final_weights.div(final_weights.sum(axis=1), axis=0).fillna(0)

    components = {
        'base': base_aligned,
        'velocity': velocity_adj,
        'acceleration': acceleration_adj,
        'cycle叠加': cycle_adj,
        'correlation': corr_adj,
    }

    return final_weights, components

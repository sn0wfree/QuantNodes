# coding=utf-8
"""信号合成 — 复合信号与增强信号.

复合信号: ≥2 个大类资产 bearish → 降仓 50%, ≥3 → 降仓 80%
增强信号: 复合信号 + 债券必须 bearish (确认系统性风险)
强制避险: 最少 60 个交易日

参考: 中金《基于统计跳跃的系统性风险预警模型》(2026-06-22)
"""
from __future__ import annotations

import pandas as pd


# ============================================================
# 资产分类
# ============================================================
ASSET_CLASSES = {
    # 权益
    '510300': 'equity', '510500': 'equity', '510050': 'equity',
    '159915': 'equity', '588000': 'equity', '159901': 'equity',
    '512760': 'equity', '512480': 'equity', '515030': 'equity',
    '515790': 'equity', '512690': 'equity', '512170': 'equity',
    '512010': 'equity', '515050': 'equity', '159928': 'equity',
    '512880': 'equity', '512000': 'equity', '512800': 'equity',
    '515220': 'equity', '512200': 'equity', '512400': 'equity',
    '512660': 'equity', '512980': 'equity', '515880': 'equity',
    '159996': 'equity', '512120': 'equity', '510900': 'equity',
    '159920': 'equity', '513010': 'equity', '513050': 'equity',
    '159740': 'equity', '513100': 'equity', '513300': 'equity',
    '513500': 'equity', '513520': 'equity', '513880': 'equity',
    '159941': 'equity', '510880': 'equity', '512890': 'equity',
    '512260': 'equity', '515900': 'equity', '512040': 'equity',
    '159786': 'equity', '515080': 'equity', '515100': 'equity',
    # 债券
    '511260': 'bond',
    # 商品
    '518880': 'commodity', '518800': 'commodity', '159985': 'commodity',
    '161226': 'commodity', '159981': 'commodity', '159766': 'commodity',
}


# ============================================================
# 复合信号
# ============================================================
def compute_composite_signal(
    asset_signals: dict[str, pd.Series],
    asset_classes: dict[str, str] | None = None,
) -> pd.Series:
    """计算复合信号.

    规则:
    - ≥2 个大类资产 bearish → reduce_ratio = 0.5 (降仓 50%)
    - ≥3 个大类资产 bearish → reduce_ratio = 0.8 (降仓 80%)
    - 否则 → reduce_ratio = 0 (不降仓)

    Parameters:
        asset_signals: 资产代码 → 信号序列 (0=bull, 1=bear)
        asset_classes: 资产代码 → 大类 (equity/bond/commodity)

    Returns:
        pd.Series, 降仓比例 (0, 0.5, 0.8)
    """
    if asset_classes is None:
        asset_classes = ASSET_CLASSES

    # 按大类聚合
    class_signals: dict[str, list[pd.Series]] = {}
    for asset, signal in asset_signals.items():
        cls = asset_classes.get(asset, 'equity')
        if cls not in class_signals:
            class_signals[cls] = []
        class_signals[cls].append(signal)

    # 每个大类取多数信号 (bearish > 50%)
    class_bearish: dict[str, pd.Series] = {}
    for cls, signals in class_signals.items():
        stacked = pd.concat(signals, axis=1)
        class_bearish[cls] = (stacked.mean(axis=1) > 0.5).astype(int)

    # 计算降仓比例
    bearish_count = sum(class_bearish.values())
    if isinstance(bearish_count, pd.Series):
        result = pd.Series(0.0, index=bearish_count.index)
        result[bearish_count >= 3] = 0.8
        result[(bearish_count >= 2) & (bearish_count < 3)] = 0.5
    else:
        result = pd.Series(0.0, index=asset_signals[list(asset_signals.keys())[0]].index)

    return result


# ============================================================
# 增强信号
# ============================================================
def compute_enhanced_signal(
    composite_signal: pd.Series,
    bond_signal: pd.Series,
) -> pd.Series:
    """计算增强信号.

    规则: composite_signal × bond_signal
    只有当债券也 bearish 时才触发避险

    Parameters:
        composite_signal: 复合信号 (降仓比例)
        bond_signal: 债券信号 (0=bull, 1=bear)

    Returns:
        pd.Series, 增强信号 (降仓比例)
    """
    return composite_signal * bond_signal


# ============================================================
# 强制避险持续期
# ============================================================
def apply_min_duration(
    signal: pd.Series,
    min_duration: int = 60,
) -> pd.Series:
    """应用最小避险持续期.

    一旦触发避险 (signal > 0), 必须持续至少 min_duration 个交易日.

    Parameters:
        signal: 原始信号 (降仓比例)
        min_duration: 最小持续天数

    Returns:
        pd.Series, 调整后的信号
    """
    result = signal.copy()
    in_defense = False
    defense_start = 0
    defense_ratio = 0.0

    for i in range(len(result)):
        if not in_defense:
            # 检查是否触发避险
            if result.iloc[i] > 0:
                in_defense = True
                defense_start = i
                defense_ratio = result.iloc[i]
        else:
            # 在避险期间
            if i - defense_start < min_duration:
                # 强制持续
                result.iloc[i] = defense_ratio
            else:
                # 检查是否结束
                if result.iloc[i] == 0:
                    in_defense = False
                    defense_ratio = 0.0
                else:
                    # 更新避险比例
                    defense_ratio = result.iloc[i]
                    defense_start = i

    return result


# ============================================================
# 完整流程
# ============================================================
def compute_regime_signals(
    daily_returns: pd.DataFrame,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    retrain_every: int = 60,
    min_duration: int = 60,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """完整 regime 信号计算流程.

    Parameters:
        daily_returns: 日频收益 DataFrame (T, N)
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口
        retrain_every: 重估频率
        min_duration: 最小避险持续天数

    Returns:
        composite: 复合信号 (降仓比例)
        enhanced: 增强信号 (降仓比例)
        bond_signal: 债券信号 (0/1)
    """
    from .jump_model import jump_model_rolling

    # 1. 对每个资产运行 Jump Model
    asset_signals = {}
    for col in daily_returns.columns:
        returns = daily_returns[col].dropna()
        if len(returns) < train_window:
            continue
        states = jump_model_rolling(
            returns, jump_penalty, train_window, retrain_every
        )
        # bear = 1, 转为信号
        asset_signals[col] = states

    # 2. 计算复合信号
    composite = compute_composite_signal(asset_signals)

    # 3. 计算增强信号 (需要债券信号)
    # 找到债券资产
    bond_assets = [k for k, v in ASSET_CLASSES.items() if v == 'bond' and k in asset_signals]
    if bond_assets:
        bond_signal = asset_signals[bond_assets[0]]
    else:
        bond_signal = pd.Series(0, index=composite.index)

    enhanced = compute_enhanced_signal(composite, bond_signal)

    # 4. 应用最小持续期
    composite = apply_min_duration(composite, min_duration)
    enhanced = apply_min_duration(enhanced, min_duration)

    return composite, enhanced, bond_signal

# coding=utf-8
"""宏观因子集 → 熵权综合得分 → risk_scalar.

借鉴 v9 银河方案 risk_scalar 机制:
    risk_scalar(t) = clip(1 + coef × factor_score, low, high)

数据源:
    v9_factors_weekly.parquet: 8 个专业宏观水平因子 (v9 银河方案)
    daily_returns (v56): 5 个 ETF-based 流动因子
    VIX / DXY / real_rate / cn_us_spread_10y: 额外宏观信号

多因子集:
    1. 5_FACTOR_ETF_ONLY: 仅 ETF 池内 5 因子 (Phase A 用)
    2. 8_V9_MACRO_LEVEL: 仅 v9 8 个宏观水平因子
    3. 8_V9_MACRO_FLOW:  v9 8 个宏观水平因子的 weekly 收益率
    4. 5_ETF + 8_V9 = 13 混合
    5. 5_ETF + VIX = 6 (含波动率)
    6. 5_ETF + 4_EXTRA = 9 (含 VIX/DXY/real_rate/spread)
    7. ALL = 5 + 8 + 4 = 17 综合
"""
from __future__ import annotations

import pandas as pd

from .factor_galaxy import entropy_weight, composite_score


FIVE_FACTOR_COLUMNS = ['510300', '518880', '中债1-3年国债财富指数', '513500', '510500']

# v9 8 个专业宏观因子 (level, 已是水平值)
V9_MACRO_COLUMNS = [
    '宏观增长因子',
    '宏观通胀因子_生活端',
    '宏观通胀因子_生产端',
    '无风险收益率',
    '信用利差因子',
    '期限利差因子_债',
    '期限利差因子_股',
    '宏观汇率因子',
]

# v9 8 个宏观因子的"方向": 高 = 好（应加仓）
V9_MACRO_SIGN = {
    '宏观增长因子': 1,
    '宏观通胀因子_生活端': -1,  # 通胀上升 → 减仓
    '宏观通胀因子_生产端': -1,
    '无风险收益率': -1,          # 实际利率上升 → 减仓
    '信用利差因子': -1,           # 信用利差扩大 = 风险加大 → 减仓
    '期限利差因子_债': 1,
    '期限利差因子_股': 1,
    '宏观汇率因子': 1,            # 本币升 = 风险偏好
}


def _weekly_returns_from_levels(weekly_levels: pd.DataFrame) -> pd.DataFrame:
    """周频 level → 周频 weekly returns (用来反映动量)."""
    return weekly_levels.pct_change().dropna()


def compute_five_macro_factors(
    daily_returns: pd.DataFrame,
    zscore_window: int = 13,
) -> pd.DataFrame:
    """5 个真实宏观因子 (周频 + 短窗 zscore).

    | 因子 | 计算 | 含义 |
    |------|------|------|
    | growth | 沪深300 周收益 | 增长好 → 满仓 |
    | inflation | -黄金 周收益 | 黄金涨 = 通胀 = 减仓 |
    | liquidity | 短债 - 沪深300 | 比率升 = 宽松 |
    | fx | 海外 - 沪深300 | 超额 → 外资流入 |
    | risk_preference | 沪深300 - 中证500 | 大盘强 = 避险 |
    """
    nav = (1 + daily_returns[FIVE_FACTOR_COLUMNS].fillna(0)).cumprod()
    weekly = nav.resample('W').last().pct_change()

    factors = pd.DataFrame({
        'growth': weekly['510300'],
        'inflation': -weekly['518880'],
        'liquidity': weekly['中债1-3年国债财富指数'] - weekly['510300'],
        'fx': weekly['513500'] - weekly['510300'],
        'risk_preference': weekly['510300'] - weekly['510500'],
    }).dropna()

    z = factors.rolling(zscore_window).mean()
    s = factors.rolling(zscore_window).std() + 1e-10
    return ((factors - z) / s).dropna()


def compute_v9_macro_factors(
    v9_weekly: pd.DataFrame,
    zscore_window: int = 13,
    use_flow: bool = True,
) -> pd.DataFrame:
    """v9 8 个宏观水平因子 → 周频 (level 或 flow) + 短窗 zscore.

    参数:
        v9_weekly: v9_factors_weekly.parquet (DataFrame, 8 列)
        zscore_window: zscore 窗口 (周)
        use_flow: True → 用 weekly returns, False → 用 level 滚动 zscore

    返回:
        DataFrame (T, 8), 每列已 zscore 化且方向已对齐 (高 = 好)
    """
    cols = [c for c in V9_MACRO_COLUMNS if c in v9_weekly.columns]

    if use_flow:
        # 用 weekly returns (动量), 方向已对齐
        raw = _weekly_returns_from_levels(v9_weekly[cols])
    else:
        # 用 level 滚动 zscore (水平估值)
        raw = v9_weekly[cols]

    for col in cols:
        sign = V9_MACRO_SIGN.get(col, 1)
        if sign == -1:
            raw = raw.copy()
            raw[col] = -raw[col]

    z = raw.rolling(zscore_window).mean()
    s = raw.rolling(zscore_window).std() + 1e-10
    return ((raw - z) / s).dropna()


def compute_extra_macro_factors(
    vix_daily: pd.Series,
    dxy_daily: pd.Series,
    real_rate_daily: pd.Series,
    cn_us_spread: pd.Series,
    zscore_window: int = 13,
) -> pd.DataFrame:
    """4 个额外宏观信号 → 周频 + zscore.

    VIX: 高 = 恐慌 → 反向 (VIX 上升 → 减仓)
    DXY: 高 = 美元强 → 中国资本外流 → 减仓
    real_rate: 高 = 紧 → 减仓
    spread (cn-us): 高 = 中国宽松 → 加仓
    """
    combined = pd.DataFrame({
        'vix': -vix_daily,
        'dxy': -dxy_daily,
        'real_rate': -real_rate_daily,
        'cn_us_spread': cn_us_spread,
    }).dropna()

    weekly = combined.resample('W').last().pct_change().dropna()

    z = weekly.rolling(zscore_window).mean()
    s = weekly.rolling(zscore_window).std() + 1e-10
    return ((weekly - z) / s).dropna()


def compute_factor_score_from_macro(
    macro_factors: pd.DataFrame,
    window: int = 104,
) -> pd.Series:
    """通用入口: 任意宏观因子集 → 熵权综合得分."""
    score_records = {}
    for t in range(window, len(macro_factors)):
        weights = entropy_weight(macro_factors.iloc[:t], window=window)
        score_records[macro_factors.index[t]] = composite_score(
            macro_factors.iloc[t], weights
        )

    return pd.Series(score_records).dropna()


def compute_factor_score(daily_returns: pd.DataFrame, window: int = 104) -> pd.Series:
    """主入口: 5 因子 → 熵权综合得分 (旧接口, 兼容)."""
    zscore_factors = compute_five_macro_factors(daily_returns)
    return compute_factor_score_from_macro(zscore_factors, window)


def compute_risk_scalar(
    factor_score: pd.Series,
    window: int = 52,
    clip_low: float = 0.3,
    clip_high: float = 1.5,
    coef: float = 0.3,
) -> pd.Series:
    """借鉴 v9: dynamic position adjustment.

    risk_scalar(t) = clip(1 + coef × factor_score, clip_low, clip_high)

    逻辑:
        factor_score 高 (宏观好) → risk_scalar > 1 (进攻)
        factor_score 低 (宏观差) → risk_scalar < 1 (防御)

    参数:
        factor_score: 综合得分 Series (均值 0, 已 zscore 化)
        clip_low/high: 仓位上下限
        coef: 敏感度系数
    """
    return ((1 + coef * factor_score).clip(clip_low, clip_high)).dropna()


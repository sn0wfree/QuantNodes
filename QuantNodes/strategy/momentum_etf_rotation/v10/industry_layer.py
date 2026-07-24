# coding=utf-8
"""Layer 2A: 行业轮动 (Industry Rotation).

来源: v9 citic_rotation (动量 + 质量打分) + 改进 (regime 条件 + 相关约束)

功能:
    1. 在 23 个行业 ETF 内做轮动
    2. 因子: 动量 (12-1 月) + 反向波动率
    3. Top-K 行业 5x 加权, 其他 0.5x (沿用 v9)
    4. Regime 条件: bull→进攻型, bear→防御型
    5. 相关性约束: 剔除 corr > 0.7 的冗余行业 (默认关闭, Stage 30 实证无效)

输出:
    weights: (T, N) 权重调整系数 (相对于等权 1/N)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_v10 import IndustryLayerConfig


# 43 ETF 中的行业 ETF (来自 v4 universe_v4.SECTOR_CODES)
SECTOR_CODES = (
    "512760",   # 半导体
    "512480",   # 半导体 (国联安)
    "515030",   # 新能源车
    "515790",   # 光伏
    "512690",   # 酒
    "512170",   # 医疗
    "512010",   # 医药
    "515050",   # 通信
    "159928",   # 消费
    "512880",   # 证券
    "512000",   # 券商
    "512800",   # 银行
    "515220",   # 煤炭
    "512200",   # 房地产
    "512400",   # 有色金属
    "512660",   # 军工
    "512980",   # 传媒
    "515880",   # 通信 ETF
    "159996",   # 家电
    "512120",   # 化工
    "161226",   # 纳指 (分类到行业)
    "159981",   # 能源化工
    "159766",   # 旅游
)

# 进攻型行业 (bull 加权)
OFFENSIVE_INDUSTRIES = frozenset({
    "512760",   # 半导体
    "512480",   # 半导体
    "515030",   # 新能源车
    "515790",   # 光伏
    "512660",   # 军工
    "515880",   # 通信
    "512000",   # 券商
    "512880",   # 证券
})

# 防御型行业 (bear 加权)
DEFENSIVE_INDUSTRIES = frozenset({
    "512800",   # 银行
    "512170",   # 医疗
    "512010",   # 医药
    "159928",   # 消费
    "159996",   # 家电
    "512120",   # 化工
})


def _cross_section_zscore(s: pd.Series) -> pd.Series:
    """横截面 z-score (单期)."""
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / (std + 1e-10)


def _industry_score_at(returns_df: pd.DataFrame, as_of: pd.Timestamp,
                        cfg: IndustryLayerConfig) -> pd.Series:
    """计算行业截面得分 (单个时点).

    score = z(momentum) - z(volatility)
    """
    sub = returns_df.loc[:as_of].iloc[:-1]   # 不包含 as_of 当期
    sector_codes = [c for c in returns_df.columns if c in SECTOR_CODES]
    if len(sub) < cfg.volatility_lookback + cfg.momentum_skip + 2:
        return pd.Series(0.0, index=sector_codes)

    # 1. 动量 (lookback - skip)
    cum_ret = (1 + sub[sector_codes]).iloc[-cfg.momentum_lookback - cfg.momentum_skip:
                                          -cfg.momentum_skip if cfg.momentum_skip > 0 else None].prod() - 1
    if isinstance(cum_ret, pd.DataFrame):
        cum_ret = cum_ret.iloc[-1]

    # 2. 波动率
    vol = sub[sector_codes].iloc[-cfg.volatility_lookback:].std()

    # 3. 横截面 z-score
    z_mom = _cross_section_zscore(cum_ret)
    z_vol = _cross_section_zscore(vol)

    return (z_mom - z_vol).fillna(0)


def _apply_regime_condition(score: pd.Series, regime: str,
                             cfg: IndustryLayerConfig) -> pd.Series:
    """Regime 条件调整."""
    if not cfg.regime_enabled or regime == 'neutral':
        return score

    s = score.copy()
    if regime == 'bull':
        # 进攻型行业加权
        for code in OFFENSIVE_INDUSTRIES:
            if code in s.index:
                s.loc[code] *= cfg.bull_offensive_boost
    elif regime == 'bear':
        # 防御型行业加权
        for code in DEFENSIVE_INDUSTRIES:
            if code in s.index:
                s.loc[code] *= cfg.bear_defensive_boost

    return s


def _apply_corr_constraint(score: pd.Series, returns_df: pd.DataFrame,
                            as_of: pd.Timestamp, cfg: IndustryLayerConfig) -> pd.Series:
    """相关性约束: 剔除 corr > threshold 的冗余行业.

    Stage 30 实证: 在完整 v4 回测中拖累表现, 默认关闭.
    """
    if not cfg.corr_constraint:
        return score

    sub = returns_df.loc[:as_of].iloc[:-1]
    sector_codes = [c for c in returns_df.columns if c in SECTOR_CODES]
    if len(sub) < cfg.corr_window + 2:
        return score

    corr_window = sub[sector_codes].iloc[-cfg.corr_window:].corr()

    s = score.copy()
    selected = []
    sorted_codes = s.sort_values(ascending=False).index.tolist()

    for code in sorted_codes:
        if not selected:
            selected.append(code)
            continue
        # 检查与已选行业的相关性
        max_corr = 0.0
        for sel in selected:
            if code in corr_window.index and sel in corr_window.columns:
                max_corr = max(max_corr, abs(corr_window.loc[code, sel]))
        if max_corr < cfg.corr_threshold:
            selected.append(code)
        else:
            # 剔除冗余 (得分置 0)
            s.loc[code] = 0

    return s


def _build_industry_tilt(returns_df: pd.DataFrame, as_of: pd.Timestamp,
                          regime: str, cfg: IndustryLayerConfig) -> pd.Series:
    """构造行业调整系数 (相对于等权 1/N).

    返回:
        tilt: Series, index=ETF codes, values ∈ [0, ~5]
    """
    codes = returns_df.columns.tolist()
    tilt = pd.Series(1.0, index=codes)   # 默认等权

    sector_codes = [c for c in codes if c in SECTOR_CODES]
    if len(sector_codes) < cfg.top_k:
        return tilt

    # 1. 计算行业得分
    score = _industry_score_at(returns_df, as_of, cfg)
    if score.empty:
        return tilt

    # 2. 相关性约束
    score = _apply_corr_constraint(score, returns_df, as_of, cfg)

    # 3. Regime 条件
    score = _apply_regime_condition(score, regime, cfg)

    # 4. Top-K 选优
    sorted_score = score.sort_values(ascending=False)
    top_k_codes = sorted_score.head(cfg.top_k).index.tolist()
    bot_codes = [c for c in sector_codes if c not in top_k_codes]

    # 5. 调整系数
    for code in top_k_codes:
        tilt.loc[code] = cfg.sector_mult
    for code in bot_codes:
        tilt.loc[code] = cfg.sector_floor_mult

    return tilt


def compute_industry_tilt(
    returns_df: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    regime_series: pd.Series,
    cfg: IndustryLayerConfig,
) -> pd.DataFrame:
    """Layer 2A 主入口: 计算行业调整系数时序.

    参数:
        returns_df: ETF 收益 DataFrame
        rebal_dates: 调仓日期
        regime_series: regime 时序 (来自 Layer 1)
        cfg: IndustryLayerConfig

    返回:
        tilt: (T_rebal, N) DataFrame, 调整系数
    """
    if not cfg.enabled:
        idx = rebal_dates
        tilt = pd.DataFrame(1.0, index=idx, columns=returns_df.columns)
        return tilt

    tilts = []
    for date in rebal_dates:
        regime = regime_series.get(date, 'neutral') if regime_series is not None else 'neutral'
        tilt = _build_industry_tilt(returns_df, date, regime, cfg)
        tilts.append(tilt)

    tilt_df = pd.DataFrame(tilts, index=rebal_dates)
    return tilt_df


# ============================================================
# 类封装
# ============================================================
class IndustryLayer:
    """Layer 2A 行业轮动封装."""

    def __init__(self, cfg: IndustryLayerConfig | None = None):
        self.cfg = cfg or IndustryLayerConfig()
        self.tilt: pd.DataFrame | None = None

    def fit(self, returns_df: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
            regime_series: pd.Series) -> "IndustryLayer":
        """计算行业调整系数时序."""
        self.tilt = compute_industry_tilt(returns_df, rebal_dates, regime_series, self.cfg)
        return self

    def get_tilt(self, date: pd.Timestamp) -> pd.Series:
        """获取指定日期的调整系数."""
        if self.tilt is None or date not in self.tilt.index:
            return pd.Series(dtype=float)
        return self.tilt.loc[date]
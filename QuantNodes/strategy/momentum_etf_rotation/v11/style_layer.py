# coding=utf-8
"""Layer 2B: 风格轮动 (Style Rotation).

来源: v4 factor_timing_v4 (IC 驱动) + 改进 (regime 条件)

功能:
    1. 6 因子 IC: momentum/value/reversal/quality/size/low_vol
    2. 因子特异性 forward_window + lag 平滑 + 阈值过滤
    3. ic_weight = max(0, ic + base) ** power
    4. Regime 条件: bull→momentum, bear→value+quality
    5. 归一化 → style_weights

输出:
    style_weights: dict[str, float], 6 风格权重和为 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_v11 import StyleLayerConfig


# 6 风格因子定义
STYLE_FACTORS = ('momentum', 'value', 'reversal', 'quality', 'size', 'low_vol')


def _compute_single_factor(factor_name: str, returns_df: pd.DataFrame,
                            as_of: pd.Timestamp, lookback: int = 60) -> pd.Series:
    """计算单个风格因子的横截面得分.

    Returns:
        Series, index=ETF codes, values=score (越大越强)
    """
    sub = returns_df.loc[:as_of].iloc[:-1]   # 不包含 as_of 当期
    if len(sub) < lookback + 5:
        return pd.Series(dtype=float)

    if factor_name == 'momentum':
        # 60 周动量 (正向)
        ret = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1
    elif factor_name == 'value':
        # 净值/MA60 - 1 (反向: 低偏离 = 低估)
        ma60 = sub.iloc[-lookback:].mean()
        ret = -(sub.iloc[-1] / (ma60 + 1e-10) - 1)
    elif factor_name == 'reversal':
        # 5 周反转 (反向: 短期下跌 = 未来反弹)
        ret = -(sub.iloc[-1] / sub.iloc[-6] - 1)
    elif factor_name == 'quality':
        # 26 周 Sharpe
        mean_ret = sub.iloc[-lookback:].mean()
        std_ret = sub.iloc[-lookback:].std()
        ret = mean_ret / (std_ret + 1e-10)
    elif factor_name == 'size':
        # 4 周均振幅 (反向: 振幅小 = 大盘)
        ret = -sub.iloc[-lookback:].abs().mean()
    elif factor_name == 'low_vol':
        # 60 周波动率 (反向: 低波)
        log_ret = np.log(sub / sub.shift(1).replace(0, np.nan))
        ret = -log_ret.iloc[-lookback:].std()
    else:
        return pd.Series(dtype=float)

    # 横截面排名
    return ret.rank(method='average', pct=True, na_option='bottom').fillna(0.5)


def _compute_forward_return(returns_df: pd.DataFrame, as_of: pd.Timestamp,
                             forward_window: int) -> pd.Series:
    """计算 forward_window 周的未来收益 (IC 标签)."""
    try:
        idx = returns_df.index.get_loc(as_of)
    except KeyError:
        return pd.Series(dtype=float)
    if idx + forward_window >= len(returns_df):
        return pd.Series(dtype=float)
    future = returns_df.iloc[idx + 1: idx + 1 + forward_window]
    return (1 + future).prod() - 1


def _safe_spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman 秩相关 (NaN-safe)."""
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    try:
        corr = aligned.corr(method='spearman')
        if corr.isna().iloc[0, 1]:
            return 0.0
        return float(corr.iloc[0, 1])
    except Exception:
        return 0.0


def compute_factor_ic(
    returns_df: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: StyleLayerConfig,
    ic_window: int | None = None,
) -> dict[str, float]:
    """计算 6 风格因子的 IC.

    参数:
        returns_df: ETF 收益
        as_of: 当前日期
        cfg: StyleLayerConfig
        ic_window: IC 计算窗口 (默认 cfg.ic_lookback)

    返回:
        ic_dict: {factor_name: IC_value}
    """
    if ic_window is None:
        ic_window = cfg.ic_lookback

    sub = returns_df.loc[:as_of].iloc[:-1]
    if len(sub) < ic_window + 60:
        return {f: 0.0 for f in STYLE_FACTORS}

    # 收集历史 IC
    ic_history = {f: [] for f in STYLE_FACTORS}

    # 在 [as_of - ic_window, as_of - 1] 区间内每 ic_step 计算一次 IC
    sample_dates = sub.index[-(ic_window + max(cfg.factor_fw.values())):]
    if len(sample_dates) < 10:
        return {f: 0.0 for f in STYLE_FACTORS}

    sample_dates = sample_dates[::cfg.ic_step]

    for past_date in sample_dates:
        for factor_name in STYLE_FACTORS:
            fw = cfg.factor_fw.get(factor_name, 60)
            # 因子得分
            factor_score = _compute_single_factor(factor_name, returns_df, past_date, lookback=60)
            if factor_score.empty:
                continue
            # 未来收益
            fwd_ret = _compute_forward_return(returns_df, past_date, fw)
            if fwd_ret.empty:
                continue
            # IC
            ic = _safe_spearman_corr(factor_score, fwd_ret)
            ic_history[factor_name].append(ic)

    # 平滑 (因子特异性 lag 平滑窗口)
    ic_dict = {}
    for factor_name in STYLE_FACTORS:
        ic_list = ic_history[factor_name]
        if len(ic_list) < 5:
            ic_dict[factor_name] = 0.0
            continue
        smooth_window = cfg.factor_smooth_window.get(factor_name, 4)
        if smooth_window > 1 and len(ic_list) >= smooth_window:
            ic_smoothed = pd.Series(ic_list).rolling(smooth_window, min_periods=1).mean().iloc[-1]
        else:
            ic_smoothed = ic_list[-1]
        ic_dict[factor_name] = float(ic_smoothed)

    return ic_dict


def _apply_regime_to_styles(ic_weight: dict[str, float], regime: str,
                             cfg: StyleLayerConfig) -> dict[str, float]:
    """Regime 条件调整风格权重."""
    if not cfg.regime_enabled or regime == 'neutral':
        return ic_weight

    w = dict(ic_weight)
    if regime == 'bull':
        w['momentum'] = w.get('momentum', 0) * cfg.bull_momentum_boost
    elif regime == 'bear':
        w['value'] = w.get('value', 0) * cfg.bear_value_boost
        w['quality'] = w.get('quality', 0) * cfg.bear_quality_boost

    return w


def compute_style_weights(
    returns_df: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    regime_series: pd.Series,
    cfg: StyleLayerConfig,
) -> pd.DataFrame:
    """Layer 2B 主入口: 计算风格权重时序.

    参数:
        returns_df: ETF 收益
        rebal_dates: 调仓日期
        regime_series: regime 时序 (来自 Layer 1)
        cfg: StyleLayerConfig

    返回:
        style_weights: (T_rebal, 6) DataFrame, 列名=风格因子
    """
    if not cfg.enabled:
        return pd.DataFrame(
            {f: 1.0 / len(STYLE_FACTORS) for f in STYLE_FACTORS},
            index=rebal_dates,
        )

    weights = []
    for date in rebal_dates:
        # 1. 计算 IC
        ic_dict = compute_factor_ic(returns_df, date, cfg)

        # 2. IC 加权: weight = max(0, ic + base) ** power
        ic_weight = {}
        for f in STYLE_FACTORS:
            raw = max(0.0, ic_dict.get(f, 0.0) + cfg.ic_base) ** cfg.ic_power
            # 阈值过滤
            if abs(ic_dict.get(f, 0.0)) < cfg.ic_threshold:
                raw = 0.0
            ic_weight[f] = raw

        # 3. Regime 条件
        regime = regime_series.get(date, 'neutral') if regime_series is not None else 'neutral'
        ic_weight = _apply_regime_to_styles(ic_weight, regime, cfg)

        # 4. 归一化
        total = sum(ic_weight.values())
        if total > 0:
            ic_weight = {k: v / total for k, v in ic_weight.items()}
        else:
            ic_weight = {f: 1.0 / len(STYLE_FACTORS) for f in STYLE_FACTORS}

        # 5. 上下限
        for f in STYLE_FACTORS:
            ic_weight[f] = max(cfg.min_weight, min(cfg.max_weight, ic_weight[f]))
        # 重新归一化
        total = sum(ic_weight.values())
        if total > 0:
            ic_weight = {k: v / total for k, v in ic_weight.items()}

        weights.append(ic_weight)

    return pd.DataFrame(weights, index=rebal_dates)


# ============================================================
# 类封装
# ============================================================
class StyleLayer:
    """Layer 2B 风格轮动封装."""

    def __init__(self, cfg: StyleLayerConfig | None = None):
        self.cfg = cfg or StyleLayerConfig()
        self.weights: pd.DataFrame | None = None

    def fit(self, returns_df: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
            regime_series: pd.Series) -> "StyleLayer":
        """计算风格权重时序."""
        self.weights = compute_style_weights(returns_df, rebal_dates, regime_series, self.cfg)
        return self

    def get_weights(self, date: pd.Timestamp) -> dict[str, float]:
        """获取指定日期的风格权重."""
        if self.weights is None or date not in self.weights.index:
            return {f: 1.0 / len(STYLE_FACTORS) for f in STYLE_FACTORS}
        return self.weights.loc[date].to_dict()

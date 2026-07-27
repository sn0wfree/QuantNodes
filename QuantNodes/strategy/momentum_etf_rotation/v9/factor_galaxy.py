# coding=utf-8
"""v9 银河证券因子配置 — 核心模块.

银河证券因子配置方法:
    1. 熵权法合成综合得分 (信息熵 → 权重)
    2. 滚动回归 → 因子敏感度 β
    3. 方差分解 → 因子贡献占比
    4. 因子风险预算权重反推

参考文献:
    银河证券 2026 公开报告:
    - 方差分解与因子暴露
    - 因子配置而非资产配置
    - 熵权法合成五类宏观指标
    - 波动率因子周度最优占比 29.28% (轮动特征)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def entropy_weight(data: pd.DataFrame, window: int = 104) -> dict:
    """熵权法计算指标权重.

    原理:
        信息熵越大 → 指标越无序 → 权重越小
        信息熵越小 → 指标信息量越大 → 权重越大

    参数:
        data: 标准化后的宏观指标 DataFrame (index=日期)
        window: 滚动窗口 (周, 默认 104 = 2 年)

    返回:
        weights: {col: weight} 每个指标的权重 (归一化和=1)
    """
    if len(data) < window:
        return {col: 1.0 / len(data.columns) for col in data.columns}

    recent = data.iloc[-window:]
    n = len(recent)
    weights = {}

    for col in recent.columns:
        x = recent[col].abs()
        p = x / x.sum()
        entropy = -(p * np.log(p + 1e-10)).sum() / np.log(n)
        weights[col] = 1 - entropy

    total = sum(weights.values())
    if total == 0:
        return {col: 1.0 / len(data.columns) for col in data.columns}
    return {k: v / total for k, v in weights.items()}


def composite_score(macro_data: pd.DataFrame, weights: dict) -> pd.Series:
    """加权合成综合得分.

    参数:
        macro_data: 宏观指标 DataFrame
        weights: {col: weight}

    返回:
        score: 综合得分 Series
    """
    return sum(macro_data[col] * w for col, w in weights.items())


def rolling_factor_beta(
    asset_returns: pd.DataFrame,
    factor_score: pd.Series,
    window: int = 52,
) -> pd.DataFrame:
    """滚动估计每个资产对因子的 β.

    公式:
        β_i = Cov(R_i, Score) / Var(Score)
        R_i = α + β × Score + ε

    参数:
        asset_returns: (T, N) 资产收益
        factor_score: (T,) 因子综合得分
        window: 滚动窗口 (默认 52)

    返回:
        beta: (T, N) 每个资产对因子的 β
    """
    common_idx = asset_returns.index.intersection(factor_score.index)
    if len(common_idx) == 0:
        return pd.DataFrame(0.0, index=asset_returns.index, columns=asset_returns.columns)

    asset_aligned = asset_returns.loc[common_idx].fillna(0)
    score_aligned = factor_score.loc[common_idx].fillna(0)

    score_mean = score_aligned.rolling(window).mean()
    factor_centered = score_aligned - score_mean

    beta = pd.DataFrame(0.0, index=asset_aligned.index, columns=asset_aligned.columns)
    factor_var = (factor_centered ** 2).rolling(window).mean()

    for col in asset_aligned.columns:
        ret_centered = asset_aligned[col] - asset_aligned[col].rolling(window).mean()
        cov = (ret_centered * factor_centered).rolling(window).mean()
        beta[col] = cov / factor_var

    beta = beta.reindex(asset_returns.index).fillna(0)
    return beta.clip(-3, 3)


def variance_decomposition(
    asset_returns: pd.DataFrame,
    factor_score: pd.Series,
    beta: pd.DataFrame,
    window: int = 52,
) -> pd.DataFrame:
    """方差分解: 因子贡献占比.

    公式:
        Factor Contribution = β² × Var(Score) / Var(R)
        占比越大 → 资产越受宏观因子驱动

    返回:
        contribution: (T, N) 因子贡献占比 (0-1)
    """
    var_factor = (beta ** 2) * factor_score.rolling(window).var()
    var_total = asset_returns.rolling(window).var()
    contribution = var_factor / (var_total + 1e-10)
    return contribution.clip(0, 1).fillna(0)


def risk_budget_weights(
    asset_returns: pd.DataFrame,
    factor_score: pd.Series,
    beta: pd.DataFrame,
    target_budget: dict | None = None,
    window: int = 52,
    floor: float = 0.02,
    cap: float = 0.20,
) -> pd.DataFrame:
    """因子风险预算权重反推.

    公式 (银河证券 v1):
        w_i ∝ |β_i| × target_risk_i / σ²_i
        归一化: Σ w_i = 1

    注: 银河证券 v2 增加 risk_budget_t 调整:
        当 factor_score 上升时, target_risk 下降 (防御);
        当 factor_score 下降时, target_risk 上升 (进攻).

    参数:
        asset_returns: (T, N) 资产收益
        factor_score: (T,) 因子综合得分
        beta: (T, N) 因子敏感度
        target_budget: {资产名: 目标风险贡献}, 默认等权
        window: 滚动窗口
        floor: 单资产最小权重
        cap: 单资产最大权重

    返回:
        weights: (T, N) 归一化权重
    """
    if target_budget is None:
        target_budget = {c: 1.0 / len(asset_returns.columns) 
                         for c in asset_returns.columns}

    target_series = pd.Series(target_budget)

    score_zscore = (factor_score - factor_score.rolling(window).mean()) / (
        factor_score.rolling(window).std() + 1e-10
    )
    risk_scalar = (1 - 0.8 * score_zscore).clip(0.3, 1.5)

    vol = asset_returns.rolling(window).std()
    var = vol ** 2 + 1e-10

    abs_beta = beta.abs()
    raw_weights = abs_beta.mul(target_series, axis=1) / var
    raw_weights = raw_weights.fillna(0)
    raw_weights = raw_weights.mul(risk_scalar, axis=0)
    raw_weights = raw_weights.clip(lower=floor)

    weights = raw_weights.div(raw_weights.sum(axis=1).replace(0, 1), axis=0).fillna(0)

    if cap < 1.0:
        weights = weights.clip(upper=cap)
        remaining = 1.0 - weights.sum(axis=1)
        deficit_mask = remaining > 0
        if deficit_mask.any():
            for t_idx in weights.index[deficit_mask]:
                rem = remaining.loc[t_idx]
                active = weights.loc[t_idx, weights.loc[t_idx] < cap]
                if len(active) > 0 and active.sum() > 0:
                    extra = (active / active.sum()) * rem
                    weights.loc[t_idx, active.index] = active + extra

    return weights.fillna(0)


def galaxy_factor_allocation(
    returns_df: pd.DataFrame,
    macro_indicators: pd.DataFrame,
    lookback_score: int = 104,
    lookback_beta: int = 52,
    target_budget: dict | None = None,
    floor: float = 0.0,
    cap: float = 0.30,
) -> tuple:
    """银河证券因子配置主入口.

    参数:
        returns_df: (T, N) 资产收益 DataFrame (周频)
        macro_indicators: (T, K) 宏观指标 DataFrame (周频)
        lookback_score: 熵权法滚动窗口 (默认 104 = 2 年)
        lookback_beta: β 回归滚动窗口 (默认 52 = 1 年)
        target_budget: 目标风险贡献
        floor/cap: 单资产权重上下限

    返回:
        weights: (T, N) 周度权重
        factor_score: (T,) 综合得分
        betas: (T, N) 因子敏感度
    """
    common_idx = returns_df.index.intersection(macro_indicators.index)
    returns_aligned = returns_df.loc[common_idx]
    macro_aligned = macro_indicators.loc[common_idx]

    n = len(common_idx)

    rolling_weights = {}
    scores = {}
    for t in range(lookback_score, n):
        weights_t = entropy_weight(macro_aligned.iloc[:t], window=lookback_score)
        rolling_weights[common_idx[t]] = weights_t
        scores[common_idx[t]] = composite_score(macro_aligned.iloc[t], weights_t)

    if not scores:
        raise ValueError(f"数据不足 (需要 {lookback_score} 周, 实际 {n})")

    score_series = pd.Series(scores).reindex(common_idx).fillna(method='ffill')

    betas = rolling_factor_beta(returns_aligned, score_series, window=lookback_beta)

    weights = risk_budget_weights(
        returns_aligned, score_series, betas,
        target_budget=target_budget,
        window=lookback_beta,
        floor=floor, cap=cap,
    )

    weights = weights.fillna(0)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)

    return weights, score_series, betas


def compute_factor_metrics(
    weights: pd.DataFrame,
    returns_df: pd.DataFrame,
    freq: str = 'W',
) -> dict:
    """计算因子配置回测指标 (复用 backtest.py 的 compute_metrics).

    重要: freq 参数决定年化因子 (W=52, D=252).
    """
    from .backtest import compute_metrics

    common = weights.index.intersection(returns_df.index)
    w = weights.loc[common]
    r = returns_df.loc[common]

    if isinstance(w.index, pd.DatetimeIndex):
        w_daily = w.resample('D').ffill().reindex(r.index, method='ffill').fillna(0)
    else:
        w_daily = w.reindex(r.index, method='ffill').fillna(0)

    nav = pd.Series(1.0, index=r.index)
    prev_w = pd.Series(0.0, index=r.columns)
    cost_bps = 5.0

    for i in range(len(r)):
        wi = w_daily.iloc[i] if i < len(w_daily) else pd.Series(0, index=r.columns)
        ri = r.iloc[i]
        port_ret = (wi * ri).sum()
        turnover = (wi - prev_w).abs().sum()
        cost = turnover * cost_bps / 10000.0
        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost) if i > 0 else 1.0
        prev_w = wi.copy()

    returns = nav.pct_change().dropna()
    if returns.std() == 0:
        return {'Sharpe': 0, 'Calmar': 0, 'MaxDD': 0, 'AnnRet': 0, 'Vol': 0, 'WinRate': 0}

    return compute_metrics(returns, freq=freq)

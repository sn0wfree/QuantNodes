# coding=utf-8
"""因子 IC 计算 (Stage 17 + Stage 27 重构: 适配 43 ETF).

8 个因子 (基于 ETF 收益本身构造, 不依赖特定 ETF):
1. momentum     (动量):   rank(60d return)
2. reversal     (反转):   -rank(5d return)
3. value        (价值):   -rank(净值/MA60 - 1)  (低偏离=低估)
4. low_vol      (低波):   -rank(60d vol)
5. momentum_12_1 (中期动量): rank(12-1 月收益)  (Stage 27: 跳过最近 1 月)
6. volatility_change (波动率变化): rank(20d vol - 60d vol)  (上升=风险)
7. value_proxy  (估值代理): -rank(52周累计收益) (越跌越便宜)
8. quality_proxy (基本面代理): rank(26周 Sharpe) (高质量≈高ROE)

IC = Spearman(factor_score, forward_return)

Stage 27 重构:
- 移除 dividend/quality 因子 (依赖特定 ETF)
- 用 momentum_12_1 替代 (更通用)
- 用 volatility_change 替代 (波动率代理)
- 适配任意 ETF 池 (43 ETF)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# 因子名 (8 个, Stage 27 重构)
FACTOR_NAMES: tuple[str, ...] = (
    "momentum",            # 动量
    "reversal",            # 反转
    "value",               # 价值
    "low_vol",             # 低波
    "momentum_12_1",       # 中期动量 (12-1 月) - Stage 27 新增
    "volatility_change",   # 波动率变化 - Stage 27 新增
    "value_proxy",         # 估值代理
    "quality_proxy",       # 基本面代理
)


def _safe_rank(s: pd.Series, pct: bool = True) -> pd.Series:
    """安全排名 (NaN-safe, 缺失值排最后)."""
    return s.rank(method="average", pct=pct, na_option="bottom")


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """安全 Spearman 相关."""
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    try:
        result = aligned.corr(method="spearman")
        if result.isna().iloc[0, 1]:
            return 0.0
        return float(result.iloc[0, 1])
    except Exception:
        return 0.0


def compute_factor_scores(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    all_codes: Sequence[str],
    lookback: int = 60,
) -> dict[str, pd.Series]:
    """计算 8 个因子的截面得分 (适配 43 ETF).

    所有因子仅基于 ETF 自身收益构造, 不依赖特定 ETF 池.

    Returns:
        dict, name → pd.Series (index=code, values=score, 越大越强)
    """
    sub = nav_df.loc[:as_of, [c for c in all_codes if c in nav_df.columns]]
    if len(sub) < lookback + 2:
        return {n: pd.Series(dtype=float) for n in FACTOR_NAMES}

    # 1. 动量 (60d)
    ret_60 = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0
    mom_score = _safe_rank(ret_60)

    # 2. 反转 (5d)
    if len(sub) >= 6:
        ret_5 = sub.iloc[-1] / sub.iloc[-6] - 1.0
        rev_score = -_safe_rank(ret_5)
    else:
        rev_score = pd.Series(0.0, index=sub.columns)

    # 3. 价值 (低偏离 = 低估)
    ma60 = sub.iloc[-lookback:].mean()
    dev = sub.iloc[-1] / ma60 - 1.0
    val_score = -_safe_rank(dev)

    # 4. 低波
    log_ret = np.log(sub / sub.shift(1).replace(0, np.nan))
    vol_60 = log_ret.iloc[-lookback:].std() * np.sqrt(52)
    lv_score = -_safe_rank(vol_60)

    # 5. 中期动量 (12-1 月, skip 1 月) - Stage 27 新增
    lookback_252 = min(252, len(sub) - 1)
    lookback_21 = min(21, len(sub) - 1)
    if lookback_252 > 30 and lookback_21 < lookback_252:
        ret_252 = sub.iloc[-1] / sub.iloc[-lookback_252 - 1] - 1.0
        ret_21 = sub.iloc[-1] / sub.iloc[-lookback_21 - 1] - 1.0
        mom_12_1 = (1 + ret_252) / (1 + ret_21) - 1
        mom_12_1_score = _safe_rank(mom_12_1)
    else:
        mom_12_1_score = mom_score  # fallback

    # 6. 波动率变化 (vol 上升 = 风险上升) - Stage 27 新增
    lookback_60 = min(60, len(sub) - 1)
    if lookback_60 > 10 and lookback_21 > 5:
        vol_20 = log_ret.iloc[-lookback_21:].std() * np.sqrt(52)
        vol_60_recent = log_ret.iloc[-lookback_60:].std() * np.sqrt(52)
        vol_change = (vol_20 - vol_60_recent) / (vol_60_recent + 1e-10)
        vol_change_score = _safe_rank(vol_change)  # 高 = 波动率上升
    else:
        vol_change_score = pd.Series(0.0, index=sub.columns)

    # 7. 估值代理 (52 周累计收益的反向)
    lookback_52 = min(52, len(sub) - 1)
    if lookback_52 > 10:
        ret_52 = sub.iloc[-1] / sub.iloc[-lookback_52 - 1] - 1.0
        val_proxy_score = -_safe_rank(ret_52)
    else:
        val_proxy_score = pd.Series(0.0, index=sub.columns)

    # 8. 基本面代理 (26 周 Sharpe ratio)
    lookback_26 = min(26, len(sub) - 1)
    if lookback_26 > 5:
        log_ret_26 = log_ret.iloc[-lookback_26:]
        mean_ret_26 = log_ret_26.mean()
        std_ret_26 = log_ret_26.std()
        sharpe_26 = mean_ret_26 / (std_ret_26 + 1e-10)
        qual_proxy_score = _safe_rank(sharpe_26)
    else:
        qual_proxy_score = pd.Series(0.0, index=sub.columns)

    return {
        "momentum": mom_score,
        "reversal": rev_score,
        "value": val_score,
        "low_vol": lv_score,
        "momentum_12_1": mom_12_1_score,
        "volatility_change": vol_change_score,
        "value_proxy": val_proxy_score,
        "quality_proxy": qual_proxy_score,
    }


def compute_forward_return(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    all_codes: Sequence[str],
    forward_window: int = 20,
) -> pd.Series:
    """计算 as_of 之后 forward_window 天的累计收益."""
    sub = nav_df.loc[as_of:, [c for c in all_codes if c in nav_df.columns]]
    if len(sub) < 2:
        return pd.Series(dtype=float)
    base = sub.iloc[0]
    if len(sub) > forward_window:
        future = sub.iloc[forward_window]
    else:
        return pd.Series(dtype=float)
    return future / base - 1.0


def factor_ic_at(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    all_codes: Sequence[str],
    forward_window: int = 20,
    lookback: int = 60,
) -> dict[str, float]:
    """计算 as_of 当天的 8 因子 IC (Stage 27 升级).

    IC = Spearman(factor_score, forward_return)

    Returns:
        dict, name → IC value (range [-1, 1])
    """
    scores = compute_factor_scores(nav_df, as_of, all_codes, lookback=lookback)
    if not scores or all(s.empty for s in scores.values()):
        return {n: 0.0 for n in FACTOR_NAMES}

    fwd_ret = compute_forward_return(nav_df, as_of, all_codes, forward_window=forward_window)
    if fwd_ret.empty:
        return {n: 0.0 for n in FACTOR_NAMES}

    ic: dict[str, float] = {}
    for name, s in scores.items():
        ic[name] = _safe_corr(s, fwd_ret)
    return ic


def rolling_factor_ic(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window: int = 60,
    forward_window: int = 20,
    step: int = 5,
    lookback: int = 60,
) -> pd.DataFrame:
    """滚动计算 8 因子 IC 序列."""
    dates = nav_df.loc[start:end].index
    last_valid_date = nav_df.index[-forward_window] if len(nav_df) > forward_window else None
    if last_valid_date is None:
        return pd.DataFrame(columns=list(FACTOR_NAMES))

    valid_dates = dates[dates <= last_valid_date]
    if len(valid_dates) < 5:
        return pd.DataFrame(columns=list(FACTOR_NAMES))

    sample_dates = valid_dates[::step]
    rows: list[dict] = []
    sample_ts: list[pd.Timestamp] = []

    for ts in sample_dates:
        if ts not in nav_df.index:
            continue
        hist = nav_df.loc[:ts]
        if len(hist) < lookback + 2:
            continue

        ic = factor_ic_at(nav_df, ts, all_codes,
                          forward_window=forward_window,
                          lookback=lookback)
        rows.append(ic)
        sample_ts.append(ts)

    if not rows:
        return pd.DataFrame(columns=list(FACTOR_NAMES))

    return pd.DataFrame(rows, index=sample_ts)


def factor_ic_rolling_mean(
    ic_series: pd.DataFrame,
    smooth_window: int = 12,
) -> pd.DataFrame:
    """IC 滚动平均 (平滑)."""
    return ic_series.rolling(window=smooth_window, min_periods=3).mean()


__all__ = [
    "FACTOR_NAMES",
    "compute_factor_scores",
    "compute_forward_return",
    "factor_ic_at",
    "rolling_factor_ic",
    "factor_ic_rolling_mean",
]

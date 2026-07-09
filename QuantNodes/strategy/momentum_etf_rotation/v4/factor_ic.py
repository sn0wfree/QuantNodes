# coding=utf-8
"""因子 IC 计算 (Stage 17, v4.0).

6 个因子 (用 5 只风格组 ETF + 7 只 Smart β ETF 构造):
1. momentum  (动量):   rank(60d return)
2. reversal  (反转):   -rank(5d return)
3. value     (价值):   -rank(净值/MA60 - 1)  (低偏离=低估)
4. low_vol   (低波):   -rank(60d vol)
5. dividend  (红利):   rank(红利组代表 return)
6. quality   (质量):   rank(质量组代表 return)

IC = Spearman(factor_score, forward_return)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .universe_v4 import (
    SMART_BETA_CODES,
    SMART_BETA_METAS,
    STYLE_GROUP_CODES,
    StyleGroup,
    SmartBetaFactor,
)


# 因子名
FACTOR_NAMES: tuple[str, ...] = (
    "momentum",   # 动量
    "reversal",   # 反转
    "value",      # 价值
    "low_vol",    # 低波
    "dividend",   # 红利
    "quality",    # 质量
)


def _safe_rank(s: pd.Series, pct: bool = True) -> pd.Series:
    """安全排名 (NaN-safe, 缺失值排最后)."""
    return s.rank(method="average", pct=pct, na_option="bottom")


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """安全 Spearman 相关."""
    aligned = pd.concat([x, y], axis=1).dropna()
    # 至少 5 个样本才能算相关
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
    """计算 6 个因子的截面得分 (index=code).

    Returns:
        dict, name → pd.Series (index=code, values=score, 越大越强)
    """
    sub = nav_df.loc[:as_of, [c for c in all_codes if c in nav_df.columns]]
    if len(sub) < lookback + 2:
        return {n: pd.Series(dtype=float) for n in FACTOR_NAMES}

    # 1. 动量
    ret_60 = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0
    mom_score = _safe_rank(ret_60)

    # 2. 反转 (短期)
    ret_5 = sub.iloc[-1] / sub.iloc[-6] - 1.0
    rev_score = -_safe_rank(ret_5)

    # 3. 价值 (低偏离=低估)
    ma60 = sub.iloc[-lookback:].mean()
    dev = sub.iloc[-1] / ma60 - 1.0
    val_score = -_safe_rank(dev)

    # 4. 低波
    log_ret = np.log(sub / sub.shift(1))
    vol_60 = log_ret.iloc[-lookback:].std() * np.sqrt(252)
    lv_score = -_safe_rank(vol_60)

    # 5. 红利 (红利组得分高)
    div_codes = SMART_BETA_CODES.get(  # 红利类 Smart β
        # 取 510880 (红利) + 512890 + 515080 + 515100
    ) if False else [
        "510880", "512890", "515080", "515100"
    ]
    # 红利组收益 = 红利类 ETF 平均收益
    div_codes_valid = [c for c in div_codes if c in ret_60.index]
    if div_codes_valid:
        div_avg_ret = ret_60[div_codes_valid].mean()
    else:
        div_avg_ret = 0.0
    # 红利得分: 红利组内为正, 其他相对为 0
    div_score = pd.Series(0.0, index=ret_60.index)
    for c in div_codes_valid:
        div_score[c] = div_avg_ret - ret_60[c]
    # 排名 (高分 = 红利组)
    div_score = _safe_rank(div_score)

    # 6. 质量 (质量类 Smart β ETF)
    qual_codes = [
        "515900",   # 中证质量
        "512040",   # 国泰价值
    ]
    qual_codes_valid = [c for c in qual_codes if c in ret_60.index]
    if qual_codes_valid:
        qual_avg_ret = ret_60[qual_codes_valid].mean()
    else:
        qual_avg_ret = 0.0
    qual_score = pd.Series(0.0, index=ret_60.index)
    for c in qual_codes_valid:
        qual_score[c] = qual_avg_ret - ret_60[c]
    qual_score = _safe_rank(qual_score)

    return {
        "momentum": mom_score,
        "reversal": rev_score,
        "value":    val_score,
        "low_vol":  lv_score,
        "dividend": div_score,
        "quality":  qual_score,
    }


def compute_forward_return(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    all_codes: Sequence[str],
    forward_window: int = 20,
) -> pd.Series:
    """计算 as_of 之后 forward_window 天的累计收益.

    实际: 用 (as_of + forward_window) 的 close / as_of 的 close - 1
    """
    sub = nav_df.loc[as_of:, [c for c in all_codes if c in nav_df.columns]]
    if len(sub) < 2:
        return pd.Series(dtype=float)
    base = sub.iloc[0]
    # 取 forward_window 之后的价格
    if len(sub) > forward_window:
        future = sub.iloc[forward_window]
    else:
        # 没有足够未来数据, 返回空
        return pd.Series(dtype=float)
    return future / base - 1.0


def factor_ic_at(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    all_codes: Sequence[str],
    forward_window: int = 20,
    lookback: int = 60,
) -> dict[str, float]:
    """计算 as_of 当天的 6 因子 IC.

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
    """滚动计算 6 因子 IC 序列.

    Args:
        nav_df: 价格面板
        all_codes: ETF codes
        start/end: 日期范围
        window: 滚动窗口 (60d)
        forward_window: 预测未来收益窗口 (20d)
        step: 采样步长 (5d, 避免太密)
        lookback: 因子 lookback (60d)

    Returns:
        pd.DataFrame, index=date, columns=factor name
    """
    dates = nav_df.loc[start:end].index
    # 截止日期需要留出 forward_window
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
        # 确保有足够历史
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
    smooth_window: int = 12,  # ~ 60 天 (5d step)
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

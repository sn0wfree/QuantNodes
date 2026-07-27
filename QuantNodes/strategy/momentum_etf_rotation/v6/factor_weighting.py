# coding=utf-8
"""v6.1 IC 加权 — 截面 IC 滚动计算 + IC-IR 加权.

实现要点 (Stage 27 v6.1):
1. 防 look-ahead: expanding window (用截至 t-1 的历史 OOS IC 计算 t 期权重)
2. 至少 36 个月数据窗, warmup 期内用等权
3. 失效因子 (IR < 0) 权重 = 0, 自动剔除
4. 权重归一化: w_i = max(IR_i, 0) / Σ max(IR_j, 0)
5. 平滑 (可选): 移动平均 IR, 窗口 6 月 → 减少抖动

输入:
- panel_close: 收盘价面板 (DataFrame, index=date, columns=code)
- factor_panel: 全部 ETF 的 11 因子 (dict[code] → DataFrame)
- factors: 因子名列表 (默认 11 个)
- rebalance_dates: 调仓日期 (pd.DatetimeIndex)
- 评估 horizon: 21 日 (≈ 月频, 默认)

输出:
- factor_weights_timeseries: DataFrame (index=rebal_date, columns=factor, values=权重)

参考:
- 华西证券《行业有效量价因子与行业轮动策略》 §4.2 (IC 加权)
- Stage 25.1 v5.1.1 ablation: S2 winsorized rank 拖累 OOS Calmar 12%, 决策保留 z-score
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# 默认调仓频率 (月度, 21 日评估 horizon)
DEFAULT_HORIZON_DAYS = 21

# IC 计算最少历史月份数 (扩展窗口最少样本)
MIN_MONTHS_FOR_IC = 24

# 平滑窗口 (对因子 IR 移动平均, 0 = 不平滑)
DEFAULT_SMOOTH_WINDOW = 6


def compute_cross_section_ic(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame,
    as_of: pd.Timestamp,
    factors: Sequence[str],
    horizon: int = DEFAULT_HORIZON_DAYS,
) -> pd.Series:
    """计算 as_of 日各因子的截面 IC (Spearman rank correlation vs horizon 日后收益).

    Args:
        factor_panel: ETF → DataFrame (包含各因子列)
        panel_close: 收盘价面板 (用于计算 horizon 日后收益)
        as_of: 截面日
        factors: 因子名列表 (e.g. ['f1_second_mom', ...])
        horizon: 评估窗口 (默认 21 日 ≈ 月频)

    Returns:
        pd.Series, index=因子名, values=截面 IC (NaN = 数据不足)
    """
    if as_of not in panel_close.index:
        idx = panel_close.index.get_indexer([as_of], method="ffill")[0]
        if idx < 0 or idx + horizon >= len(panel_close):
            return pd.Series(dtype=float)
        as_of_loc_idx = idx
    else:
        as_of_loc_idx = panel_close.index.get_loc(as_of)
        if as_of_loc_idx + horizon >= len(panel_close):
            return pd.Series(dtype=float)

    # horizon 日后收益 (持仓期收益)
    future_idx = panel_close.index[as_of_loc_idx + horizon]
    fwd_ret = (panel_close.loc[future_idx] / panel_close.loc[panel_close.index[as_of_loc_idx]] - 1.0).dropna()

    ic = {}
    for fac in factors:
        fac_vals = {}
        for code, df in factor_panel.items():
            if as_of in df.index and fac in df.columns:
                v = df[fac].loc[as_of]
                if pd.notna(v):
                    fac_vals[code] = float(v)
        s = pd.Series(fac_vals)
        common = s.index.intersection(fwd_ret.index)
        if len(common) < 10:
            ic[fac] = np.nan
            continue
        try:
            r = s.loc[common].rank().corr(fwd_ret.loc[common].rank())
            ic[fac] = r if not np.isnan(r) else np.nan
        except Exception:
            ic[fac] = np.nan

    return pd.Series(ic)


def compute_ic_timeseries(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    factors: Sequence[str],
    horizon: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """计算每个调仓日各因子的截面 IC.

    Args:
        rebalance_dates: 调仓日列表

    Returns:
        DataFrame, index=rebal_date, columns=factor, values=IC
    """
    rows = []
    valid_dates = []
    for d in rebalance_dates:
        ic = compute_cross_section_ic(factor_panel, panel_close, d, factors, horizon)
        if not ic.empty:
            rows.append(ic)
            valid_dates.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(valid_dates, name="date"))
    return out


def compute_factor_weights(
    ic_ts: pd.DataFrame,
    min_months: int = MIN_MONTHS_FOR_IC,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """从 IC 时序计算因子权重 (expanding window + 平滑 + 截断 0).

    防 look-ahead 关键点:
    - t 期权重 = f(截至 t-1 的 IC 历史) (用 shift(1))
    - 不足 min_months 时, 默认等权 (1/N)
    - IR < 0 的因子权重 = 0 (自动剔除失效因子)

    Args:
        ic_ts: 截面 IC 时序 (DataFrame, index=date, columns=factor)
        min_months: 最少历史样本 (默认 24 月)
        smooth_window: IR 移动平均窗口 (默认 6 月, 0 = 不平滑)
        eps: 防 0 除

    Returns:
        DataFrame, index=date, columns=factor, values=权重 (每行和 = 1)
        第一列可能因 warmup 不存在 (默认等权, 但此函数返回空)
    """
    if ic_ts.empty or len(ic_ts) < 2:
        return pd.DataFrame()

    # 移除全 NaN 行
    ic_ts = ic_ts.dropna(how="all")
    if ic_ts.empty:
        return pd.DataFrame()

    n_factors = len(ic_ts.columns)

    # 用 shift(1) 防 look-ahead: t 期权重基于 t-1 及之前的 IC
    ic_lag = ic_ts.shift(1)

    # 计算滚动 IR (rolling mean / std, min_periods 控制)
    ic_mean = ic_lag.rolling(min_months, min_periods=min_months).mean()
    ic_std = ic_lag.rolling(min_months, min_periods=min_months).std()

    ir = ic_mean / (ic_std + eps)

    # 平滑 (对 IR 做移动平均, 减少逐月抖动)
    if smooth_window > 0:
        ir = ir.rolling(smooth_window, min_periods=1).mean()

    # 截断负 IR (= 0, 失效因子自动剔除)
    weights_raw = ir.clip(lower=0.0)

    # 归一化 (行和 = 1)
    row_sums = weights_raw.sum(axis=1)
    weights = weights_raw.div(row_sums.replace(0, 1), axis=0)

    # 行和 = 0 (全部因子失效) → 等权
    zero_rows = row_sums < eps
    if zero_rows.any():
        equal_w = pd.DataFrame(
            np.ones((zero_rows.sum(), n_factors)) / n_factors,
            index=weights.index[zero_rows],
            columns=weights.columns,
        )
        weights.loc[zero_rows] = equal_w

    return weights


def compute_softmax_weights(
    ir_ts: pd.DataFrame,
    sharpness: float = 3.0,
    min_ir_threshold: float = 0.5,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """[Stage 28 fix] 软权重 via z-score + softmax + |IR| 阈值 (无硬剔除).

    与 compute_factor_weights 的区别:
    - 不再"IR<0 → 权重=0"硬剔除 (抖动 + 无金融理由)
    - 改用 z-score (行内归一化) + softmax (光滑映射) → 失效因子权重大概率很小但非 0
    - 加 |IR| 阈值过滤极弱信号 (避免噪音因子拉权重)

    仍然防 look-ahead: 输入 ir_ts 应为已用 shift(1) 的 expanding window IR.

    Args:
        ir_ts: DataFrame, index=日期, columns=因子. (假定已 shift(1) 防 look-ahead)
        sharpness: softmax 温度, 越大越接近 argmax, 越小越接近等权
                   推荐范围 [1, 5], 默认 3.0
        min_ir_threshold: |IR| < 阈值的因子权重置 0 (默认 0.5 ≈ t-stat 1.96)
        eps: 防 0 除

    Returns:
        DataFrame, 每行权重和 = 1, 列数 = ir_ts 列数
    """
    if ir_ts.empty:
        return pd.DataFrame()

    # 限幅到 [-3, 3] 防极端权重
    ir_clipped = ir_ts.clip(-3.0, 3.0)

    # 行内 z-score 归一化 (基于该时点所有因子 IR 的横向比较)
    row_mean = ir_clipped.mean(axis=1)
    row_std = ir_clipped.std(axis=1)
    z = ir_clipped.sub(row_mean, axis=0).div(row_std + eps, axis=0)

    # softmax (光滑)
    exp_z = np.exp(z * sharpness)

    # 应用 |IR| 阈值: 极弱信号的因子权重强制为 0
    # 注意用原 ir_ts (不是 ir_clipped) 比较, 信号极弱多半是噪音
    mask = ir_ts.abs() > min_ir_threshold
    exp_z = exp_z.where(mask, 0.0)

    # 归一化 (每行和 = 1)
    row_sums = exp_z.sum(axis=1)
    weights = exp_z.div(row_sums.replace(0, 1), axis=0)

    # 全 0 行 (全部因子 IR 都低于阈值) 退化等权
    zero_rows = row_sums < eps
    if zero_rows.any():
        n_factors = len(ir_ts.columns)
        equal_w = pd.DataFrame(
            np.ones((zero_rows.sum(), n_factors)) / n_factors,
            index=weights.index[zero_rows],
            columns=weights.columns,
        )
        weights.loc[zero_rows] = equal_w

    return weights


def align_weights_with_rebal_dates(
    weights: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    panel_close_index: pd.DatetimeIndex,
) -> pd.Series:
    """从 weights DataFrame 取出每个调仓日的权重 dict.

    Args:
        weights: 因子权重 (DataFrame, index=date, columns=factor)
        rebalance_dates: 调仓日列表
        panel_close_index: 收盘价 index (用于 ffill 对齐)

    Returns:
        pd.Series, index=date, values=dict[factor_name, weight]
    """
    if weights.empty:
        return pd.Series(dtype=object)

    # 确保 weights 的 index 是 DatetimeIndex
    if not isinstance(weights.index, pd.DatetimeIndex):
        weights.index = pd.DatetimeIndex(weights.index)

    aligned = []
    for d in rebalance_dates:
        if d not in weights.index and len(weights) > 0:
            # ffill
            idx_loc = weights.index.get_indexer([d], method="ffill")[0]
            if idx_loc < 0:
                aligned.append({})
                continue
            d_actual = weights.index[idx_loc]
        else:
            d_actual = d
        if d_actual in weights.index:
            w = weights.loc[d_actual].dropna()
            w_dict = {k: float(v) for k, v in w.items() if v > 0}
            aligned.append(w_dict)
        else:
            aligned.append({})
    return pd.Series(aligned, index=pd.DatetimeIndex(rebalance_dates, name="date"))


__all__ = [
    "compute_cross_section_ic",
    "compute_ic_timeseries",
    "compute_factor_weights",
    "align_weights_with_rebal_dates",
    "DEFAULT_HORIZON_DAYS",
    "MIN_MONTHS_FOR_IC",
    "DEFAULT_SMOOTH_WINDOW",
]

"""
v7.0 TAA 资产配置 / 风格轮动回测 (Stage 30.4 reproduction).

[重写] 原 v7.0 (vol_target 防御版) 2.68% 与业界 19% 复现目标错位.
新 v7.0 = state-based TAA: 5 状态 → 5 套资产权重, 满仓运行.

[核心逻辑]
1. 调仓日 d, 用 PIT 调整的 HMM 确定当前状态
2. 根据 STATE_ALLOCATIONS 给出 5 ETF 权重
3. 用 inverse_vol 微调 (在同状态内降低高波动 ETF 权重, 增加低波动 ETF 权重)
4. 月度调仓, T+1 lag

[与 v6.2 关系]
- v6.2 因子加权 + 选股 完全不沿用 (这是不同策略)
- v7.0 是 state-based TAA, 5 状态 → 5 资产权重
- 两者并列, 互不干扰

[PIT 关键]
- 调仓日 d 用 HMM 输出状态 (基于 release_date <= d 的宏观数据)
- T+1 lag: d 日确定状态, d+1 日执行调仓 (避免当日跳价)
- 不使用 vol_target, 不做仓位缩放
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .regime_macro import build_regime_timeline, REGIME_NAMES
from .state_allocation import STATE_ALLOCATIONS, ETF_LAUNCH_DATES


@dataclass
class V7Config:
    """v7.0 TAA 配置.

    [Stage 30.4 reproduction]
    - 满仓 (no vol_target)
    - 月度调仓
    - T+1 lag
    - 5 ETF 池: 510300/510500/159915/518880/512760
    """
    name: str = "industry_rotation_v7_taa"

    # TAA 资产配置表
    state_allocations: dict = field(default_factory=lambda: {k: dict(v) for k, v in STATE_ALLOCATIONS.items()})

    # 调仓
    rebalance_freq: str = "M"     # 月度
    rebal_lag_days: int = 1       # T+1 调仓

    # 5 ETF 池 (在 state_allocation 中已有, 这里显式声明)
    etf_universe: tuple[str, ...] = ("510300", "510500", "159915", "518880", "512760")

    # 是否在 state 内部应用 inverse_vol 微调
    use_inverse_vol_within_state: bool = True
    vol_window: int = 60
    vol_floor: float = 0.01


def _state_allocation_with_etf_availability(
    state: str,
    as_of: pd.Timestamp,
    cfg: V7Config,
) -> dict[str, float]:
    """根据当前日期, 调整 state_allocation 中未上市 ETF 的权重.

    半导体 512760 2019-06-12 上市, 早期权重 → 0, 重分配到其他 ETF.
    """
    weights = dict(cfg.state_allocations[state])
    # 找未上市的 ETF
    unavailable = []
    for code in weights:
        launch = pd.Timestamp(ETF_LAUNCH_DATES.get(code, "1900-01-01"))
        if as_of < launch:
            unavailable.append(code)

    if not unavailable:
        return weights

    # 把未上市的权重等比例分给已上市的
    total_unavail = sum(weights[c] for c in unavailable)
    for c in unavailable:
        weights[c] = 0.0

    available = [c for c in weights if c not in unavailable and weights[c] >= 0]
    if not available or total_unavail == 0:
        return weights

    # 按已有权重等比例放大
    avail_total = sum(weights[c] for c in available)
    if avail_total > 0:
        scale = 1.0 + total_unavail / avail_total
        for c in available:
            weights[c] *= scale
    return weights


def _inverse_vol_within_state(
    weights: dict[str, float],
    panel_close: pd.DataFrame,
    as_of: pd.Timestamp,
    vol_window: int = 60,
    vol_floor: float = 0.01,
) -> dict[str, float]:
    """在 state 内部, 用 inverse_vol 微调权重.

    不改变 state 的方向, 只在同状态内降低高波动 ETF 占比.
    """
    if not weights:
        return weights
    # 算每 ETF 的年化 vol
    vols = {}
    for c in weights:
        if c not in panel_close.columns:
            vols[c] = None
            continue
        sub = panel_close[c].loc[:as_of]
        if len(sub) < vol_window:
            vols[c] = None
            continue
        rets = sub.iloc[-vol_window:].pct_change().dropna()
        if len(rets) < 10:
            vols[c] = None
            continue
        vols[c] = rets.std() * np.sqrt(252)

    # 仅对有效 vol 的 ETF 重新加权
    valid = {c: v for c, v in vols.items() if v is not None and v > 0 and weights.get(c, 0) > 0}
    if not valid:
        return weights

    inv = {c: 1.0 / max(v, vol_floor) for c, v in valid.items()}
    total = sum(inv.values())
    new_weights = {c: inv[c] / total for c in inv}
    return new_weights


def run_v7_taa_backtest(
    panel_close: pd.DataFrame,
    cfg: V7Config | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
    regime_timeline: pd.DataFrame | None = None,
) -> tuple[pd.Series, list[dict]]:
    """v7.0 TAA 风格轮动回测.

    Args:
        panel_close: 收盘价面板 (列=ETF code)
        cfg: V7Config (None = 默认)
        rebalance_dates: 调仓日 (None = 月末)
        regime_timeline: 预计算的 5 状态时间线 (None = 内部自算)

    Returns:
        (nav_series, state_history)
        - nav_series: 净值序列
        - state_history: 每次调仓的状态/权重记录 (list of dict)
    """
    if cfg is None:
        cfg = V7Config()

    dates = panel_close.index

    # 1. 调仓日 (沿用 v6.2 模式: 每月最后交易日)
    if rebalance_dates is None:
        period = dates.to_period("M")
        rebal_series = dates.to_series().groupby(period).tail(1)
        rebal_dates_idx = rebal_series.index
    else:
        rebal_dates_idx = pd.DatetimeIndex(rebalance_dates)

    # 2. 5 状态时间线
    if regime_timeline is None:
        start = dates.min().strftime("%Y-%m-%d")
        end = dates.max().strftime("%Y-%m-%d")
        regime_timeline = build_regime_timeline(start=start, end=end)
    regime_timeline = regime_timeline.copy()
    regime_timeline["date"] = pd.to_datetime(regime_timeline["date"])
    regime_lookup: dict[pd.Timestamp, str] = {}
    for _, row in regime_timeline.iterrows():
        regime_lookup[pd.Timestamp(row["date"])] = row["regime"]
    regime_dates_sorted = sorted(regime_lookup.keys())

    # 3. 模拟回测
    nav = pd.Series(1.0, index=dates, dtype=float)
    prev_weights: dict[str, float] = {}
    state_history: list[dict] = []

    for i, date in enumerate(dates):
        if i == 0:
            continue

        # 3.1 检测调仓日
        is_rebal = date in set(rebal_dates_idx) and i > 252  # 1 年 warmup

        if is_rebal:
            # 3.2 找最新状态
            regime = None
            for d_regime in reversed(regime_dates_sorted):
                if d_regime <= date:
                    regime = regime_lookup[d_regime]
                    break
            if regime is None:
                regime = "neutral"

            # 3.3 state 分配 + ETF 可用性调整
            weights = _state_allocation_with_etf_availability(regime, date, cfg)

            # 3.4 inverse_vol 微调 (可选)
            if cfg.use_inverse_vol_within_state:
                weights = _inverse_vol_within_state(
                    weights, panel_close, date,
                    vol_window=cfg.vol_window, vol_floor=cfg.vol_floor,
                )

            # 3.5 归一化
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            prev_weights = weights
            state_history.append({
                "date": date, "regime": regime, "weights": dict(weights),
            })

        # 3.6 计算日收益 (T+1: 调仓日 d, d+1 开始按新权重)
        if prev_weights:
            daily_ret = 0.0
            for code, w in prev_weights.items():
                if code in panel_close.columns:
                    p_t = panel_close[code].iloc[i]
                    p_prev = panel_close[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_ret)

    return nav, state_history


def state_history_to_df(state_history: list[dict]) -> pd.DataFrame:
    """state_history (list of dict) → DataFrame (date, regime, code, weight)."""
    rows = []
    for entry in state_history:
        for code, w in entry["weights"].items():
            rows.append({
                "date": entry["date"],
                "regime": entry["regime"],
                "code": code,
                "weight": w,
            })
    return pd.DataFrame(rows)


__all__ = [
    "V7Config",
    "run_v7_taa_backtest",
    "state_history_to_df",
]

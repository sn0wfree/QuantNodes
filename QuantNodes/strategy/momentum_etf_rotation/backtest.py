# coding=utf-8
"""端到端回测: 月度调仓循环, 输出业绩 + 与 CICC 基线对照.

不依赖任何外部数据 (除可选 akshare), 全部用 tests/fixtures 合成数据驱动.
基线对照范围:
    纯动量轮动 (逆波动):  Calmar 0.76, 最大回撤 -18.78%
    等权动量轮动:           Calmar 0.51, 最大回撤 -26.56%
    沪深 300 (同期):        Calmar 0.08, 最大回撤 -45.60%
    80/20 固收+:            年化 6.34%, Calmar 1.73, 2025 YTD 年化 7.77%, DD -1.48%

基线容忍: ±20% 视为复现成功 (与 CICC ETF 池"近似"而非一一对应).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
import pandas as pd

from .common.fixed_income_plus import (
    FixedIncomePlus,
    FixedIncomePlusConfig,
    FixedIncomePlusResult,
)
from .common.metrics import performance_metrics_legacy as performance_metrics
from .portfolio import (
    DiversificationCaps,
    PortfolioState,
    RotationConfig,
    apply_stops,
    apply_vol_targeting,
    calculate_turnover_cost,
    equal_weights,
    inverse_vol_weights,
    select_and_weight,
)
from .common.universe import ETFPool


@dataclass
class BacktestConfig:
    rotation: RotationConfig = field(default_factory=RotationConfig)
    freq: str = "ME"                       # 调仓频率
    init_value: float = 1.0


@dataclass
class RotationBacktestResult:
    """纯动量轮动 (无债券) 回测结果."""
    nav: pd.Series
    states: list[PortfolioState] = field(default_factory=list)
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def run_rotation_backtest(
    etf_nav: pd.DataFrame,
    pool: ETFPool,
    cfg: BacktestConfig | None = None,
) -> RotationBacktestResult:
    """纯动量轮动回测 (用于与 沪深 300 / 等权 对照)."""
    cfg = cfg or BacktestConfig()
    rot = cfg.rotation
    etf = etf_nav.dropna(how="all")
    etf_norm = etf / etf.iloc[0]

    dates = etf.index
    # 调仓日: 每月最后交易日 (用 Series 而非 resample, 避免日历标签 ≠ 实际交易日)
    rebal_dates = pd.Series(dates).groupby(dates.to_period("M")).max().tolist()
    lookback = rot.lookback
    valid = [d for d in rebal_dates if dates.searchsorted(d) >= lookback]
    if not valid:
        raise ValueError(f"数据不足: 需要 {lookback} 天")
    rebal_dates = valid
    first_rebal = rebal_dates[0]
    first_pos = dates.searchsorted(first_rebal)
    sim_start = dates[max(0, first_pos - lookback)]
    dates = dates[dates >= sim_start]
    etf_norm = etf_norm.loc[dates]

    nav = np.ones(len(dates))
    prev_weights: dict[str, float] = {}
    states: list[PortfolioState] = []
    actual: list[pd.Timestamp] = []

    # Stage 9-D: 训练 HMM regime 检测器
    detector = None
    if rot.regime_detector is not None and rot.regime_detector.enabled:
        from .regime_detector import HMMRegimeDetector, get_regime_params
        detector = HMMRegimeDetector(
            n_regimes=rot.regime_detector.n_regimes,
            lookback_train=rot.regime_detector.lookback_train,
        )
        # 用前 lookback_train 天训练
        if rot.regime_detector.benchmark_code in etf_norm.columns:
            train_nav = etf_norm[rot.regime_detector.benchmark_code]
            if len(train_nav) >= rot.regime_detector.lookback_train:
                try:
                    detector.fit(train_nav)
                except Exception as e:
                    print(f"HMM 训练失败: {e}, 禁用 detector")
                    detector = None
        else:
            detector = None

    for i, date in enumerate(dates):
        if date in rebal_dates:
            # Stage 9-D: 根据 regime 动态调整参数
            rot_eff = rot
            if detector is not None and i > 0:
                try:
                    regime = detector.predict(etf_norm[rot.regime_detector.benchmark_code].loc[:date])
                    overrides = get_regime_params(rot_eff, regime, rot.regime_detector.regime_params)
                    rot_eff = replace(rot, **overrides)
                except Exception:
                    rot_eff = rot

            if prev_weights:
                state = apply_stops(etf_norm, pool, rot_eff, prev_weights, date)
            else:
                state = select_and_weight(etf_norm, pool, rot_eff, date)
            if not state.weights:
                state.weights = equal_weights(etf_norm.columns.tolist())
            total = sum(state.weights.values())
            if total > 0:
                state.weights = {k: v / total for k, v in state.weights.items()}

            # Stage 9-C: 波动率目标缩放 (缩放后不归一化, 让 cash 体现缩放)
            if rot.vol_targeting.enabled and i > 0:
                nav_series_so_far = pd.Series(nav[:i+1], index=dates[:i+1])
                apply_vol_targeting(rot, nav_series_so_far, date, state)

            prev_weights = state.weights
            states.append(state)
            actual.append(date)
            if i > 0:
                nav[i] = nav[i - 1]  # 调仓日 NAV 不变
            else:
                nav[i] = 1.0

            # Stage 13: 交易成本扣减 (在调仓日, 计算换手率并扣减成本)
            if rot.cost_model.enabled and i > 0:
                # 计算换手率 = 0.5 * Σ |new_w - old_w|
                if len(states) >= 2:
                    old_w = states[-2].weights
                    new_w = state.weights
                    all_codes = set(old_w.keys()) | set(new_w.keys())
                    turnover = sum(abs(new_w.get(c, 0) - old_w.get(c, 0))
                                   for c in all_codes) / 2
                    cost = calculate_turnover_cost(turnover, rot.cost_model)
                    nav[i] = nav[i] * (1 - cost)
        else:
            if i > 0 and prev_weights:
                daily_ret = 0.0
                for code, w in prev_weights.items():
                    if code in etf_norm.columns:
                        col = etf_norm[code]
                        # 防御: 列重复时 .iloc[i] 返回 Series
                        a = col.iloc[i] if hasattr(col, 'iloc') else col
                        b = col.iloc[i - 1] if hasattr(col, 'iloc') else col
                        if isinstance(a, pd.Series): a = a.iloc[0]
                        if isinstance(b, pd.Series): b = b.iloc[0]
                        if not pd.isna(a) and not pd.isna(b) and b != 0:
                            daily_ret += w * (a / b - 1)
                nav[i] = nav[i - 1] * (1 + daily_ret)
            else:
                nav[i] = 1.0 if i == 0 else nav[i - 1]

    nav_series = pd.Series(nav, index=dates, name="rotation")
    return RotationBacktestResult(
        nav=nav_series,
        states=states,
        rebalance_dates=actual,
        metrics=performance_metrics(nav_series),
    )


def run_equal_weight_baseline(
    etf_nav: pd.DataFrame,
    pool: ETFPool,
    cfg: BacktestConfig | None = None,
) -> RotationBacktestResult:
    """等权对照: 选 top_n 后等权, 不做逆波动."""
    cfg = cfg or BacktestConfig()
    rot = cfg.rotation
    # 临时把 weight_method 改 equal
    rot_eq = RotationConfig(
        lookback=rot.lookback, top_n=rot.top_n, corr_threshold=rot.corr_threshold,
        corr_window=rot.corr_window, ma_window=rot.ma_window, rank_cutoff=rot.rank_cutoff,
        diversification=rot.diversification, weight_method="equal",
        vol_window=rot.vol_window, weight_floor=rot.weight_floor, min_history=rot.min_history,
    )
    return run_rotation_backtest(etf_nav, pool, BacktestConfig(rotation=rot_eq, freq=cfg.freq))


def compare_to_cicc(
    result_inv_vol: RotationBacktestResult,
    result_equal: RotationBacktestResult,
    result_fi_plus: FixedIncomePlusResult | None = None,
) -> pd.DataFrame:
    """生成与 CICC 报告数字的对照表.

    CICC 公布 (全区间):
        逆波动:    最大回撤 -18.78%, Calmar 0.76
        等权:      最大回撤 -26.56%, Calmar 0.51
        沪深 300:  最大回撤 -45.60%, Calmar 0.08
        80/20 固收+: 年化 6.34%, Calmar 1.73
    """
    rows = [
        {
            "策略": "动量轮动(逆波动)",
            "本实现 Calmar": result_inv_vol.metrics["calmar"],
            "CICC Calmar": 0.76,
            "本实现 最大回撤": result_inv_vol.metrics["max_drawdown"],
            "CICC 最大回撤": -0.1878,
        },
        {
            "策略": "动量轮动(等权)",
            "本实现 Calmar": result_equal.metrics["calmar"],
            "CICC Calmar": 0.51,
            "本实现 最大回撤": result_equal.metrics["max_drawdown"],
            "CICC 最大回撤": -0.2656,
        },
    ]
    if result_fi_plus is not None:
        m = performance_metrics(result_fi_plus.nav)
        rows.append({
            "策略": "80/20 固收+",
            "本实现 年化": m["ann_return"],
            "CICC 年化": 0.0634,
            "本实现 Calmar": m["calmar"],
            "CICC Calmar": 1.73,
        })
    return pd.DataFrame(rows)


# CICC 基线常数 (供测试断言 import)
CICC_BASELINES = {
    "inv_vol_calmar": 0.76,
    "inv_vol_max_dd": -0.1878,
    "equal_calmar": 0.51,
    "equal_max_dd": -0.2656,
    "csi300_calmar": 0.08,
    "csi300_max_dd": -0.4560,
    "fi_plus_ann_return": 0.0634,
    "fi_plus_calmar": 1.73,
    "fi_plus_2025_ann": 0.0777,
    "fi_plus_max_dd": -0.0148,
}

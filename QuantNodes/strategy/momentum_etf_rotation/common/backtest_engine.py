# coding=utf-8
"""统一回测引擎.

消除 v1-v7 中 8 个 backtest 文件的重复主循环.

架构:
  - BacktestCallbacks: 回调基类 (版本特定逻辑通过继承覆盖)
  - BacktestResult: 结果容器 (始终包含日频 NAV)
  - run_backtest(): 统一主循环

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_engine import (
        BacktestCallbacks, BacktestResult, run_backtest,
    )

    class V1Callbacks(BacktestCallbacks):
        def compute_signals(self, data, date, state, context):
            return momentum_scores(data, date)

    config = BacktestConfig(rebal_freq="M", top_n=10, cost=CostConfig(enabled=False))
    result = run_backtest(price_panel, daily_returns, config, V1Callbacks())
    print(result.nav_daily)  # 日频 NAV
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .backtest_config import BacktestConfig, CostConfig
from .backtest_utils import (
    apply_max_weight,
    calculate_turnover,
    calculate_turnover_cost,
    generate_rebalance_dates,
    normalize_weights,
)


# ============================================================
# 1. 回调基类
# ============================================================
class BacktestCallbacks:
    """回调基类. 版本特定逻辑通过继承覆盖.

    默认实现提供最简单的行为 (等权, 无风控).
    各版本只需覆盖需要自定义的方法.
    """

    def compute_signals(
        self,
        price_panel: pd.DataFrame,
        date: pd.Timestamp,
        state: dict,
        context: dict,
    ) -> dict[str, float]:
        """计算信号分数 (调用日).

        Parameters:
            price_panel: 价格面板
            date: 当前调仓日
            state: {"prev_weights": dict, "nav": np.ndarray}
            context: 预计算数据 (如因子面板, HMM 状态等)

        Returns:
            {code: score} 信号分数 (越高越优先选择)
        """
        raise NotImplementedError("Subclasses must implement compute_signals")

    def select_assets(
        self,
        signals: dict[str, float],
        config: BacktestConfig,
    ) -> list[str]:
        """选择资产 (调用日).

        默认: 按分数降序取 top_n.

        Parameters:
            signals: {code: score}
            config: 回测配置

        Returns:
            选中的资产代码列表
        """
        sorted_codes = sorted(signals, key=signals.get, reverse=True)
        return sorted_codes[:config.top_n]

    def compute_weights(
        self,
        selected: list[str],
        price_panel: pd.DataFrame,
        date: pd.Timestamp,
        config: BacktestConfig,
    ) -> dict[str, float]:
        """计算权重 (调用日).

        默认: 等权. 覆盖以实现 inverse-vol 等.

        Parameters:
            selected: 选中的资产代码
            price_panel: 价格面板
            date: 当前调仓日
            config: 回测配置

        Returns:
            {code: weight} 权重
        """
        n = len(selected)
        if n == 0:
            return {}
        return {c: 1.0 / n for c in selected}

    def apply_risk_controls(
        self,
        weights: dict[str, float],
        nav_history: pd.Series,
        date: pd.Timestamp,
        config: BacktestConfig,
    ) -> dict[str, float]:
        """应用风控 (调用日).

        默认: 无操作. 覆盖以实现 VT, TF, stop-loss 等.

        Parameters:
            weights: 当前权重
            nav_history: 历史 NAV Series (DatetimeIndex, 用于 VT rolling vol)
            date: 当前调仓日
            config: 回测配置

        Returns:
            调整后的权重
        """
        return weights

    def post_weights(
        self,
        weights: dict[str, float],
        config: BacktestConfig,
    ) -> dict[str, float]:
        """权重后处理 (调用日).

        默认: max_weight 约束 + 归一化.

        Parameters:
            weights: 当前权重
            config: 回测配置

        Returns:
            处理后的权重
        """
        weights = apply_max_weight(weights, config.max_weight)
        return normalize_weights(weights)


# ============================================================
# 2. 结果容器
# ============================================================
@dataclass
class BacktestResult:
    """回测结果.

    Attributes:
        nav_daily: 日频 NAV (始终输出)
        weights_history: [(rebal_date, weights), ...] 权重历史
        rebalance_dates: 调仓日列表
        metrics: 日频指标 (ann_return, ann_vol, sharpe, max_dd, calmar)
    """
    nav_daily: pd.Series
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] = field(default_factory=list)
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ============================================================
# 3. 统一回测引擎
# ============================================================
def run_backtest(
    price_panel: pd.DataFrame,
    daily_returns: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    callbacks: BacktestCallbacks | None = None,
    context: dict | None = None,
) -> BacktestResult:
    """统一回测引擎.

    流程:
    1. 生成调仓日
    2. 每日循环: 调仓日 → 信号/选择/权重/风控/后处理; 非调仓日 → 日收益累积
    3. 计算指标

    Parameters:
        price_panel: (T_daily, N) 价格面板 (用于信号计算和日收益计算)
        daily_returns: (T_daily, N) 日频收益. None 则从 price_panel 自动计算
            (推荐 None, 避免用户 pct_change().fillna(0) 导致 NaN→0 错误)
        config: 回测配置
        callbacks: 回调 (版本特定逻辑)
        context: 预计算数据 (如因子面板, HMM 状态等)

    Returns:
        BacktestResult (含日频 NAV)
    """
    context = context or {}
    config = config or BacktestConfig()
    callbacks = callbacks or BacktestCallbacks()
    dates = price_panel.index

    # 自动计算日收益 (不做 fillna, 保持 NaN 语义与原始代码一致)
    if daily_returns is None:
        daily_returns = price_panel.pct_change(fill_method=None)

    # 1. 生成调仓日
    rebal_dates_list = generate_rebalance_dates(
        dates, config.rebal_freq, min_lookback=config.min_history
    )
    rebal_set = set(rebal_dates_list)

    # 2. 主循环: 逐日计算 (inline NAV, 避免 VT 看到零 vol)
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] = []
    prev_weights: dict[str, float] = {}
    nav_arr = np.ones(len(dates))

    for i, date in enumerate(dates):
        if date in rebal_set and i >= config.min_history:
            # 信号
            state = {"prev_weights": prev_weights, "nav": nav_arr[:i + 1]}
            signals = callbacks.compute_signals(price_panel, date, state, context)

            # 选择
            selected = callbacks.select_assets(signals, config)

            # 权重
            weights = callbacks.compute_weights(selected, price_panel, date, config)

            # 风控 (传入 pandas Series, VT 需要正确的 rolling vol)
            nav_series = pd.Series(nav_arr[:i + 1], index=dates[:i + 1])
            weights = callbacks.apply_risk_controls(weights, nav_series, date, config)

            # 后处理 (max_weight + normalize)
            weights = callbacks.post_weights(weights, config)

            # 记录
            weights_history.append((date, dict(weights)))
            prev_weights = weights

            # 调仓日: 扣成本, 不算日收益 (与 v1/v2 一致)
            if config.cost.enabled and len(weights_history) >= 2:
                old_w = weights_history[-2][1]
                new_w = weights_history[-1][1]
                turnover = calculate_turnover(old_w, new_w)
                cost = turnover * config.cost.cost_rate()
                nav_arr[i] = nav_arr[i - 1] * (1 - cost) if i > 0 else 1.0
            elif i > 0:
                nav_arr[i] = nav_arr[i - 1]
        else:
            # 非调仓日: 累积日收益
            if i > 0 and prev_weights:
                daily_ret = 0.0
                for code, w in prev_weights.items():
                    if code in daily_returns.columns:
                        ret = daily_returns.loc[date, code]
                        if pd.notna(ret):
                            daily_ret += w * ret
                nav_arr[i] = nav_arr[i - 1] * (1 + daily_ret)
            else:
                nav_arr[i] = 1.0 if i == 0 else nav_arr[i - 1]

    # 3. 构造日频 NAV Series
    nav_daily = pd.Series(nav_arr, index=dates, name="nav")

    # 4. 计算指标 (日频 NAV)
    from ..fi_plus import performance_metrics
    metrics = performance_metrics(nav_daily)

    return BacktestResult(
        nav_daily=nav_daily,
        weights_history=weights_history,
        rebalance_dates=[d for d, _ in weights_history],
        metrics=metrics,
    )

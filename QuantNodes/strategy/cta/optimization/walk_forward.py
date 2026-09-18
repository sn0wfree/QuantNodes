# coding=utf-8
"""Walk-forward 优化引擎.

方法论:
    滚动窗口: 2年训练 + 1年验证
    每个窗口内:
        1. 在训练期检测状态 (TSI)
        2. 对参数网格做全量搜索
        3. 在验证期应用最优参数, 测量性能
    窗口间: 收集每窗口的最优参数, 供稳定性分析

数据要求:
    prices: pd.DataFrame (T, N), 索引为 DatetimeIndex
    states: pd.DataFrame (T, N), 取值 STATE_*
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_risk import PositionLimitRisk
from ..factors.indicators import rsi as compute_rsi
from ..factors.volatility import vol_target_positions
from ..regime import STATE_RANGE, STATE_TREND_DOWN, STATE_TREND_UP, kaufman_adaptive_window
from .evaluator import EvalResult, evaluate_strategy
from .param_grid import PARAM_GRID, sample_grid


@dataclass
class WindowResult:
    """单个 walk-forward 窗口的结果."""

    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    best_params: dict[str, Any]
    train_calmar: float
    val_calmar: float
    val_sharpe: float
    val_max_dd: float
    val_coverage: float
    val_trades: int
    eval_result: EvalResult | None = None


def _split_windows(
    start: str, end: str, train_years: int = 2, val_years: int = 1,
) -> list[tuple[str, str, str, str]]:
    """划分 walk-forward 窗口.

    Returns:
        list of (train_start, train_end, val_start, val_end)
    """
    from dateutil.relativedelta import relativedelta

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    windows = []
    cur = s
    while True:
        train_s = cur
        train_e = train_s + relativedelta(years=train_years) - pd.Timedelta(days=1)
        val_s = train_e + pd.Timedelta(days=1)
        val_e = val_s + relativedelta(years=val_years) - pd.Timedelta(days=1)
        if val_e > e:
            break
        windows.append((
            str(train_s.date()), str(train_e.date()),
            str(val_s.date()), str(val_e.date()),
        ))
        cur = val_s
    return windows


def _backtest_single(
    product: str,
    prices: pd.DataFrame,
    states: pd.DataFrame,
    params: dict[str, Any],
    start: str,
    end: str,
    registry: CtaContractRegistry,
) -> CtaBacktestResult | None:
    """单次回测: 给定参数和时间区间."""
    price_slice = prices.loc[start:end]
    state_slice = states.loc[start:end]
    if len(price_slice) < 100:
        return None

    close = price_slice[product]
    cfg_ema_slow = params.get("ema_slow", 60)
    use_slope = params.get("use_ema_slope", False)
    rsi_win = params.get("rsi_window", 14)
    rsi_os = params.get("rsi_oversold", 30.0)
    rsi_ob = params.get("rsi_overbought", 70.0)
    tgt_vol = params.get("target_vol", 0.15)

    # 构造 entry 信号
    n = len(close)
    entry = pd.Series(0.0, index=close.index)
    slow_ema = close.ewm(span=cfg_ema_slow, adjust=False).mean()
    er = kaufman_adaptive_window(close, 5, 20, 30)
    for t in range(cfg_ema_slow, n):
        state = state_slice[product].iloc[t]
        h_raw = er.iloc[t]
        h = int(h_raw) if pd.notna(h_raw) else 5
        h = max(5, min(h, 20))
        fast = close.iloc[t - h + 1: t + 1].mean()
        slow = slow_ema.iloc[t]
        if state == STATE_TREND_UP:
            if use_slope:
                fast_prev = close.iloc[t - h: t].mean() if t >= h + 1 else fast
                entry.iloc[t] = 1.0 if fast > fast_prev else 0.0
            else:
                entry.iloc[t] = 1.0 if fast > slow else 0.0
        elif state == STATE_TREND_DOWN:
            if use_slope:
                fast_prev = close.iloc[t - h: t].mean() if t >= h + 1 else fast
                entry.iloc[t] = -1.0 if fast < fast_prev else 0.0
            else:
                entry.iloc[t] = -1.0 if fast < slow else 0.0
        elif state == STATE_RANGE:
            rsi_v = compute_rsi(close.iloc[:t + 1], rsi_win).iloc[t]
            if rsi_v < rsi_os:
                entry.iloc[t] = 1.0
            elif rsi_v > rsi_ob:
                entry.iloc[t] = -1.0

    entry_df = pd.DataFrame({product: entry})

    # vol-targeting → lots
    weights = vol_target_positions(
        entry_df, price_slice[[product]],
        target_vol=tgt_vol, vol_lookback=20, max_leverage=3.0,
    )
    equity = 2_000_000.0
    signal_on = (weights.abs() > 1e-9).astype(float)
    n_active = signal_on.sum(axis=1).clip(lower=1)
    spec = registry.get_or_default(product)
    notional = price_slice[product] * spec.contract_multiplier
    cap = signal_on[product] * equity / n_active
    with np.errstate(divide="ignore", invalid="ignore"):
        target = weights[product] * cap / notional
    lots = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()

    prices_map = {product: price_slice[product].tolist()}
    products_map = {product: product}
    dates = list(price_slice.index)

    class _Strat(CtaStrategy):
        def __init__(self):
            self._last = {}

        def target_positions(self, date, data):
            if date not in lots.index:
                return {}
            i = lots.index.get_loc(date)
            qty = float(lots.iloc[i])
            targets = {product: qty} if abs(qty) > 1e-9 else {}
            if targets == self._last:
                return {}
            self._last = targets
            return targets

    risk = PositionLimitRisk(products=products_map, registry=registry)
    try:
        result = run_cta_backtest(
            dates=dates, prices=prices_map, products=products_map,
            strategy=_Strat(),
            config=CtaBacktestConfig(initial_cash=equity, commission_rate=0.001),
            registry=registry, risk=risk,
        )
    except Exception:
        return None

    # 计算信号覆盖率
    coverage = float((entry != 0).sum() / len(entry))
    result.metrics["signal_coverage"] = coverage
    return result


def _select_score(metrics: dict, objective: str) -> float:
    """根据目标函数从 metrics 中挑选评分 (越大越好)."""
    if objective == "calmar":
        return metrics.get("Calmar", -999.0)
    if objective == "sharpe":
        return metrics.get("Sharpe", -999.0)
    if objective == "max_dd":
        # MaxDD 为负数, 越接近 0 越好, 取负号使越大=越好
        return -metrics.get("MaxDD", -1.0)
    raise ValueError(f"unknown objective: {objective!r}")


def grid_search(
    product: str,
    prices: pd.DataFrame,
    states: pd.DataFrame,
    param_grid: dict[str, list],
    start: str,
    end: str,
    registry: CtaContractRegistry,
    n_samples: int | None = None,
    objective: str = "calmar",
) -> tuple[dict, float]:
    """参数网格搜索: 在给定区间内找最优参数.

    Args:
        objective: 优化目标 ('calmar' / 'sharpe' / 'max_dd')

    Returns:
        (best_params, best_score) — 即使无可行解也返回最佳
    """
    if objective not in ("calmar", "sharpe", "max_dd"):
        raise ValueError(f"unknown objective: {objective!r}")
    combos = sample_grid("all", n_samples=n_samples)
    best_score = -999.0
    best_params = {}
    feasible_count = 0

    for params in combos:
        result = _backtest_single(
            product, prices, states, params, start, end, registry,
        )
        if result is None:
            continue
        score = _select_score(result.metrics, objective)
        # 宽松评估: 只要 score > 当前最佳就更新 (不强制可行性)
        if score > best_score:
            best_score = score
            best_params = params.copy()
        # 记录可行解数量
        eval_r = evaluate_strategy(
            calmar=result.metrics.get("Calmar", 0),
            sharpe=result.metrics.get("Sharpe", 0),
            max_dd=result.metrics.get("MaxDD", -1),
            signal_coverage=result.metrics.get("signal_coverage", 0),
            n_trades=len(result.trades),
        )
        if eval_r.feasible:
            feasible_count += 1

    if feasible_count == 0:
        # 无可行解时仍返回最佳 (用于后续分析)
        pass

    return best_params, best_score


@dataclass
class CalibrationResult:
    """标定结果."""

    product: str
    windows: list[WindowResult]
    calibrated_params: dict[str, dict[str, Any]]  # {regime: {param: value}}
    baseline_calmar: float = 0.0


def walk_forward(
    product: str,
    prices: pd.DataFrame,
    states: pd.DataFrame,
    param_grid: dict[str, list] | None = None,
    train_years: int = 2,
    val_years: int = 1,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    n_samples: int | None = None,
    objective: str = "calmar",
) -> list[WindowResult]:
    """Walk-forward 优化.

    Args:
        product: 品种代码
        prices: 价格面板 (T, N)
        states: 状态面板 (T, N)
        param_grid: 参数网格 (None = 使用默认)
        train_years: 训练窗口年数
        val_years: 验证窗口年数
        start/end: 回测区间
        n_samples: 参数采样数 (None = 全量)
        objective: 优化目标 ('calmar' / 'sharpe' / 'max_dd')

    Returns:
        list[WindowResult]: 每个窗口的结果
    """
    param_grid = param_grid or PARAM_GRID
    registry = CtaContractRegistry()
    windows = _split_windows(start, end, train_years, val_years)
    results = []

    for i, (train_s, train_e, val_s, val_e) in enumerate(windows):
        t0 = time.time()
        best_params, train_score = grid_search(
            product, prices, states, param_grid,
            train_s, train_e, registry, n_samples,
            objective=objective,
        )
        # 验证期
        val_result = _backtest_single(
            product, prices, states, best_params,
            val_s, val_e, registry,
        )
        elapsed = time.time() - t0  # noqa: F841
        if val_result is None:
            wr = WindowResult(
                window_id=i, train_start=train_s, train_end=train_e,
                val_start=val_s, val_end=val_e,
                best_params=best_params, train_calmar=train_score,
                val_calmar=0, val_sharpe=0, val_max_dd=-1,
                val_coverage=0, val_trades=0,
            )
        else:
            coverage = val_result.metrics.get("signal_coverage", 0)
            eval_r = evaluate_strategy(
                calmar=val_result.metrics.get("Calmar", 0),
                sharpe=val_result.metrics.get("Sharpe", 0),
                max_dd=val_result.metrics.get("MaxDD", -1),
                signal_coverage=coverage,
                n_trades=len(val_result.trades),
            )
            wr = WindowResult(
                window_id=i, train_start=train_s, train_end=train_e,
                val_start=val_s, val_end=val_e,
                best_params=best_params, train_calmar=train_score,
                val_calmar=val_result.metrics.get("Calmar", 0),
                val_sharpe=val_result.metrics.get("Sharpe", 0),
                val_max_dd=val_result.metrics.get("MaxDD", -1),
                val_coverage=coverage,
                val_trades=len(val_result.trades),
                eval_result=eval_r,
            )
        results.append(wr)
        print(f"  Window {i}: train={train_s}~{train_e}, val={val_s}~{val_e}, "
              f"train_score={train_score:.3f}, val_calmar={wr.val_calmar:.3f}, "
              f"feasible={wr.eval_result.feasible if wr.eval_result else 'N/A'}, "
              f"elapsed={time.time() - t0:.1f}s")

    return results


def run_calibration(
    product: str,
    prices: pd.DataFrame,
    states: pd.DataFrame,
    param_grid: dict[str, list] | None = None,
    train_years: int = 2,
    val_years: int = 1,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    n_samples: int | None = None,
    objective: str = "calmar",
) -> CalibrationResult:
    """完整标定流程: walk-forward + 稳定性分析 + 参数选择.

    Args:
        objective: 优化目标 ('calmar' / 'sharpe' / 'max_dd')

    Returns:
        CalibrationResult: 包含每窗口结果 + 最终标定参数
    """
    from .stability import analyze_stability
    print(f"=== 标定 {product} (objective={objective}) ===")
    windows = walk_forward(
        product, prices, states, param_grid,
        train_years, val_years, start, end, n_samples,
        objective=objective,
    )
    stable = analyze_stability(windows, param_grid or PARAM_GRID)
    # 构造标定参数字典
    calibrated = {}
    for param_name, info in stable.items():
        calibrated[param_name] = info["value"]
    print(f"  稳定参数: {sum(1 for v in stable.values() if v['stable'])}/{len(stable)}")
    return CalibrationResult(
        product=product, windows=windows,
        calibrated_params={"global": calibrated},
    )

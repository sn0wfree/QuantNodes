# coding=utf-8
"""华泰时序CTA对照实验 (消融研究).

研报: 华泰金工《时序CTA方法论综述》(2026-06-12), §"实证分析".

文章 §"实证分析" 提出三个对照组:
    1. 仅趋势 (trend_only): 假设市场恒为趋势, 只用均线突破开仓
    2. 仅震荡 (range_only): 假设市场恒为震荡, 只用 RSI 反转开仓
    3. 无 ATR 移动止损 (no_chandelier): 不考虑退出机制四层中的吊灯层

目的: 验证文章核心论点 — 复合策略 (状态识别 + 多类策略组合 + 多层退出)
    优于任何单类策略或简化的退出机制.

研报引用:
    - 沪深300ETF和黄金ETF: 验证趋势突破策略低胜率/高赔率, 震荡反转策略高胜率/低赔率
    - 国债ETF: 震荡反转策略不适用 (大部分时间趋势向上)
    - ATR移动止损: 不仅能截断大级别回撤, 同时弥补市场状态识别误差
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    run_cta_backtest,
)
from ..backtest.cta_risk import PositionLimitRisk
from ..data_loader import CtaDataLoader
from ..factors.indicators import rsi as compute_rsi
from ..factors.volatility import vol_target_positions
from .huatai_timing_cta import (
    HUATAI_PRODUCTS,
    HuataiTimingConfig,
    HuataiTimingCTAStrategy,
    _HuataiPanelStrategy,
)


@dataclass(frozen=True)
class AblationResult:
    """单个对照变体的回测结果."""

    variant: str
    metrics: dict
    nav: pd.Series
    num_trades: int
    signal_coverage: float  # 仓位非零的时间占比


# ──────────────────────────────────────────────────────────────────
# 三个对照变体的开仓信号构造
# ──────────────────────────────────────────────────────────────────


def _trend_only_signals(
    prices: pd.DataFrame,
    ema_fast_min: int,
    ema_fast_max: int,
    ema_slow: int,
    er_window: int,
) -> pd.DataFrame:
    """trend_only: 假设恒为 trend → 自适应均线交叉开仓 (不读状态)."""
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    slow_ema = prices.ewm(span=ema_slow, adjust=False).mean()
    from ..regime import kaufman_adaptive_window
    for col in prices.columns:
        close = prices[col]
        er_series = kaufman_adaptive_window(close, ema_fast_min, ema_fast_max, er_window)
        n = len(close)
        for t in range(n):
            if t < ema_slow:
                continue
            h_raw = er_series.iloc[t]
            h = int(h_raw) if pd.notna(h_raw) else ema_fast_min
            h = max(ema_fast_min, min(h, ema_fast_max))
            fast = close.iloc[t - h + 1: t + 1].mean()
            slow = slow_ema[col].iloc[t]
            if fast > slow:
                out.iat[t, out.columns.get_loc(col)] = 1.0
            elif fast < slow:
                out.iat[t, out.columns.get_loc(col)] = -1.0
    return out


def _range_only_signals(
    prices: pd.DataFrame,
    rsi_window: int,
    oversold: float,
    overbought: float,
) -> pd.DataFrame:
    """range_only: 假设恒为 range → RSI 超超反转开仓 (不读状态)."""
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        close = prices[col]
        rsi_series = compute_rsi(close, rsi_window)
        n = len(close)
        for t in range(n):
            if t < rsi_window:
                continue
            rsi_v = rsi_series.iloc[t]
            if np.isnan(rsi_v):
                continue
            if rsi_v < oversold:
                out.iat[t, out.columns.get_loc(col)] = 1.0
            elif rsi_v > overbought:
                out.iat[t, out.columns.get_loc(col)] = -1.0
    return out


# ──────────────────────────────────────────────────────────────────
# 三个对照变体: 各跑一次完整回测
# ──────────────────────────────────────────────────────────────────


def _run_with_entry(
    entry: pd.DataFrame,
    prices: pd.DataFrame,
    ohlc: pd.DataFrame,
    states: pd.DataFrame,
    cfg: HuataiTimingConfig,
    registry: CtaContractRegistry,
    disable_chandelier: bool = False,
) -> CtaBacktestResult:
    """按给定开仓信号 + 四层退出 (可关闭吊灯层) 跑回测."""
    cols = list(prices.columns)
    weights = vol_target_positions(
        entry, prices,
        target_vol=cfg.target_vol, vol_lookback=cfg.vol_lookback,
        max_leverage=cfg.max_leverage,
    )
    equity = cfg.initial_cash
    signal_on = (weights.abs() > 1e-9).astype(float)
    n_active = signal_on.sum(axis=1).clip(lower=1)
    lots = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    for product in weights.columns:
        price = prices[product]
        spec = registry.get_or_default(product)
        notional = price * spec.contract_multiplier
        capital = signal_on[product] * equity / n_active
        with np.errstate(divide="ignore", invalid="ignore"):
            target = weights[product] * capital / notional
        lots[product] = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()

    prices_map = {p: prices[p].tolist() for p in cols}
    products_map = {p: p for p in prices_map}
    dates = list(prices.index)

    chandelier_kw = {
        p: {
            "window": cfg.chandelier_window,
            "atr_window": cfg.chandelier_atr_window,
            "mult": cfg.chandelier_mult.get(p, 3.0),
        }
        for p in cols
    }
    from ..exits import DrawdownCircuitBreaker
    cb = DrawdownCircuitBreaker(
        threshold=cfg.drawdown_threshold,
        recovery_ratio=cfg.drawdown_recovery,
    )
    strategy = _HuataiPanelStrategy(
        positions=lots, ohlc=ohlc, states=states,
        chandelier_kwargs=chandelier_kw, breaker=cb,
        max_holding_days=cfg.max_holding_days,
    )
    # disable_chandelier: 在 evaluate_exit 的 chandelier_kwargs 中注入大乘数,
    # 使吊灯止损永远不会被触发 (mult=1e9)
    if disable_chandelier:
        for p in cols:
            strategy._chand[p]["mult"] = 1e9

    risk = PositionLimitRisk(products=products_map, registry=registry)
    return run_cta_backtest(
        dates=dates,
        prices=prices_map,
        products=products_map,
        strategy=strategy,
        config=CtaBacktestConfig(
            initial_cash=cfg.initial_cash,
            commission_rate=cfg.commission_rate,
        ),
        registry=registry,
        risk=risk,
    )


def _build_full_signals(
    strat: HuataiTimingCTAStrategy,
    prices: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """复刻主策略的开仓信号 (state-dependent), 复用其方法."""
    return strat._entry_signals() if strat._entry is not None else strat._entry_signals()


def run_trend_only(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 1: 仅趋势 (恒为 trend)."""
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = _trend_only_signals(
        prices, cfg.ema_fast_min, cfg.ema_fast_max, cfg.ema_slow, cfg.er_window,
    )
    return _run_with_entry(entry, prices, ohlc, states, cfg, strat._registry)


def run_range_only(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 2: 仅震荡 (恒为 range)."""
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = _range_only_signals(
        prices, cfg.rsi_window, cfg.rsi_oversold, cfg.rsi_overbought,
    )
    return _run_with_entry(entry, prices, ohlc, states, cfg, strat._registry)


def run_no_chandelier(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 3: 关闭 ATR 吊灯止损层 (其余3 层保留)."""
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = strat._entry if strat._entry is not None else strat._entry_signals()
    return _run_with_entry(
        entry, prices, ohlc, states, cfg, strat._registry,
        disable_chandelier=True,
    )


def run_multi_period(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 5: 多周期共振 (5/20/60 日 EMA 多数决) 作为趋势状态二次确认.

    区别于 full: 仅开仓信号不同, 退出四层不变.
    """
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = strat._entry_signals_multi_period()
    return _run_with_entry(entry, prices, ohlc, states, cfg, strat._registry)


def run_weighted_period(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 6: 加权多周期共振 (近期 5d 权 0.5, 远期 60d 权 0.2)."""
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = strat._entry_signals_multi_period(
        windows=(5, 20, 60), weights=(0.5, 0.3, 0.2),
    )
    return _run_with_entry(entry, prices, ohlc, states, cfg, strat._registry)


def run_vol_regime(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 7: 波动率体制感知 (vol > 75 百分位时才启用共振)."""
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = strat._entry_signals_vol_regime(percentile=0.75)
    return _run_with_entry(entry, prices, ohlc, states, cfg, strat._registry)


def run_cross_dim(
    strat: HuataiTimingCTAStrategy,
) -> CtaBacktestResult:
    """对照 8: 跨维度共振 (EMA + 动量 + 波动率压缩 三维度投票)."""
    cfg = strat._config
    prices, ohlc = strat._price_panel, strat._ohlc_panel
    states = strat._states
    entry = strat._entry_signals_cross_dim()
    return _run_with_entry(entry, prices, ohlc, states, cfg, strat._registry)


# ──────────────────────────────────────────────────────────────────
# 统一跑批
# ──────────────────────────────────────────────────────────────────


_METRIC_KEYS = ["AnnRet", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar",
                "WinRate", "PayoffRatio", "Years"]


def _summary_row(variant: str, result: CtaBacktestResult, signal_coverage: float) -> dict:
    m = result.metrics
    return {
        "variant": variant,
        **{k: m.get(k, np.nan) for k in _METRIC_KEYS},
        "trades": len(result.trades),
        "signal_coverage": round(signal_coverage, 4),
    }


def run_ablations(
    products: Optional[list[str]] = None,
    config: Optional[HuataiTimingConfig] = None,
    data_loader: Optional[CtaDataLoader] = None,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    include_presets: bool = True,
) -> pd.DataFrame:
    """批量运行所有对照变体 + 参数预设, 返回对比表.

    Args:
        products / config / data_loader / start / end: 透传 HuataiTimingCTAStrategy
        include_presets: 是否包含参数预设 (responsive / levered / patient / aggressive)

    Returns:
        pd.DataFrame, 行 = 变体名, 列 = 指标.
    """
    products = products or HUATAI_PRODUCTS
    base_config = config or HuataiTimingConfig()

    rows: list[dict] = []

    # 主策略 (full)
    strat_full = HuataiTimingCTAStrategy(
        products=products, config=base_config,
        data_loader=data_loader, start=start, end=end,
    )
    strat_full.load_panel()
    strat_full._estimate_states()
    full_entry = strat_full._entry_signals()
    full_result = strat_full.run()
    full_signal_cov = float((full_entry != 0).sum().sum() / full_entry.size)
    rows.append(_summary_row("full", full_result, full_signal_cov))

    # 对照 1: 仅趋势
    trend_result = run_trend_only(strat_full)
    trend_signal = _trend_only_signals(
        strat_full._price_panel, base_config.ema_fast_min,
        base_config.ema_fast_max, base_config.ema_slow, base_config.er_window,
    )
    cov = float((trend_signal != 0).sum().sum() / trend_signal.size)
    rows.append(_summary_row("trend_only", trend_result, cov))

    # 对照 2: 仅震荡
    range_result = run_range_only(strat_full)
    range_signal = _range_only_signals(
        strat_full._price_panel, base_config.rsi_window,
        base_config.rsi_oversold, base_config.rsi_overbought,
    )
    cov = float((range_signal != 0).sum().sum() / range_signal.size)
    rows.append(_summary_row("range_only", range_result, cov))

    # 对照 3: 无吊灯止损
    no_chand_result = run_no_chandelier(strat_full)
    cov = float((full_entry != 0).sum().sum() / full_entry.size)
    rows.append(_summary_row("no_chandelier", no_chand_result, cov))

    # 对照 5: 多周期共振
    multi_entry = strat_full._entry_signals_multi_period()
    multi_result = run_multi_period(strat_full)
    cov = float((multi_entry != 0).sum().sum() / multi_entry.size)
    rows.append(_summary_row("multi_period", multi_result, cov))

    # 对照 6: 加权多周期共振
    weighted_entry = strat_full._entry_signals_multi_period(
        windows=(5, 20, 60), weights=(0.5, 0.3, 0.2),
    )
    weighted_result = run_weighted_period(strat_full)
    cov = float((weighted_entry != 0).sum().sum() / weighted_entry.size)
    rows.append(_summary_row("weighted_period", weighted_result, cov))

    # 对照 7: 波动率体制感知
    vr_entry = strat_full._entry_signals_vol_regime(percentile=0.75)
    vr_result = run_vol_regime(strat_full)
    cov = float((vr_entry != 0).sum().sum() / vr_entry.size)
    rows.append(_summary_row("vol_regime", vr_result, cov))

    # 对照 8: 跨维度共振
    cd_entry = strat_full._entry_signals_cross_dim()
    cd_result = run_cross_dim(strat_full)
    cov = float((cd_entry != 0).sum().sum() / cd_entry.size)
    rows.append(_summary_row("cross_dim", cd_result, cov))

    # 参数预设 (可选)
    if include_presets:
        for preset_name, preset_cfg in HuataiTimingConfig.presets().items():
            strat_p = HuataiTimingCTAStrategy(
                products=products, config=preset_cfg,
                data_loader=data_loader, start=start, end=end,
            )
            strat_p.load_panel()
            strat_p._estimate_states()
            p_entry = strat_p._entry_signals()
            p_result = strat_p.run()
            cov = float((p_entry != 0).sum().sum() / p_entry.size)
            rows.append(_summary_row(f"preset:{preset_name}", p_result, cov))

    return pd.DataFrame(rows).set_index("variant")

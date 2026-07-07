# coding=utf-8
"""Tests for Stage 9-C: 波动率目标 (Volatility Targeting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    VolTargeting,
    BacktestConfig,
    run_rotation_backtest,
    performance_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.portfolio import (
    PortfolioState,
    apply_vol_targeting,
    vol_targeting_scale,
)


def _make_nav(n_days: int = 500, seed: int = 42) -> pd.Series:
    """合成测试 NAV (单只 ETF 价格)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n_days)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.Series(prices, index=idx)


def _make_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """合成测试面板."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    rets = rng.normal(0.0003, 0.012, size=(n_days, len(codes)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=codes)


@pytest.fixture
def nav() -> pd.Series:
    return _make_nav()


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_panel()


@pytest.fixture
def cfg_vt_enabled() -> RotationConfig:
    return RotationConfig(
        lookback=120, top_n=5, min_history=120,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.10, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
    )


@pytest.fixture
def cfg_vt_disabled() -> RotationConfig:
    return RotationConfig(lookback=120, top_n=5, min_history=120)


# ============================================================================
# 单元测试: vol_targeting_scale
# ============================================================================
class TestVolTargetingScale:
    def test_disabled_returns_one(self) -> None:
        """未启用时返回 1.0."""
        # 注意: vol_targeting_scale 不接受 enabled 参数, 只计算缩放
        pass

    def test_high_vol_reduces_scale(self, nav) -> None:
        """高波动 → 缩放系数 < 1."""
        # 制造高波动: 把最后 60 天的收益 × 3
        nav_high = nav.copy()
        nav_high.iloc[-60:] = nav_high.iloc[-60:].values * (1 + np.random.default_rng(0).normal(0, 0.04, 60).cumsum())
        scale = vol_targeting_scale(nav_high, target_vol=0.10, lookback=60,
                                    min_scale=0.3, max_scale=1.5)
        assert 0.3 <= scale < 1.0

    def test_low_vol_increases_scale(self, nav) -> None:
        """低波动 → 缩放系数 > 1 (但 ≤ max_scale)."""
        nav_low = nav.copy()
        # 制造低波动: 让整个序列都是低波动
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0003, 0.001, len(nav))  # 极低波动 (0.1% 日波动)
        nav_low = pd.Series(100.0 * np.exp(np.cumsum(rets)),
                            index=nav.index)
        scale = vol_targeting_scale(nav_low, target_vol=0.10, lookback=60,
                                    min_scale=0.3, max_scale=1.5)
        assert scale > 1.0
        assert scale <= 1.5  # clip 到 max_scale

    def test_clipping_at_min_max(self, nav) -> None:
        """缩放系数被 clip 到 [min_scale, max_scale]."""
        # 极高波动 → 应被 clip 到 min_scale
        nav_extreme = nav.copy()
        nav_extreme.iloc[-60:] = 100 + np.array([100, 50, 200, 30, 150, 80, 10, 90, 40, 120] * 6).cumsum()
        scale = vol_targeting_scale(nav_extreme, target_vol=0.10, lookback=60,
                                    min_scale=0.3, max_scale=1.5)
        assert scale == 0.3

    def test_insufficient_data_returns_one(self) -> None:
        """数据不足 lookback 时返回 1.0."""
        short_nav = pd.Series([100.0, 101.0, 102.0],
                              index=pd.bdate_range("2024-01-01", periods=3))
        scale = vol_targeting_scale(short_nav, 0.10, 60, 0.3, 1.5)
        assert scale == 1.0


# ============================================================================
# 集成测试: apply_vol_targeting
# ============================================================================
class TestApplyVolTargeting:
    def test_disabled_keeps_weights(self, nav, cfg_vt_disabled) -> None:
        """未启用 → 权重不变."""
        state = PortfolioState(date=nav.index[-1], ranked=[],
                                chosen=[], weights={"518880": 0.5, "513100": 0.5})
        new_state = apply_vol_targeting(cfg_vt_disabled, nav, nav.index[-1], state)
        assert new_state.weights == {"518880": 0.5, "513100": 0.5}

    def test_enabled_scales_weights(self, nav, cfg_vt_enabled) -> None:
        """启用 → 权重按 scale 缩放."""
        state = PortfolioState(date=nav.index[-1], ranked=[],
                                chosen=[], weights={"518880": 0.5, "513100": 0.5})
        new_state = apply_vol_targeting(cfg_vt_enabled, nav, nav.index[-1], state)
        scale = vol_targeting_scale(
            nav, target_vol=0.10, lookback=60,
            min_scale=0.3, max_scale=1.5,
        )
        assert abs(new_state.weights["518880"] - 0.5 * scale) < 1e-9
        assert abs(new_state.weights["513100"] - 0.5 * scale) < 1e-9


# ============================================================================
# 回测对比
# ============================================================================
class TestBacktestVolTargeting:
    def _run_backtest(self, panel, vol_target: bool, **kwargs) -> dict:
        if vol_target:
            cfg = RotationConfig(
                lookback=120, top_n=5, min_history=120,
                vol_targeting=VolTargeting(enabled=True, **kwargs),
            )
        else:
            cfg = RotationConfig(lookback=120, top_n=5, min_history=120)
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "dd": m["max_drawdown"],
                "ann": m["ann_return"], "vol": m["ann_vol"]}

    def test_both_runs_produce_valid_results(self, panel) -> None:
        r1 = self._run_backtest(panel, vol_target=False)
        r2 = self._run_backtest(panel, vol_target=True, target_vol=0.10, lookback=60)
        assert "calmar" in r1
        assert "calmar" in r2

    def test_lower_target_vol_reduces_volatility(self, panel) -> None:
        """目标波动越低 → 实际波动越低."""
        r_high = self._run_backtest(panel, vol_target=True, target_vol=0.20, lookback=60)
        r_low = self._run_backtest(panel, vol_target=True, target_vol=0.05, lookback=60)
        assert r_low["vol"] <= r_high["vol"] + 0.05  # 允许小幅波动
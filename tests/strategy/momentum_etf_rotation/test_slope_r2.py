# coding=utf-8
"""Tests for Stage 12A: 斜率 × R² 动量信号 (来自猫哥 5年10倍策略)."""
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
from QuantNodes.strategy.momentum_etf_rotation.core.momentum import (
    slope_r2_score,
    hybrid_momentum_score,
    compute_momentum_score,
    rank_by_momentum,
    rank_pctl,
)


def _make_uptrend_prices(n_days: int = 200, start: float = 100.0,
                         daily_return: float = 0.001, noise: float = 0.005,
                         seed: int = 42) -> pd.DataFrame:
    """合成完美上升趋势 (无噪声)."""
    rng = np.random.default_rng(seed)
    n_codes = 5
    rets = rng.normal(daily_return, noise, (n_days, n_codes))
    # 让所有 code 有相同的 beta
    rets = np.tile(np.linspace(daily_return * 0.8, daily_return * 1.2, n_codes), (n_days, 1)) + \
           rng.normal(0, noise, (n_days, n_codes))
    prices = start * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_codes)])


def _make_downtrend_prices(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """合成下降趋势."""
    rng = np.random.default_rng(seed)
    n_codes = 5
    rets = rng.normal(-0.002, 0.005, (n_days, n_codes))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_codes)])


def _make_sideways_prices(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """合成震荡 (无趋势, 高噪声)."""
    rng = np.random.default_rng(seed)
    n_codes = 5
    rets = rng.normal(0, 0.02, (n_days, n_codes))  # 大噪声
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_codes)])


def _make_realistic_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """合成测试面板 (类似 conftest.synthetic_nav)."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)[:20]
    rets = rng.normal(0.0003, 0.012, (n_days, len(codes)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=pd.bdate_range("2020-01-01", periods=n_days), columns=codes)


# ============================================================================
# 单元测试: slope_r2_score
# ============================================================================
class TestSlopeR2Score:
    def test_uptrend_positive(self):
        """上升趋势 → 多数 score 为正."""
        prices = _make_uptrend_prices()
        score = slope_r2_score(prices, lookback=150)
        # 多数 (≥ 80%) 应为正 (允许个别因噪声略负)
        n_pos = (score > 0).sum()
        assert n_pos >= 0.8 * len(score), f"上升趋势应有 ≥ 80% 正 score, 实际 {n_pos}/{len(score)}"

    def test_uptrend_mean_positive(self):
        """上升趋势 → 平均 score > 下降趋势."""
        up = _make_uptrend_prices(seed=42)
        down = _make_downtrend_prices(seed=42)
        up_score = slope_r2_score(up, lookback=150)
        down_score = slope_r2_score(down, lookback=150)
        assert up_score.mean() > down_score.mean(), (
            f"上升 {up_score.mean():.2f} 应 > 下降 {down_score.mean():.2f}"
        )

    def test_downtrend_negative(self):
        """下降趋势 → 斜率为负 → score < 0."""
        prices = _make_downtrend_prices()
        score = slope_r2_score(prices, lookback=150)
        assert all(score < 0), f"下降趋势应有负 score, 实际 {score.values}"

    def test_sideways_low_magnitude(self):
        """震荡市 → 斜率近 0 → |score| 小."""
        prices = _make_sideways_prices()
        score = slope_r2_score(prices, lookback=150)
        # 震荡市 score 应明显小于强趋势市场 (<50)
        assert score.abs().max() < 50.0, f"震荡市 score 应小, 实际 max={score.abs().max()}"

    def test_returns_series(self):
        """返回 pd.Series, index=code."""
        prices = _make_uptrend_prices()
        score = slope_r2_score(prices, lookback=150)
        assert isinstance(score, pd.Series)
        assert set(score.index) >= set(prices.columns[:3])

    def test_insufficient_data(self):
        """数据不足 20 天 → 返回空或全 0."""
        prices = _make_uptrend_prices(n_days=10)
        score = slope_r2_score(prices, lookback=100)
        # 应全部为 0 或空 Series
        assert len(score) == 0 or (score == 0).all() or score.isna().all()


# ============================================================================
# 单元测试: hybrid_momentum_score
# ============================================================================
class TestHybridMomentumScore:
    def test_hybrid_combines_both(self):
        """hybrid 应该是 price 和 slope_r2 的混合."""
        prices = _make_uptrend_prices()
        hybrid = hybrid_momentum_score(prices, lookback=100, fused_weight=0.5)
        # hybrid 应是连续的实数, 不是 NaN
        assert hybrid.notna().all()

    def test_hybrid_different_from_pure(self):
        """hybrid 不等于 pure price 或 pure slope_r2 (即使相关)."""
        prices = _make_uptrend_prices()
        price = rank_by_momentum(prices, lookback=100)
        slope = slope_r2_score(prices, lookback=100)
        hybrid = hybrid_momentum_score(prices, lookback=100, fused_weight=0.5)
        # hybrid 不应完全等于任何一个
        assert not hybrid.equals(price), "hybrid 不应等于 pure price"
        assert not hybrid.equals(slope), "hybrid 不应等于 pure slope_r2"

    def test_fused_weight_changes_result(self):
        """fused_weight 变化应改变结果."""
        prices = _make_uptrend_prices()
        # 不同的 fused_weight 应产生不同结果
        results = []
        for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            r = hybrid_momentum_score(prices, lookback=100, fused_weight=w)
            results.append(r)
        # 至少有两对不同
        n_unique = len(set(tuple(r.values) for r in results))
        assert n_unique >= 3, f"fused_weight 应产生多种结果, 实际唯一 {n_unique}"


# ============================================================================
# 单元测试: compute_momentum_score (统一接口)
# ============================================================================
class TestComputeMomentumScore:
    def test_price_type_matches_rank(self):
        """momentum_type='price' 应与 rank_by_momentum 一致."""
        prices = _make_uptrend_prices()
        result = compute_momentum_score(prices, lookback=100, momentum_type="price")
        expected = rank_by_momentum(prices, lookback=100)
        assert result.equals(expected)

    def test_slope_r2_type_matches(self):
        """momentum_type='slope_r2' 应与 slope_r2_score 一致."""
        prices = _make_uptrend_prices()
        result = compute_momentum_score(prices, lookback=100, momentum_type="slope_r2")
        expected = slope_r2_score(prices, lookback=100)
        assert result.equals(expected)

    def test_hybrid_type_matches(self):
        """momentum_type='hybrid' 应与 hybrid_momentum_score 一致."""
        prices = _make_uptrend_prices()
        result = compute_momentum_score(prices, lookback=100, momentum_type="hybrid", fused_weight=0.5)
        expected = hybrid_momentum_score(prices, lookback=100, fused_weight=0.5)
        assert result.equals(expected)


# ============================================================================
# 单元测试: rank_pctl 支持 momentum_type
# ============================================================================
class TestRankPctlWithMomentumType:
    def test_price_default(self):
        prices = _make_uptrend_prices()
        pctl = rank_pctl(prices, lookback=100, momentum_type="price")
        assert isinstance(pctl, pd.Series)
        assert all((pctl >= 0) & (pctl <= 1))

    def test_slope_r2(self):
        prices = _make_uptrend_prices()
        pctl = rank_pctl(prices, lookback=100, momentum_type="slope_r2")
        assert isinstance(pctl, pd.Series)


# ============================================================================
# 集成测试: RotationConfig.momentum_type
# ============================================================================
class TestRotationConfigMomentumType:
    def test_default_is_price(self):
        """默认 momentum_type 应为 'price' (向后兼容)."""
        cfg = RotationConfig(lookback=90, top_n=10)
        assert cfg.momentum_type == "price"

    def test_all_three_types_accepted(self):
        """3 种方式都应被接受."""
        for mt in ["price", "slope_r2", "hybrid"]:
            cfg = RotationConfig(lookback=90, top_n=10, momentum_type=mt)
            assert cfg.momentum_type == mt


# ============================================================================
# 回测对比
# ============================================================================
class TestBacktestMomentumType:
    def _run(self, momentum_type: str, fused_weight: float = 0.5) -> dict:
        cfg = RotationConfig(
            lookback=120, top_n=5,
            momentum_type=momentum_type,
            momentum_fused_weight=fused_weight,
        )
        panel = _make_realistic_panel()
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "dd": m["max_drawdown"], "ann": m["ann_return"]}

    def test_all_three_run(self):
        """3 种方式都能跑通."""
        for mt in ["price", "slope_r2", "hybrid"]:
            r = self._run(mt)
            assert "calmar" in r
            assert r["calmar"] > 0 or r["dd"] < 0

    def test_baseline_price(self):
        """baseline (price) 跑通."""
        r = self._run("price")
        assert r["calmar"] > 0

    def test_slope_r2_runs(self):
        """slope_r2 跑通."""
        r = self._run("slope_r2")
        assert r["calmar"] > 0

    def test_hybrid_runs(self):
        """hybrid 跑通."""
        r = self._run("hybrid")
        assert r["calmar"] > 0

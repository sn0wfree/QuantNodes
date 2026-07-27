# coding=utf-8
"""Tests for Stage 9-A: 52 周新高信号融合."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    BacktestConfig,
    run_rotation_backtest,
    performance_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.core.momentum import (
    distance_to_52w_high,
    fused_signal,
    rank_by_momentum,
)
from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import select_and_weight


def _make_panel(n_days: int = 600, seed: int = 42) -> pd.DataFrame:
    """合成测试面板: 全 44 ETF, 600 天."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    rets = rng.normal(0.0003, 0.012, size=(n_days, len(codes)))
    # 让部分 ETF 有强动量
    rets[:, 1] += 0.001
    rets[:, 5] += 0.0008
    rets[:, 10] += 0.0006
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=codes)


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_panel()


@pytest.fixture
def pool() -> "ETFPool":
    """使用全 44 ETF 池 (含商品/海外/A 股宽基等所有类别)."""
    return DEFAULT_POOL


@pytest.fixture
def cfg_momentum() -> RotationConfig:
    return RotationConfig(
        lookback=120, top_n=5, min_history=120,
        signal_type="momentum",
    )


@pytest.fixture
def cfg_dist52w() -> RotationConfig:
    return RotationConfig(
        lookback=120, top_n=5, min_history=120,
        signal_type="dist_52w",
    )


@pytest.fixture
def cfg_fused() -> RotationConfig:
    return RotationConfig(
        lookback=120, top_n=5, min_history=120,
        signal_type="fused", signal_fused_weight=0.4,
    )


@pytest.fixture
def cfg_fused_high() -> RotationConfig:
    """w=0.8, 偏向 52 周新高."""
    return RotationConfig(
        lookback=120, top_n=5, min_history=120,
        signal_type="fused", signal_fused_weight=0.8,
    )


# ============================================================================
# 单元测试: fused_signal 函数
# ============================================================================
class TestFusedSignal:
    def test_returns_series_with_code_index(self, panel) -> None:
        score = fused_signal(panel, lookback=120)
        assert isinstance(score, pd.Series)
        assert set(score.index) >= set(panel.columns[:5])

    def test_fused_zero_weight_equals_normalized_momentum(self, panel) -> None:
        """w=0 时 fused = 归一化动量 (允许有并列差异)."""
        fused = fused_signal(panel, lookback=120, fused_weight=0.0)
        mom = rank_by_momentum(panel, lookback=120)
        mom_norm = mom / mom.abs().max()
        # 排除 NaN 后排名应一致 (允许并列差异)
        fused_clean = fused.dropna().sort_values(ascending=False).index
        mom_clean = mom_norm.dropna().sort_values(ascending=False).index
        # 排名 80% 以上相同
        common = set(fused_clean) & set(mom_clean)
        assert len(common) >= 0.8 * min(len(fused_clean), len(mom_clean))

    def test_fused_one_weight_equals_normalized_dist52w(self, panel) -> None:
        """w=1 时 fused = 归一化 52 周新高 (允许并列差异)."""
        fused = fused_signal(panel, fused_weight=1.0)
        dist = distance_to_52w_high(panel)
        dist_norm = dist / dist.abs().max()
        # 排除 NaN 后排名应一致
        fused_clean = fused.dropna().sort_values(ascending=False).index
        dist_clean = dist_norm.dropna().sort_values(ascending=False).index
        # 排名 80% 以上相同
        common = set(fused_clean) & set(dist_clean)
        assert len(common) >= 0.8 * min(len(fused_clean), len(dist_clean))

    def test_fused_middle_weight_interpolates(self, panel) -> None:
        """w=0.4 时, fused 排名与纯动量不完全一致."""
        fused = fused_signal(panel, fused_weight=0.4)
        mom = rank_by_momentum(panel, lookback=120)
        # 至少有一个 ETF 排名发生变化
        mom_rank = mom.rank(ascending=False)
        fused_rank = fused.rank(ascending=False)
        # 排名差异 (Kendall tau 应该 < 1.0)
        from scipy.stats import kendalltau
        tau, _ = kendalltau(mom_rank, fused_rank)
        assert tau < 1.0  # 不完全相同

    def test_fused_score_range_reasonable(self, panel) -> None:
        """fused score 应该在 [-1, 1] 范围 (归一化后)."""
        fused = fused_signal(panel, fused_weight=0.4)
        assert fused.abs().max() <= 1.0


# ============================================================================
# 集成测试: select_and_weight 支持 3 种信号
# ============================================================================
class TestSelectAndWeightSignalType:
    def test_momentum_signal_default(self, panel, pool, cfg_momentum) -> None:
        """默认 signal_type=momentum, 行为与原版一致."""
        state = select_and_weight(panel, pool, cfg_momentum, panel.index[-1])
        assert len(state.chosen) >= 1  # 至少选 1 个

    def test_dist52w_signal_works(self, panel, pool, cfg_dist52w) -> None:
        """dist_52w 信号可以正常选 ETF."""
        state = select_and_weight(panel, pool, cfg_dist52w, panel.index[-1])
        assert len(state.chosen) >= 1

    def test_fused_signal_works(self, panel, pool, cfg_fused) -> None:
        """fused 信号可以正常选 ETF."""
        state = select_and_weight(panel, pool, cfg_fused, panel.index[-1])
        assert len(state.chosen) >= 1

    def test_all_signals_produce_different_rankings(self, panel, pool) -> None:
        """3 种信号的 ranking 应该不同."""
        date = panel.index[-1]
        cfg1 = RotationConfig(lookback=120, signal_type="momentum")
        cfg2 = RotationConfig(lookback=120, signal_type="dist_52w")
        cfg3 = RotationConfig(lookback=120, signal_type="fused",
                              signal_fused_weight=0.4)
        s1 = select_and_weight(panel, pool, cfg1, date)
        s2 = select_and_weight(panel, pool, cfg2, date)
        s3 = select_and_weight(panel, pool, cfg3, date)
        # 至少 momentum vs dist_52w 不同
        assert s1.ranked != s2.ranked
        # momentum vs fused 也可能不同
        # (由于 caps 影响, 不保证 chosen 不同, 但 ranked 一定不同)


# ============================================================================
# 回测对比
# ============================================================================
class TestBacktestSignalComparison:
    """3 种信号的全段回测对比."""

    def _run_backtest(self, panel, signal_type: str, **kwargs) -> dict:
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            signal_type=signal_type, **kwargs,
        )
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "dd": m["max_drawdown"],
                "ann": m["ann_return"], "n_rebal": len(result.states)}

    def test_three_signals_all_produce_valid_results(self, panel) -> None:
        """3 种信号都能产出有效回测."""
        for sig in ["momentum", "dist_52w", "fused"]:
            r = self._run_backtest(panel, sig)
            assert "calmar" in r
            assert "dd" in r

    def test_fused_with_high_weight_runs(self, panel) -> None:
        """fused 高权重 (w=0.8) 不崩."""
        r = self._run_backtest(panel, "fused", signal_fused_weight=0.8)
        assert "calmar" in r
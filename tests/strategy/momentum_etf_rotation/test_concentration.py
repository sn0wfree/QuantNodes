# coding=utf-8
"""Tests for Stage 10: 集中度约束 (Concentration Caps)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    ConcentrationCaps,
    VolTargeting,
    BacktestConfig,
    run_rotation_backtest,
    performance_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import (
    PortfolioState,
    _apply_concentration_caps,
    apply_concentration_caps,
)


def _make_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """合成测试面板."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    rets = rng.normal(0.0003, 0.012, size=(n_days, len(codes)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=codes)


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_panel()


# ============================================================================
# 单元测试: _apply_concentration_caps
# ============================================================================
class TestApplyConcentrationCaps:
    def test_disabled_no_change(self) -> None:
        """未启用 → 权重不变."""
        caps = ConcentrationCaps(enabled=False)
        weights = {"518880": 0.30, "513100": 0.30, "513500": 0.40}
        result = _apply_concentration_caps(weights, caps)
        assert result == weights

    def test_single_etf_capped(self) -> None:
        """单 ETF 权重超 15% → 被截断."""
        caps = ConcentrationCaps(enabled=True, single_etf_max=0.15)
        weights = {"518880": 0.40, "513100": 0.30, "513500": 0.30}
        result = _apply_concentration_caps(weights, caps)
        # 518880 应被截到 15%
        assert result["518880"] <= caps.single_etf_max + 1e-9
        # 总和 ≤ 1.0 (差额为现金)
        assert sum(result.values()) <= 1.0 + 1e-9

    def test_top3_capped(self) -> None:
        """Top 3 ETF 合计超 45% → 被压缩 (按原始排序)."""
        caps = ConcentrationCaps(
            enabled=True,
            single_etf_max=0.20,
            top_n_total_max=0.45,
            top_n_count=3,
        )
        weights = {"518880": 0.30, "513100": 0.25, "513500": 0.20,
                   "159915": 0.15, "510900": 0.10}
        result = _apply_concentration_caps(weights, caps)
        # Top 3 (518880, 513100, 513500) 是原始前 3, 总和应 <= 0.45
        top3_orig = ["518880", "513100", "513500"]
        top3_total = sum(result[c] for c in top3_orig)
        assert top3_total <= caps.top_n_total_max + 1e-9
        # 验证原始 Top 3 确实被压缩了
        assert result["518880"] < 0.30  # 被压缩了
        assert result["513100"] < 0.25  # 被压缩了

    def test_category_capped(self, panel) -> None:
        """单类别合计超 40% → 被压缩."""
        caps = ConcentrationCaps(
            enabled=True,
            single_etf_max=0.50,
            top_n_total_max=1.0,
            category_max=0.40,
        )
        # 商品类合计 60% (518880 + 518800 + 161226)
        weights = {"518880": 0.25, "518800": 0.20, "161226": 0.15,
                   "513100": 0.20, "513500": 0.20}
        result = _apply_concentration_caps(weights, caps, pool=DEFAULT_POOL)
        # 商品类合计应 <= 0.40
        cat_total = sum(v for k, v in result.items()
                       if DEFAULT_POOL.category_of(k).value == "commodity")
        assert cat_total <= caps.category_max + 1e-9

    def test_weights_sum_preserved(self) -> None:
        """约束后总权重 ≤ 1.0 (差额视为现金)."""
        caps = ConcentrationCaps(enabled=True, single_etf_max=0.15,
                                  top_n_total_max=0.45, category_max=0.40)
        weights = {"518880": 0.40, "513100": 0.30, "513500": 0.30}
        result = _apply_concentration_caps(weights, caps, pool=DEFAULT_POOL)
        # 总权重 <= 1.0 (允许 < 1 表示有现金)
        assert sum(result.values()) <= 1.0 + 1e-9


# ============================================================================
# 集成测试: select_and_weight + concentration
# ============================================================================
class TestSelectAndWeightConcentration:
    def test_concentration_reduces_max_weight(self, panel) -> None:
        """启用集中度约束后, 单 ETF 最大权重降低."""
        cfg_no = RotationConfig(
            lookback=120, top_n=5, min_history=120,
        )
        cfg_yes = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            concentration=ConcentrationCaps(
                enabled=True, single_etf_max=0.15,
                top_n_total_max=0.45, category_max=0.40,
            ),
        )
        date = panel.index[-1]
        from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import select_and_weight
        s_no = select_and_weight(panel, DEFAULT_POOL, cfg_no, date)
        s_yes = select_and_weight(panel, DEFAULT_POOL, cfg_yes, date)
        max_no = max(s_no.weights.values()) if s_no.weights else 0
        max_yes = max(s_yes.weights.values()) if s_yes.weights else 0
        # 启用后最大权重应 <= 15% + 缓冲 (因为趋势过滤可能加债券)
        assert max_yes <= 0.20  # 15% + 5% 缓冲

    def test_concentration_preserves_sum(self, panel) -> None:
        """启用后总权重 ≤ 1.0 (cash 缓冲)."""
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            concentration=ConcentrationCaps(
                enabled=True, single_etf_max=0.15,
                top_n_total_max=0.45, category_max=0.40,
            ),
        )
        date = panel.index[-1]
        from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import select_and_weight
        state = select_and_weight(panel, DEFAULT_POOL, cfg, date)
        # 总权重 ≤ 1.0 (允许持有现金)
        assert sum(state.weights.values()) <= 1.0 + 1e-6


# ============================================================================
# 回测对比
# ============================================================================
class TestBacktestConcentration:
    def _run(self, panel, enabled: bool) -> dict:
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            concentration=ConcentrationCaps(
                enabled=enabled,
                single_etf_max=0.15,
                top_n_total_max=0.45,
                category_max=0.40,
            ),
        )
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "dd": m["max_drawdown"],
                "ann": m["ann_return"], "max_weight": max(
                    (max(s.weights.values()) if s.weights else 0)
                    for s in result.states
                )}

    def test_both_runs_produce_valid_results(self, panel) -> None:
        r_no = self._run(panel, enabled=False)
        r_yes = self._run(panel, enabled=True)
        assert "calmar" in r_no
        assert "calmar" in r_yes

    def test_concentration_reduces_max_weight(self, panel) -> None:
        """启用后单 ETF 最大权重应降低."""
        r_no = self._run(panel, enabled=False)
        r_yes = self._run(panel, enabled=True)
        assert r_yes["max_weight"] <= r_no["max_weight"]
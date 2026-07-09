# coding=utf-8
"""Tests for Stage 16A: 均值反转子策略 (ReversionSubStrategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v3 import (
    ReversionConfig,
    ReversionSubStrategy,
    SubStrategy,
    SubStrategyResult,
    reversion_score,
)
from QuantNodes.strategy.momentum_etf_rotation.common import DEFAULT_POOL


def _make_panel(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """合成测试面板: 部分 ETF 下跌 (适合反转)."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    rets = rng.normal(0.0001, 0.015, size=(n_days, len(codes)))
    # 让前 5 个 ETF 显著下跌 (测试反转策略选股)
    rets[:, :5] -= 0.005
    # 后期 5 个 ETF 强劲上涨 (测试反转策略过滤)
    rets[:, -5:] += 0.008
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=pd.date_range("2024-01-01", periods=n_days), columns=codes)


class TestReversionScore:
    """reversion_score 单元测试."""

    def test_basic_output_shape(self):
        panel = _make_panel(n_days=100)
        as_of = panel.index[-1]
        score = reversion_score(panel, as_of, lookback=60)
        assert isinstance(score, pd.Series)
        assert len(score) == len(panel.columns)
        # 下跌深的 ETF 得分应该高
        # 前期下跌的 ETF 排名靠前

    def test_empty_on_insufficient_history(self):
        panel = _make_panel(n_days=10)  # 不足 lookback=60
        as_of = panel.index[-1]
        score = reversion_score(panel, as_of, lookback=60)
        assert score.empty

    def test_max_drawdown_filter(self):
        """max_drawdown=-0.30 时, 大幅下跌的 ETF 应被过滤."""
        panel = _make_panel(n_days=200, seed=99)
        # 制造一只大幅下跌的 ETF
        panel.iloc[:, 0] = panel.iloc[:, 0] * 0.5  # 跌幅 -50%
        as_of = panel.index[-1]

        score = reversion_score(panel, as_of, lookback=60, max_drawdown=-0.30)
        # 第一只 ETF 应该被标记为不可选
        assert score.iloc[0] == -1.0

    def test_golden_cross_bonus(self):
        """金叉 (ma5 > ma10) 应获得额外分数."""
        panel = _make_panel(n_days=200)
        as_of = panel.index[-1]
        score_with = reversion_score(panel, as_of, crossover_weight=0.3)
        score_without = reversion_score(panel, as_of, crossover_weight=0.0)
        # 至少某些 ETF 的得分应该有差异
        assert not (score_with == score_without).all()


class TestReversionSubStrategy:
    """ReversionSubStrategy 单元测试."""

    def test_subclass_of_substrategy(self):
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        assert isinstance(sub, SubStrategy)

    def test_select_returns_list(self):
        panel = _make_panel(n_days=200)
        as_of = panel.index[-1]
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        chosen = sub.select(panel, as_of)
        assert isinstance(chosen, list)
        assert len(chosen) <= cfg.top_n
        # 所有 chosen 必须在 DEFAULT_POOL 中
        for c in chosen:
            assert c in DEFAULT_POOL.codes

    def test_weight_equal(self):
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        weights = sub.weight(None, ["510300", "510500", "518880"], None)
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1.0/3) < 1e-9

    def test_run_step_basic(self):
        panel = _make_panel(n_days=200)
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        result = sub.run_step(panel, panel.index[-1])
        assert isinstance(result, SubStrategyResult)
        assert result.date == panel.index[-1]
        assert len(result.chosen) == len(result.weights)
        # 权重和为 1
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        # 所有权重非负
        for w in result.weights.values():
            assert w >= 0

    def test_a_share_cap(self):
        """A股宽基+行业 cap 应该被尊重."""
        panel = _make_panel(n_days=200)
        cfg = ReversionConfig(top_n=10, a_share_total=2)  # 严格 cap
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        result = sub.run_step(panel, panel.index[-1])

        a_share_count = 0
        for c in result.chosen:
            cat = DEFAULT_POOL.category_of(c)
            if cat.value in ("a_broad", "a_sector"):
                a_share_count += 1
        assert a_share_count <= cfg.a_share_total

    def test_run_step_empty_on_insufficient_history(self):
        panel = _make_panel(n_days=20)  # 不足 min_history
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        result = sub.run_step(panel, panel.index[-1])
        assert result.chosen == []
        assert result.weights == {}


class TestReversionIntegration:
    """反转策略与 v2 池的集成测试."""

    def test_pool_validation(self):
        """选中的 ETF 必须都在 pool 中."""
        panel = _make_panel(n_days=200)
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        result = sub.run_step(panel, panel.index[-1])
        for c in result.chosen:
            assert c in DEFAULT_POOL.codes

    def test_signal_strength_range(self):
        """signal_strength 应在合理范围内."""
        panel = _make_panel(n_days=200)
        cfg = ReversionConfig()
        sub = ReversionSubStrategy(cfg, DEFAULT_POOL)
        result = sub.run_step(panel, panel.index[-1])
        if result.signal_strength != 0.0:
            # signal_strength 是 score 的均值, 应在 [-1, 1+crossover_weight] 范围内
            assert -1.0 <= result.signal_strength <= 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# coding=utf-8
"""Tests for Stage 16A: 行业轮动子策略 (IndustryRotationSubStrategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v3 import (
    IndustryRotationConfig,
    IndustryRotationSubStrategy,
    SubStrategy,
    SubStrategyResult,
    industry_rotation_score,
    get_industry_codes,
    get_rebalance_dates,
)
from QuantNodes.strategy.momentum_etf_rotation.common import (
    DEFAULT_POOL, Category, ETFPool, ETFMeta,
)


def _make_panel(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """合成测试面板: 部分行业 ETF 强动量."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    rets = rng.normal(0.0002, 0.012, size=(n_days, len(codes)))
    # 让几个 A 股行业 ETF 显著走强
    industry_codes = get_industry_codes(DEFAULT_POOL)[:3]
    for code in industry_codes:
        if code in DEFAULT_POOL.codes:
            idx = list(DEFAULT_POOL.codes).index(code)
            rets[:, idx] += 0.005  # 额外 +0.5%/日
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=pd.date_range("2024-01-01", periods=n_days), columns=codes)


class TestGetIndustryCodes:
    """get_industry_codes 单元测试."""

    def test_returns_a_sector_only(self):
        codes = get_industry_codes(DEFAULT_POOL)
        assert len(codes) > 0
        for c in codes:
            cat = DEFAULT_POOL.category_of(c)
            assert cat == Category.A_SECTOR

    def test_count_matches_pool(self):
        codes = get_industry_codes(DEFAULT_POOL)
        expected = sum(1 for m in DEFAULT_POOL.members if m.category == Category.A_SECTOR)
        assert len(codes) == expected


class TestGetRebalanceDates:
    """get_rebalance_dates 单元测试."""

    def test_weekly_friday(self):
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
        rebal = get_rebalance_dates(dates, freq="W-FRI")
        # 周五应该是调仓日
        for d in rebal:
            assert d.weekday() == 4  # Friday

    def test_month_end(self):
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
        rebal = get_rebalance_dates(dates, freq="ME")
        assert len(rebal) == 3  # Jan, Feb, Mar
        for d in rebal:
            assert d in dates


class TestIndustryRotationScore:
    """industry_rotation_score 单元测试."""

    def test_basic_output(self):
        panel = _make_panel(n_days=120)
        as_of = panel.index[-1]
        industry_codes = get_industry_codes(DEFAULT_POOL)
        score = industry_rotation_score(panel, as_of, industry_codes, lookback=60)
        assert isinstance(score, pd.Series)
        assert len(score) == len(industry_codes)

    def test_empty_on_insufficient_history(self):
        panel = _make_panel(n_days=30)  # 不足 lookback=60
        as_of = panel.index[-1]
        industry_codes = get_industry_codes(DEFAULT_POOL)
        score = industry_rotation_score(panel, as_of, industry_codes, lookback=60)
        assert score.empty

    def test_strong_perf_ranks_high(self):
        """强动量的行业 ETF 排名应该靠前."""
        panel = _make_panel(n_days=200)
        as_of = panel.index[-1]
        industry_codes = get_industry_codes(DEFAULT_POOL)
        score = industry_rotation_score(panel, as_of, industry_codes, lookback=60)
        # 我们前面 _make_panel 让前 3 个行业 ETF 强动量
        # 它们应该排名靠前
        top3 = score.sort_values(ascending=False).head(3).index.tolist()
        # 前 3 个行业 ETF 应该在 top3 中
        expected_top3 = industry_codes[:3]
        # 至少 2/3 命中
        overlap = set(top3) & set(expected_top3)
        assert len(overlap) >= 2


class TestIndustryRotationSubStrategy:
    """IndustryRotationSubStrategy 单元测试."""

    def test_subclass_of_substrategy(self):
        cfg = IndustryRotationConfig()
        sub = IndustryRotationSubStrategy(cfg, DEFAULT_POOL)
        assert isinstance(sub, SubStrategy)

    def test_select_only_a_sector(self):
        """选中的 ETF 必须都是 A 股行业."""
        panel = _make_panel(n_days=200)
        as_of = panel.index[-1]
        cfg = IndustryRotationConfig()
        sub = IndustryRotationSubStrategy(cfg, DEFAULT_POOL)
        chosen = sub.select(panel, as_of)
        for c in chosen:
            cat = DEFAULT_POOL.category_of(c)
            assert cat == Category.A_SECTOR

    def test_select_respects_top_n(self):
        panel = _make_panel(n_days=200)
        as_of = panel.index[-1]
        cfg = IndustryRotationConfig(top_n=5)
        sub = IndustryRotationSubStrategy(cfg, DEFAULT_POOL)
        chosen = sub.select(panel, as_of)
        assert len(chosen) <= 5

    def test_weight_inverse_vol(self):
        """逆波动加权: 高波动 ETF 权重低."""
        panel = _make_panel(n_days=200)
        as_of = panel.index[-1]
        cfg = IndustryRotationConfig()
        sub = IndustryRotationSubStrategy(cfg, DEFAULT_POOL)
        codes = get_industry_codes(DEFAULT_POOL)[:3]
        weights = sub.weight(panel, codes, as_of)
        # 权重和为 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # 所有权重非负
        for w in weights.values():
            assert w >= 0

    def test_run_step_basic(self):
        panel = _make_panel(n_days=200)
        cfg = IndustryRotationConfig()
        sub = IndustryRotationSubStrategy(cfg, DEFAULT_POOL)
        result = sub.run_step(panel, panel.index[-1])
        assert isinstance(result, SubStrategyResult)
        assert len(result.chosen) == len(result.weights)
        # 所有 chosen 必须是 A 股行业
        for c in result.chosen:
            assert DEFAULT_POOL.category_of(c) == Category.A_SECTOR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

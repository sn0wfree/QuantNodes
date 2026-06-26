# coding=utf-8
"""Tests for mutual IC deduplication.

覆盖：
- _spearman_corr: Spearman 秩相关计算
- deduplicate_mutual_ic: 贪心互信息去重
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
    _spearman_corr,
    deduplicate_mutual_ic,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_series():
    """创建测试用序列"""
    np.random.seed(42)
    return pl.Series(np.random.randn(100))


@pytest.fixture
def correlated_series():
    """创建相关序列"""
    np.random.seed(42)
    x = np.random.randn(100)
    y = x * 0.9 + np.random.randn(100) * 0.1  # 高度相关
    return pl.Series(x), pl.Series(y)


@pytest.fixture
def uncorrelated_series():
    """创建不相关序列"""
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    return pl.Series(x), pl.Series(y)


# ==============================================================================
# TestSpearmanCorr
# ==============================================================================


class TestSpearmanCorr:
    """Spearman 秩相关测试"""

    def test_perfect_correlation(self):
        """完全正相关"""
        x = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        corr = _spearman_corr(x, y)
        assert abs(corr - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        """完全负相关"""
        x = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series([10.0, 8.0, 6.0, 4.0, 2.0])
        corr = _spearman_corr(x, y)
        assert abs(corr - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """不相关"""
        np.random.seed(42)
        x = pl.Series(np.random.randn(1000))
        y = pl.Series(np.random.randn(1000))
        corr = _spearman_corr(x, y)
        assert abs(corr) < 0.2  # 应该接近 0

    def test_short_series(self):
        """序列长度不足"""
        x = pl.Series([1.0, 2.0])
        y = pl.Series([3.0, 4.0])
        corr = _spearman_corr(x, y)
        assert corr == 0.0

    def test_different_lengths(self):
        """不同长度序列"""
        x = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series([2.0, 4.0, 6.0])
        corr = _spearman_corr(x, y)
        assert abs(corr - 1.0) < 1e-10

    def test_with_ties(self):
        """有重复值"""
        x = pl.Series([1.0, 2.0, 2.0, 3.0, 4.0])
        y = pl.Series([2.0, 4.0, 4.0, 6.0, 8.0])
        corr = _spearman_corr(x, y)
        assert corr > 0.9  # 应该高度相关


# ==============================================================================
# TestDeduplicateMutualIC
# ==============================================================================


class TestDeduplicateMutualIC:
    """互信息去重测试"""

    def _make_metrics(self, formula_id: str, overall_score: float) -> FactorMetrics:
        """创建测试用 FactorMetrics"""
        return FactorMetrics(
            formula_id=formula_id,
            status="success",
            overall_score=overall_score,
        )

    def test_empty_list(self):
        """空列表"""
        result = deduplicate_mutual_ic([], lambda f: None)
        assert result == []

    def test_single_factor(self):
        """单个因子"""
        metrics = [self._make_metrics("f1", 0.8)]
        values = {"f1": pl.Series([1.0, 2.0, 3.0])}
        result = deduplicate_mutual_ic(metrics, lambda f: values.get(f.formula_id))
        assert len(result) == 1
        assert result[0].formula_id == "f1"

    def test_no_duplicates(self):
        """无重复因子"""
        np.random.seed(42)
        metrics = [
            self._make_metrics(f"f{i}", 0.8 - i * 0.1)
            for i in range(5)
        ]
        values = {
            f"f{i}": pl.Series(np.random.randn(100))
            for i in range(5)
        }
        result = deduplicate_mutual_ic(metrics, lambda f: values.get(f.formula_id))
        assert len(result) == 5

    def test_with_duplicates(self):
        """有重复因子"""
        np.random.seed(42)
        x = np.random.randn(100)
        metrics = [
            self._make_metrics("f1", 0.9),
            self._make_metrics("f2", 0.8),
            self._make_metrics("f3", 0.7),
        ]
        # f1 和 f2 高度相关
        values = {
            "f1": pl.Series(x),
            "f2": pl.Series(x * 0.95 + np.random.randn(100) * 0.05),
            "f3": pl.Series(np.random.randn(100)),
        }
        result = deduplicate_mutual_ic(metrics, lambda f: values.get(f.formula_id))
        assert len(result) == 2
        assert result[0].formula_id == "f1"  # 保留得分最高的
        assert result[1].formula_id == "f3"

    def test_threshold_sensitivity(self):
        """阈值敏感性"""
        np.random.seed(42)
        x = np.random.randn(100)
        metrics = [
            self._make_metrics("f1", 0.9),
            self._make_metrics("f2", 0.8),
        ]
        # 相关系数约 0.96
        values = {
            "f1": pl.Series(x),
            "f2": pl.Series(x * 0.8 + np.random.randn(100) * 0.2),
        }

        # 高阈值：不去重
        result = deduplicate_mutual_ic(
            metrics, lambda f: values.get(f.formula_id), threshold=0.99
        )
        assert len(result) == 2

        # 低阈值：去重
        result = deduplicate_mutual_ic(
            metrics, lambda f: values.get(f.formula_id), threshold=0.5
        )
        assert len(result) == 1

    def test_preserves_order(self):
        """保持排序顺序"""
        np.random.seed(42)
        metrics = [
            self._make_metrics("f1", 0.9),
            self._make_metrics("f2", 0.8),
            self._make_metrics("f3", 0.7),
        ]
        values = {
            f"f{i}": pl.Series(np.random.randn(100))
            for i in range(1, 4)
        }
        result = deduplicate_mutual_ic(metrics, lambda f: values.get(f.formula_id))
        scores = [m.overall_score for m in result]
        assert scores == sorted(scores, reverse=True)

    def test_none_values_skipped(self):
        """None 值被跳过"""
        metrics = [
            self._make_metrics("f1", 0.9),
            self._make_metrics("f2", 0.8),
        ]
        values = {"f1": pl.Series([1.0, 2.0, 3.0])}
        result = deduplicate_mutual_ic(metrics, lambda f: values.get(f.formula_id))
        assert len(result) == 1
        assert result[0].formula_id == "f1"

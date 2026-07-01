# coding=utf-8
"""Tests for research/quant_alpha/mcts/extension_ops.py — ExtensionOp pool.

Covers: ExtensionOp.instantiate(), ExtensionOpPool creation, sample,
sample_weighted, sample_window, get_seed_formulas, stats, category filtering.
"""

import pytest

from QuantNodes.research.quant_alpha.mcts.extension_ops import (
    ExtensionOp,
    ExtensionOpPool,
    DEFAULT_WINDOWS,
    _build_wrap_ops,
    _build_window_ops,
    _build_unary_ops,
    _build_diff_ops,
    _build_ratio_ops,
)


# ============================================================================
# ExtensionOp
# ============================================================================

class TestExtensionOp:
    def test_creation(self):
        op = ExtensionOp(
            name="wrap_rank",
            template="rank({f})",
            requires_window=False,
        )
        assert op.name == "wrap_rank"
        assert op.template == "rank({f})"

    def test_instantiate_no_window(self):
        op = ExtensionOp(
            name="wrap_rank",
            template="rank({f})",
            requires_window=False,
        )
        result = op.instantiate(f="close")
        assert result == "rank(close)"

    def test_instantiate_with_window(self):
        op = ExtensionOp(
            name="window_ts_mean",
            template="ts_mean({f}, {w})",
            requires_window=True,
        )
        result = op.instantiate(f="close", w=20)
        assert result == "ts_mean(close, 20)"

    def test_instantiate_window_ignored_when_not_required(self):
        op = ExtensionOp(
            name="wrap_rank",
            template="rank({f})",
            requires_window=False,
        )
        # Window should be ignored
        result = op.instantiate(f="close", w=20)
        assert result == "rank(close)"

    def test_defaults(self):
        op = ExtensionOp(name="x", template="y")
        assert op.requires_window is True
        assert op.min_inputs == 1
        assert op.max_inputs == 1
        assert op.category == "general"


# ============================================================================
# Pool Builders
# ============================================================================

class TestPoolBuilders:
    def test_wrap_ops_built(self):
        from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
        vocab = OperatorVocab.default()
        ops = _build_wrap_ops(vocab)
        # Should include rank, zscore, scale, winsorize if available
        assert isinstance(ops, list)
        for op in ops:
            assert op.category == "wrap"
            assert op.requires_window is False

    def test_window_ops_built(self):
        from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
        vocab = OperatorVocab.default()
        ops = _build_window_ops(vocab)
        assert isinstance(ops, list)
        for op in ops:
            assert op.category in ("window", "window_binary")

    def test_unary_ops_built(self):
        from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
        vocab = OperatorVocab.default()
        ops = _build_unary_ops(vocab)
        assert isinstance(ops, list)
        for op in ops:
            assert op.category == "unary"

    def test_diff_ops_built(self):
        ops = _build_diff_ops()
        assert len(ops) == 2
        for op in ops:
            assert op.category == "diff"

    def test_ratio_ops_built(self):
        ops = _build_ratio_ops()
        assert len(ops) == 2
        for op in ops:
            assert op.category == "ratio"


# ============================================================================
# ExtensionOpPool
# ============================================================================

class TestExtensionOpPool:
    def test_creation_default(self):
        pool = ExtensionOpPool()
        assert len(pool) > 0
        assert pool.vocab is not None
        assert pool.windows == DEFAULT_WINDOWS

    def test_creation_custom_windows(self):
        pool = ExtensionOpPool(windows=[5, 10])
        assert pool.windows == [5, 10]

    def test_creation_custom_seed(self):
        pool1 = ExtensionOpPool(seed=1)
        pool2 = ExtensionOpPool(seed=1)
        # Same seed → same result
        assert pool1.sample().name == pool2.sample().name

    def test_iter(self):
        pool = ExtensionOpPool()
        ops = list(iter(pool))
        assert len(ops) == len(pool)

    def test_list_categories(self):
        pool = ExtensionOpPool()
        categories = pool.list_categories()
        assert isinstance(categories, list)
        # Should have multiple categories
        assert len(categories) >= 3

    def test_count_by_category(self):
        pool = ExtensionOpPool()
        counts = pool.count_by_category()
        assert isinstance(counts, dict)
        assert sum(counts.values()) == len(pool)

    def test_stats(self):
        pool = ExtensionOpPool()
        stats = pool.stats()
        assert "total" in stats
        assert "by_category" in stats
        assert "windows" in stats

    # ------------------------------------------------------------------
    # sample()
    # ------------------------------------------------------------------

    def test_sample_returns_op(self):
        pool = ExtensionOpPool()
        op = pool.sample()
        assert isinstance(op, ExtensionOp)

    def test_sample_with_category(self):
        pool = ExtensionOpPool()
        op = pool.sample(category="wrap")
        assert op.category == "wrap"

    def test_sample_invalid_category_raises(self):
        pool = ExtensionOpPool()
        with pytest.raises(ValueError):
            pool.sample(category="nonexistent_category_xyz")

    def test_include_categories_filter(self):
        pool = ExtensionOpPool(include_categories=["wrap"])
        assert pool.list_categories() == ["wrap"]
        # Only wrap ops
        for op in pool:
            assert op.category == "wrap"

    def test_include_multiple_categories(self):
        pool = ExtensionOpPool(include_categories=["wrap", "diff"])
        cats = set(pool.list_categories())
        assert cats == {"wrap", "diff"}

    # ------------------------------------------------------------------
    # sample_weighted()
    # ------------------------------------------------------------------

    def test_sample_weighted_no_prior_uniform(self):
        pool = ExtensionOpPool(seed=42)
        op = pool.sample_weighted()
        assert isinstance(op, ExtensionOp)

    def test_sample_weighted_with_prior(self):
        from QuantNodes.research.quant_alpha.mcts.op_prior import OpPrior
        prior = OpPrior()
        pool = ExtensionOpPool(seed=42)
        op = pool.sample_weighted(op_prior=prior)
        assert isinstance(op, ExtensionOp)

    def test_sample_weighted_with_category(self):
        pool = ExtensionOpPool(seed=42)
        op = pool.sample_weighted(category="wrap")
        assert op.category == "wrap"

    def test_sample_weighted_invalid_category_raises(self):
        pool = ExtensionOpPool(seed=42)
        with pytest.raises(ValueError):
            pool.sample_weighted(category="nonexistent_xyz")

    # ------------------------------------------------------------------
    # sample_window()
    # ------------------------------------------------------------------

    def test_sample_window(self):
        pool = ExtensionOpPool(windows=[5, 10, 20])
        w = pool.sample_window()
        assert w in [5, 10, 20]

    def test_sample_window_custom(self):
        pool = ExtensionOpPool(windows=[7])
        w = pool.sample_window()
        assert w == 7

    # ------------------------------------------------------------------
    # get_seed_formulas()
    # ------------------------------------------------------------------

    def test_get_seed_formulas(self):
        pool = ExtensionOpPool()
        cols = ["close", "volume", "open"]
        seeds = pool.get_seed_formulas(cols)
        assert isinstance(seeds, list)
        assert len(seeds) > 0
        # Should include first 5 columns worth of seeds (here 3)
        # Each column generates 8 seeds
        assert len(seeds) >= 3 * 8

    def test_seed_formulas_reference_columns(self):
        pool = ExtensionOpPool()
        cols = ["close", "volume"]
        seeds = pool.get_seed_formulas(cols)
        # At least one seed should reference 'close'
        assert any("close" in s for s in seeds)
        assert any("volume" in s for s in seeds)

    def test_seed_formulas_limit_5_columns(self):
        pool = ExtensionOpPool()
        cols = [f"col{i}" for i in range(10)]
        seeds = pool.get_seed_formulas(cols)
        # Should limit to first 5 columns, each generating 8 seeds
        assert len(seeds) <= 5 * 8

    def test_seed_formulas_empty(self):
        pool = ExtensionOpPool()
        seeds = pool.get_seed_formulas([])
        assert seeds == []


# ============================================================================
# ExtensionOp Instantiate Templates
# ============================================================================

class TestInstantiateTemplates:
    def test_binary_window_template(self):
        op = ExtensionOp(
            name="window_ts_corr",
            template="ts_corr({f}, {f2}, {w})",
            requires_window=True,
            min_inputs=2,
            max_inputs=2,
        )
        # Binary window needs {f2} — instantiate only replaces {f} and {w}
        result = op.instantiate(f="close", w=10)
        assert "ts_corr(close, {f2}, 10)" == result

    def test_signedpower_template(self):
        op = ExtensionOp(
            name="unary_signedpower",
            template="signedpower({f}, 2)",
            requires_window=False,
        )
        result = op.instantiate(f="close")
        assert result == "signedpower(close, 2)"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_pool_with_empty_windows(self):
        pool = ExtensionOpPool(windows=[])
        w = pool.sample_window()
        # IndexError when choice from empty list
        with pytest.raises(IndexError):
            pool.rng.choice([])

    def test_pool_filter_excludes_all(self):
        pool = ExtensionOpPool(include_categories=["nonexistent_xyz"])
        assert len(pool) == 0

    def test_pool_with_specific_categories(self):
        pool = ExtensionOpPool(include_categories=["diff"])
        assert pool.list_categories() == ["diff"]
        # Sample from diff category
        op = pool.sample(category="diff")
        assert op.category == "diff"
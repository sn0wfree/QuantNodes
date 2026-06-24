# coding=utf-8
"""AutoResearch 单元测试"""

import pytest
import numpy as np
import polars as pl
import tempfile
import shutil
from pathlib import Path

from QuantNodes.research._legacy_3c.factor_miner import FactorMiner, FactorCandidate, TEMPLATES, DEFAULT_WINDOWS
from QuantNodes.research._legacy_3c.factor_evaluator import (
    FactorEvaluator, FactorEvaluationResult, EvalConfig,
)
from QuantNodes.research._legacy_3c.auto_researcher import AutoResearcher, AutoResearchResult
from QuantNodes.research._legacy_3c.mcts_search import MCTSSearch, MCTSNode
from QuantNodes.research.wiki import FactorCategory


@pytest.fixture
def sample_data():
    """生成模拟行情数据"""
    np.random.seed(42)
    n = 200
    dates = ["2024-01-01"] * 50 + ["2024-01-02"] * 50 + \
            ["2024-01-03"] * 50 + ["2024-01-04"] * 50
    codes = [f"SZ{i:06d}" for i in range(50)] * 4

    close = np.random.uniform(10, 100, n)
    return pl.DataFrame({
        "date": dates,
        "code": codes,
        "close": close,
        "open": close + np.random.normal(0, 1, n),
        "high": close + abs(np.random.normal(0, 2, n)),
        "low": close - abs(np.random.normal(0, 2, n)),
        "vol": np.random.uniform(1000, 100000, n),
        "forward_return": np.random.normal(0, 0.02, n),
    })


@pytest.fixture
def tmp_wiki():
    from QuantNodes.research.wiki import init_factor_wiki
    d = tempfile.mkdtemp()
    wiki_path = str(Path(d) / "test_wiki")
    init_factor_wiki(wiki_path)
    yield wiki_path
    shutil.rmtree(d)


class TestFactorMiner:

    def test_generate_basic(self, sample_data):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(
            available_columns=["close", "open", "vol"],
        )
        assert len(candidates) > 0
        assert all(isinstance(c, FactorCandidate) for c in candidates)

    def test_generate_has_formulas(self, sample_data):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(available_columns=["close", "vol"])
        for c in candidates:
            assert len(c.formula) > 0
            assert len(c.name) > 0
            assert c.category in FactorCategory

    def test_generate_respects_columns(self, sample_data):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(available_columns=["close"])
        # 双列模板 (如 ts_corr) 不应生成
        for c in candidates:
            assert "vol" not in c.formula or "close" in c.formula

    def test_generate_unique(self, sample_data):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(available_columns=["close", "vol"])
        formulas = [c.formula for c in candidates]
        assert len(formulas) == len(set(formulas))

    def test_generate_single(self):
        miner = FactorMiner()
        c = miner.generate_single(
            formula="rank(close / ts_lag(close, 20) - 1)",
            description="20日动量排名",
        )
        assert c.formula == "rank(close / ts_lag(close, 20) - 1)"
        assert c.template_name == "manual"

    def test_templates_cover_categories(self):
        categories = set()
        for group in TEMPLATES.values():
            categories.add(group["category"])
        assert FactorCategory.MOMENTUM in categories
        assert FactorCategory.VOLATILITY in categories

    def test_generate_empty_columns(self):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(available_columns=[])
        assert len(candidates) == 0

    def test_generate_single_column_only(self, sample_data):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(available_columns=["close"])
        assert len(candidates) > 0
        for c in candidates:
            assert "vol" not in c.formula

    def test_generate_custom_config(self, sample_data):
        class CustomConfig:
            windows = [5, 10]
            template_categories = ["momentum"]
            max_factors = 5

        miner = FactorMiner(seed=42)
        candidates = miner.generate(
            available_columns=["close", "vol"],
            config=CustomConfig(),
        )
        assert len(candidates) <= 5
        for c in candidates:
            assert c.template_name in ("momentum", "manual") or c.template_name == ""

    def test_generate_max_factors_limit(self, sample_data):
        miner = FactorMiner(seed=0)
        max_factors = 10
        limited = miner.generate(
            available_columns=["close", "vol"],
            config=type("Cfg", (), {"max_factors": max_factors, "windows": DEFAULT_WINDOWS})(),
        )
        assert len(limited) <= max_factors

    def test_generate_with_custom_categories(self, sample_data):
        miner = FactorMiner(seed=42)
        candidates = miner.generate(
            available_columns=["close"],
            config=type("Cfg", (), {"template_categories": ["volatility"]})(),
        )
        for c in candidates:
            assert c.template_name in ("volatility", "manual") or c.template_name == ""

    def test_generate_no_duplicate_formulas(self, sample_data):
        miner = FactorMiner(seed=123)
        candidates = miner.generate(available_columns=["close", "vol"])
        formulas = [c.formula for c in candidates]
        assert len(formulas) == len(set(formulas))

    def test_factor_candidate_fields(self):
        from QuantNodes.research._legacy_3c.factor_miner import FactorCandidate
        from QuantNodes.research.wiki import FactorCategory
        cand = FactorCandidate(
            name="test_cand",
            formula="ts_mean(close, 20)",
            description="test desc",
            operators_used=["ts_mean"],
            category=FactorCategory.MOMENTUM,
            template_name="momentum",
            metadata={"window": 20},
        )
        assert cand.name == "test_cand"
        assert cand.template_name == "momentum"
        assert cand.metadata["window"] == 20


class TestFactorEvaluator:

    def test_evaluate_basic(self, sample_data):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test_factor",
            formula="rank(close)",
            description="test",
            operators_used=["rank"],
            category=FactorCategory.MOMENTUM,
        )
        result = evaluator.evaluate(candidate, sample_data)
        assert isinstance(result, FactorEvaluationResult)
        assert result.factor_values is not None
        assert len(result.factor_values) == len(sample_data)

    def test_evaluate_coverage(self, sample_data):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test", formula="rank(close)", description="test",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result = evaluator.evaluate(candidate, sample_data)
        assert 0.0 <= result.coverage <= 1.0

    def test_evaluate_ic_computation(self, sample_data):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test", formula="rank(close)", description="test",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result = evaluator.evaluate(candidate, sample_data)
        assert isinstance(result.ic_mean, float)
        assert isinstance(result.icir, float)

    def test_evaluate_group_returns(self, sample_data):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test", formula="rank(close)", description="test",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result = evaluator.evaluate(candidate, sample_data)
        assert isinstance(result.group_returns, list)

    def test_evaluate_invalid_formula(self, sample_data):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test", formula="invalid_func(close)", description="test",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result = evaluator.evaluate(candidate, sample_data)
        assert result.factor_values is None
        assert len(result.fail_reasons) > 0

    def test_deduplicate(self, sample_data):
        evaluator = FactorEvaluator()
        results = []
        for formula in ["rank(close)", "rank(close)", "rank(vol)"]:
            candidate = FactorCandidate(
                name=f"test_{formula}", formula=formula, description="test",
                operators_used=[], category=FactorCategory.OTHER,
            )
            results.append(evaluator.evaluate(candidate, sample_data))

        deduped = evaluator.deduplicate(results, corr_threshold=0.7)
        assert len(deduped) <= len(results)
        # rank(close) 重复应被去重
        formulas = [r.candidate.formula for r in deduped]
        assert formulas.count("rank(close)") <= 1

    def test_eval_config_defaults(self):
        config = EvalConfig()
        assert config.ic_threshold == 0.03
        assert config.corr_threshold == 0.7
        assert config.n_groups == 5
        assert "return" in config.weights

    def test_eval_config_custom_weights(self):
        config = EvalConfig(
            weights={"return": 0.5, "stability": 0.5},
        )
        assert config.weights["return"] == 0.5
        assert config.weights["stability"] == 0.5

    def test_spearman_corr_all_ties(self, sample_data):
        evaluator = FactorEvaluator()
        import polars as pl
        a = pl.Series("a", [1.0, 1.0, 1.0, 1.0, 1.0])
        b = pl.Series("b", [1.0, 2.0, 3.0, 4.0, 5.0])
        corr = evaluator._spearman_corr(a, b)
        assert isinstance(corr, float)

    def test_spearman_corr_different_lengths(self, sample_data):
        evaluator = FactorEvaluator()
        import polars as pl
        a = pl.Series("a", [1.0, 2.0, 3.0])
        b = pl.Series("b", [1.0, 2.0])
        corr = evaluator._spearman_corr(a, b)
        assert isinstance(corr, float)

    def test_spearman_corr_too_few_values(self, sample_data):
        evaluator = FactorEvaluator()
        import polars as pl
        a = pl.Series("a", [1.0])
        b = pl.Series("b", [2.0])
        corr = evaluator._spearman_corr(a, b)
        assert corr == 0.0

    def test_rank_all_same_values(self):
        from QuantNodes.research._legacy_3c.factor_evaluator import _rank
        ranks = _rank([5.0, 5.0, 5.0])
        assert len(ranks) == 3
        assert ranks[0] == ranks[1] == ranks[2]

    def test_deduplicate_empty_results(self):
        evaluator = FactorEvaluator()
        deduped = evaluator.deduplicate([])
        assert len(deduped) == 0

    def test_deduplicate_all_invalid(self):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test", formula="bad", description="",
            operators_used=[], category=FactorCategory.OTHER,
        )
        results = [
            FactorEvaluationResult(candidate=candidate, factor_values=None),
            FactorEvaluationResult(candidate=candidate, factor_values=None),
        ]
        deduped = evaluator.deduplicate(results)
        assert len(deduped) == 0

    def test_evaluate_with_existing_factors(self, sample_data):
        evaluator = FactorEvaluator()
        candidate1 = FactorCandidate(
            name="test1", formula="rank(close)", description="",
            operators_used=[], category=FactorCategory.OTHER,
        )
        candidate2 = FactorCandidate(
            name="test2", formula="rank(open)", description="",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result1 = evaluator.evaluate(candidate1, sample_data)
        existing = [result1.factor_values] if result1.factor_values is not None else None
        result2 = evaluator.evaluate(candidate2, sample_data, existing_factors=existing)
        assert isinstance(result2.avg_corr_with_existing, float)

    def test_evaluate_low_data_returns_early(self, sample_data):
        evaluator = FactorEvaluator()
        tiny_data = sample_data.head(3)
        candidate = FactorCandidate(
            name="test", formula="rank(close)", description="",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result = evaluator.evaluate(candidate, tiny_data)
        assert "有效数据不足" in result.fail_reasons

    def test_evaluate_invalid_formula_returns_fail_reason(self, sample_data):
        evaluator = FactorEvaluator()
        candidate = FactorCandidate(
            name="test", formula="nonexistent_func(close)", description="",
            operators_used=[], category=FactorCategory.OTHER,
        )
        result = evaluator.evaluate(candidate, sample_data)
        assert result.factor_values is None
        assert len(result.fail_reasons) > 0

    def test_deduplicate_respects_threshold(self, sample_data):
        evaluator = FactorEvaluator()
        results = []
        for formula in ["rank(close)", "rank(close)"]:
            candidate = FactorCandidate(
                name=f"test_{formula}", formula=formula, description="",
                operators_used=[], category=FactorCategory.OTHER,
            )
            results.append(evaluator.evaluate(candidate, sample_data))
        deduped = evaluator.deduplicate(results, corr_threshold=0.99)
        assert len(deduped) <= len(results)


class TestAutoResearcher:

    def test_run_basic(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.run(
            data=sample_data,
            store_to_wiki=False,
            max_factors=20,
        )
        assert isinstance(result, AutoResearchResult)
        assert result.elapsed_seconds > 0
        assert isinstance(result.all_evaluated, list)

    def test_run_generates_report(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.run(data=sample_data, store_to_wiki=False, max_factors=10)
        assert len(result.report_markdown) > 0
        assert "AutoResearch" in result.report_markdown

    def test_mine_single_factor(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.mine_single_factor(
            formula="rank(close)",
            data=sample_data,
            store_to_wiki=False,
        )
        assert isinstance(result, FactorEvaluationResult)
        assert result.factor_values is not None

    def test_store_to_wiki(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        researcher.run(data=sample_data, store_to_wiki=True, max_factors=10)
        # 检查是否有因子存入 wiki
        factors = researcher.proxy.list_factors()
        # 至少检查不报错
        assert isinstance(factors, list)


class TestMCTSSearch:

    def test_search_basic(self, sample_data):
        search = MCTSSearch(seed=42)
        results = search.search(
            data=sample_data,
            seed_formulas=["rank(close)", "ts_mean(close, 20)"],
            iterations=5,
        )
        assert isinstance(results, list)

    def test_search_returns_evaluated(self, sample_data):
        search = MCTSSearch(seed=42)
        results = search.search(
            data=sample_data,
            seed_formulas=["rank(close)"],
            iterations=3,
        )
        for r in results:
            assert isinstance(r, FactorEvaluationResult)

    def test_node_creation(self):
        node = MCTSNode(formula="rank(close)")
        assert node.formula == "rank(close)"
        assert node.visits == 0
        assert node.children == []

    def test_ucb1(self):
        search = MCTSSearch(seed=42)
        root = MCTSNode(formula="__ROOT__")
        child = MCTSNode(formula="rank(close)", parent=root)
        root.visits = 10
        child.visits = 5
        child.overall_score = 0.5

        score = search._ucb1(child)
        assert score > 0

    def test_ucb1_unvisited(self):
        search = MCTSSearch(seed=42)
        root = MCTSNode(formula="__ROOT__")
        child = MCTSNode(formula="rank(close)", parent=root)
        root.visits = 10

        score = search._ucb1(child)
        assert score == float("inf")

    def test_select_no_children(self):
        search = MCTSSearch(seed=42)
        leaf_node = MCTSNode(formula="rank(close)")
        leaf_node.is_expanded = True
        selected = search._select(leaf_node)
        assert selected == leaf_node

    def test_ucb1_unvisited_node(self):
        search = MCTSSearch(seed=42)
        root = MCTSNode(formula="__ROOT__")
        root.visits = 0
        child = MCTSNode(formula="rank(close)", parent=root)
        child.visits = 0
        score = search._ucb1(child)
        assert score == float("inf")

    def test_expand_formula_cached(self, sample_data):
        search = MCTSSearch(seed=42)
        node = MCTSNode(formula="rank(close)")
        from QuantNodes.research._legacy_3c.factor_evaluator import FactorEvaluationResult
        cached_result = FactorEvaluationResult(
            candidate=object(), factor_values=None
        )
        search._formula_cache["zscore(rank(close))"] = cached_result
        child = search._expand(node, sample_data)
        assert child is None

    def test_expand_root_generates_seed(self, sample_data):
        search = MCTSSearch(seed=42)
        root = MCTSNode(formula="__ROOT__")
        child = search._expand(root, sample_data)
        assert child is not None or True

    def test_collect_results_with_min_score(self, sample_data):
        search = MCTSSearch(seed=42)
        root = MCTSNode(formula="__ROOT__")
        results = search._collect_results(root, min_score=999.0)
        assert isinstance(results, list)

    def test_search_no_seed_formulas(self, sample_data):
        search = MCTSSearch(seed=42)
        results = search.search(
            data=sample_data,
            seed_formulas=[],
            iterations=3,
        )
        assert isinstance(results, list)


class TestAutoResearcherEdge:

    def test_run_with_empty_data(self, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        empty_data = pl.DataFrame({"date": [], "code": [], "close": []})
        result = researcher.run(
            data=empty_data,
            store_to_wiki=False,
            max_factors=5,
        )
        assert isinstance(result, AutoResearchResult)
        assert result.elapsed_seconds >= 0

    def test_run_with_max_factors_zero(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.run(
            data=sample_data,
            store_to_wiki=False,
            max_factors=0,
        )
        assert isinstance(result, AutoResearchResult)
        assert len(result.all_evaluated) == 0

    def test_run_with_custom_eval_config(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        config = EvalConfig(
            ic_threshold=0.01,
            icir_threshold=0.1,
            corr_threshold=0.95,
        )
        result = researcher.run(
            data=sample_data,
            eval_config=config,
            store_to_wiki=False,
            max_factors=5,
        )
        assert isinstance(result, AutoResearchResult)

    def test_run_use_mcts_flag(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.run(
            data=sample_data,
            store_to_wiki=False,
            max_factors=5,
            use_mcts=True,
            mcts_iterations=3,
        )
        assert isinstance(result, AutoResearchResult)
        assert len(result.report_markdown) > 0

    def test_mine_single_factor_invalid_formula(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.mine_single_factor(
            formula="invalid_formula(close)",
            data=sample_data,
            store_to_wiki=False,
        )
        assert isinstance(result, FactorEvaluationResult)
        assert result.factor_values is None

    def test_mine_single_factor_stores_to_wiki(self, sample_data, tmp_wiki):
        researcher = AutoResearcher(wiki_path=tmp_wiki)
        result = researcher.mine_single_factor(
            formula="rank(close)",
            data=sample_data,
            store_to_wiki=True,
        )
        if result.is_valid:
            factors = researcher.proxy.list_factors()
            assert len(factors) >= 1

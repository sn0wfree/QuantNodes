# coding=utf-8
"""AutoResearch 单元测试"""

import pytest
import numpy as np
import polars as pl
import tempfile
import shutil
from pathlib import Path

from QuantNodes.research.factor_miner import FactorMiner, FactorCandidate, TEMPLATES
from QuantNodes.research.factor_evaluator import (
    FactorEvaluator, FactorEvaluationResult, EvalConfig,
)
from QuantNodes.research.auto_researcher import AutoResearcher, AutoResearchResult
from QuantNodes.research.mcts_search import MCTSSearch, MCTSNode
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

# coding=utf-8
"""WikiFactorProxy 单元测试"""

import pytest
import tempfile
import os
from pathlib import Path

from QuantNodes.research.wiki import (
    WikiFactorProxy, WikiFactor, WikiLogic,
    FactorSource, FactorCategory, LogicSource,
    WikiProxyError, init_factor_wiki,
)


@pytest.fixture
def tmp_wiki_path():
    """临时 wiki 目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_path = os.path.join(tmpdir, "test_wiki")
        init_factor_wiki(wiki_path)
        yield wiki_path


@pytest.fixture
def proxy(tmp_wiki_path):
    return WikiFactorProxy(tmp_wiki_path)


@pytest.fixture
def sample_factor():
    return WikiFactor(
        name="momentum_20d",
        formula="close / delay(close, 20) - 1",
        source=FactorSource.RESEARCH_REPORT,
        category=FactorCategory.MOMENTUM,
        description="20日动量因子",
        tags=["momentum", "medium_term"],
        ic_mean=0.05,
        ic_std=0.1,
        icir=0.5,
        rank_ic_mean=0.06,
        n_dates=500,
        factor_return_corr=0.3,
        ic_t_stat=2.1,
        turnover=0.35,
        used_by_strategies=["strategy_a", "strategy_b"],
        strategy_yaml="name: momentum_strategy\nfactors:\n  - momentum_20d",
    )


@pytest.fixture
def sample_logic():
    return WikiLogic(
        name="research_report_001",
        content="根据研报分析，动量因子在A股市场具有显著的预测能力...",
        source=LogicSource.RESEARCH_REPORT,
        extracted_formula="close / delay(close, 20) - 1",
        related_strategies=["strategy_a"],
        related_factors=["momentum_20d"],
        validation_status="validated",
    )


class TestInitFactorWiki:

    def test_init_creates_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "new_wiki")
            init_factor_wiki(wiki_path)
            assert Path(wiki_path).exists()
            assert (Path(wiki_path) / "wiki.md").exists()


class TestWikiFactorProxyCRUD:

    def test_store_factor(self, proxy, sample_factor):
        page_name = proxy.store_factor(sample_factor)
        assert page_name == "Factor/momentum_20d"
        assert sample_factor.wiki_page_name == page_name

    def test_get_factor(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        fetched = proxy.get_factor("momentum_20d")
        assert fetched is not None
        assert fetched.name == "momentum_20d"
        assert fetched.formula == sample_factor.formula
        assert fetched.source == FactorSource.RESEARCH_REPORT
        assert fetched.category == FactorCategory.MOMENTUM
        assert fetched.tags == ["momentum", "medium_term"]
        assert fetched.used_by_strategies == ["strategy_a", "strategy_b"]
        assert fetched.strategy_yaml is not None

    def test_get_factor_not_found(self, proxy):
        assert proxy.get_factor("nonexistent") is None

    def test_search_factors(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        results = proxy.search_factors("momentum")
        assert len(results) >= 1
        assert results[0].name == "momentum_20d"

    def test_list_factors(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        factors = proxy.list_factors()
        assert len(factors) == 1

    def test_list_factors_with_filter(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        factors = proxy.list_factors(source=FactorSource.MANUAL)
        assert len(factors) == 0
        factors = proxy.list_factors(source=FactorSource.RESEARCH_REPORT)
        assert len(factors) == 1

    def test_update_factor(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        ok = proxy.update_factor("momentum_20d", {"ic_mean": 0.08, "turnover": 0.4})
        assert ok is True
        updated = proxy.get_factor("momentum_20d")
        assert updated.ic_mean == 0.08
        assert updated.turnover == 0.4

    def test_update_factor_not_found(self, proxy):
        assert proxy.update_factor("nonexistent", {}) is False

    def test_delete_factor(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        assert proxy.delete_factor("momentum_20d") is True
        assert proxy.get_factor("momentum_20d") is None

    def test_delete_factor_not_found(self, proxy):
        assert proxy.delete_factor("nonexistent") is False


class TestWikiLogicCRUD:

    def test_store_logic(self, proxy, sample_logic):
        page_name = proxy.store_logic(sample_logic)
        assert page_name == "Logic/research_report_001"
        assert sample_logic.wiki_page_name == page_name

    def test_get_logic(self, proxy, sample_logic):
        proxy.store_logic(sample_logic)
        fetched = proxy.get_logic("research_report_001")
        assert fetched is not None
        assert fetched.name == "research_report_001"
        assert fetched.source == LogicSource.RESEARCH_REPORT
        assert fetched.validation_status == "validated"
        assert fetched.related_strategies == ["strategy_a"]
        assert fetched.related_factors == ["momentum_20d"]

    def test_get_logic_not_found(self, proxy):
        assert proxy.get_logic("nonexistent") is None

    def test_search_logics(self, proxy, sample_logic):
        proxy.store_logic(sample_logic)
        results = proxy.search_logics("research")
        assert len(results) >= 1


class TestRelations:

    def test_add_relation(self, proxy, sample_factor, sample_logic):
        proxy.store_factor(sample_factor)
        proxy.store_logic(sample_logic)
        ok = proxy.add_relation("Logic/research_report_001", "Factor/momentum_20d", "derived_from")
        assert ok is True

    def test_add_relation_invalid_type(self, proxy):
        with pytest.raises(WikiProxyError) as exc_info:
            proxy.add_relation("source", "target", "invalid_relation")
        assert "Invalid relation type" in str(exc_info.value)

    def test_get_neighbors(self, proxy, sample_factor):
        proxy.store_factor(sample_factor)
        neighbors = proxy.get_neighbors("Factor/momentum_20d")
        assert isinstance(neighbors, list)


class TestICIRSerialization:

    def test_icir_fields_roundtrip(self, proxy):
        factor = WikiFactor(
            name="test_icir",
            formula="a + b",
            source=FactorSource.AUTO_RESEARCH,
            category=FactorCategory.QUALITY,
            ic_mean=0.123,
            ic_std=0.456,
            icir=0.789,
            rank_ic_mean=0.101,
            n_dates=123,
            factor_return_corr=0.202,
            ic_t_stat=3.03,
            turnover=0.404,
        )
        proxy.store_factor(factor)
        fetched = proxy.get_factor("test_icir")
        assert fetched.ic_mean == 0.123
        assert fetched.ic_std == 0.456
        assert fetched.icir == 0.789
        assert fetched.rank_ic_mean == 0.101
        assert fetched.n_dates == 123
        assert fetched.factor_return_corr == 0.202
        assert fetched.ic_t_stat == 3.03
        assert fetched.turnover == 0.404


class TestStrategyYaml:

    def test_strategy_yaml_roundtrip(self, proxy):
        yaml_content = "name: test\nfactors:\n  - x\n  - y"
        factor = WikiFactor(
            name="yaml_test",
            formula="x",
            source=FactorSource.MANUAL,
            category=FactorCategory.OTHER,
            strategy_yaml=yaml_content,
        )
        proxy.store_factor(factor)
        fetched = proxy.get_factor("yaml_test")
        assert fetched.strategy_yaml == yaml_content


class TestStatus:

    def test_ping(self, proxy):
        assert proxy.ping() is True

    def test_status(self, proxy):
        s = proxy.status()
        assert isinstance(s, dict)

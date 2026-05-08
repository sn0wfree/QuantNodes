# coding=utf-8
"""
测试 WikiTool
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.agent.tools.wiki import WikiTool
from QuantNodes.research.wiki import (
    WikiFactor,
    WikiStrategy,
    FactorSource,
    FactorCategory,
)


@pytest.fixture
def temp_wiki_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def wiki_tool(temp_wiki_dir):
    with patch("QuantNodes.research.wiki.create_wiki") as mock_create_wiki:
        mock_wiki = MagicMock()
        mock_wiki.root.exists.return_value = True
        mock_create_wiki.return_value = mock_wiki
        tool = WikiTool(wiki_path=str(temp_wiki_dir))
        tool.proxy._wiki = mock_wiki
        yield tool


class TestWikiToolStoreFactor:
    @pytest.mark.asyncio
    async def test_store_factor_success(self, wiki_tool):
        wiki_tool.proxy.store_factor = MagicMock(return_value="Factor/test_factor")
        result = await wiki_tool.execute(
            action="store_factor",
            name="test_factor",
            formula="close / open - 1",
            source="manual",
            category="momentum",
        )
        assert result == "Factor/test_factor"
        wiki_tool.proxy.store_factor.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_factor_with_ic(self, wiki_tool):
        wiki_tool.proxy.store_factor = MagicMock(return_value="Factor/momentum_factor")
        result = await wiki_tool.execute(
            action="store_factor",
            name="momentum_factor",
            formula="close / close.shift(20) - 1",
            source="auto_research",
            category="momentum",
            ic_mean=0.042,
            icir=0.65,
            tags=["momentum", "dual_ma"],
        )
        assert result == "Factor/momentum_factor"


class TestWikiToolGetFactor:
    @pytest.mark.asyncio
    async def test_get_factor_found(self, wiki_tool):
        mock_factor = WikiFactor(
            name="test_factor",
            formula="close / open - 1",
            source=FactorSource.MANUAL,
            category=FactorCategory.MOMENTUM,
        )
        mock_factor.wiki_page_name = "Factor/test_factor"
        wiki_tool.proxy.get_factor = MagicMock(return_value=mock_factor)

        result = await wiki_tool.execute(action="get_factor", name="test_factor")
        assert result is not None
        assert result["name"] == "test_factor"
        assert result["formula"] == "close / open - 1"

    @pytest.mark.asyncio
    async def test_get_factor_not_found(self, wiki_tool):
        wiki_tool.proxy.get_factor = MagicMock(return_value=None)
        result = await wiki_tool.execute(action="get_factor", name="nonexistent")
        assert result is None


class TestWikiToolSearch:
    @pytest.mark.asyncio
    async def test_search_factors(self, wiki_tool):
        mock_factors = [
            WikiFactor(
                name="factor1",
                formula="close / open",
                source=FactorSource.MANUAL,
                category=FactorCategory.MOMENTUM,
            ),
            WikiFactor(
                name="factor2",
                formula="volume / open",
                source=FactorSource.MANUAL,
                category=FactorCategory.VOLATILITY,
            ),
        ]
        wiki_tool.proxy.search_factors = MagicMock(return_value=mock_factors)

        result = await wiki_tool.execute(action="search_factors", query="momentum", limit=10)
        assert len(result) == 2
        assert result[0]["name"] == "factor1"


class TestWikiToolStoreStrategy:
    @pytest.mark.asyncio
    async def test_store_strategy_success(self, wiki_tool):
        wiki_tool.proxy.store_strategy = MagicMock(return_value="Strategy/dual_ma")
        result = await wiki_tool.execute(
            action="store_strategy",
            name="dual_ma",
            strategy_yaml="name: dual_ma\nfactors:\n  - ma_20: rolling_mean(close, 20)",
            description="经典双均线策略",
            category="trend",
            tags=["trend", "ma"],
        )
        assert result == "Strategy/dual_ma"


class TestWikiToolGetStrategy:
    @pytest.mark.asyncio
    async def test_get_strategy_found(self, wiki_tool):
        mock_strategy = WikiStrategy(
            name="dual_ma",
            strategy_yaml="name: dual_ma",
            description="经典双均线策略",
            category="trend",
        )
        mock_strategy.wiki_page_name = "Strategy/dual_ma"
        wiki_tool.proxy.get_strategy = MagicMock(return_value=mock_strategy)

        result = await wiki_tool.execute(action="get_strategy", name="dual_ma")
        assert result is not None
        assert result["name"] == "dual_ma"

    @pytest.mark.asyncio
    async def test_get_strategy_not_found(self, wiki_tool):
        wiki_tool.proxy.get_strategy = MagicMock(return_value=None)
        result = await wiki_tool.execute(action="get_strategy", name="nonexistent")
        assert result is None


class TestWikiToolPing:
    @pytest.mark.asyncio
    async def test_ping_success(self, wiki_tool):
        wiki_tool.proxy.ping = MagicMock(return_value=True)
        result = await wiki_tool.execute(action="ping")
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_failure(self, wiki_tool):
        wiki_tool.proxy.ping = MagicMock(return_value=False)
        result = await wiki_tool.execute(action="ping")
        assert result is False


class TestWikiToolStatus:
    @pytest.mark.asyncio
    async def test_status(self, wiki_tool):
        wiki_tool.proxy.status = MagicMock(return_value={"total_pages": 10, "factors": 5})
        result = await wiki_tool.execute(action="status")
        assert result["total_pages"] == 10
        assert result["factors"] == 5


class TestWikiToolAddRelation:
    @pytest.mark.asyncio
    async def test_add_relation_success(self, wiki_tool):
        wiki_tool.proxy.add_relation = MagicMock(return_value=True)
        result = await wiki_tool.execute(
            action="add_relation",
            source_name="Strategy/dual_ma",
            target_name="Factor/momentum",
            relation="uses",
        )
        assert result is True
        wiki_tool.proxy.add_relation.assert_called_once_with(
            "Strategy/dual_ma", "Factor/momentum", "uses"
        )


class TestWikiToolSearch:
    @pytest.mark.asyncio
    async def test_search(self, wiki_tool):
        mock_results = [
            {"page_name": "Factor/momentum", "content": "momentum factor"},
            {"page_name": "Strategy/momentum", "content": "momentum strategy"},
        ]
        wiki_tool.proxy.wiki.search = MagicMock(return_value=mock_results)

        result = await wiki_tool.execute(action="search", query="momentum", limit=10)
        assert len(result) == 2


class TestWikiToolErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_action(self, wiki_tool):
        with pytest.raises(ValueError) as exc_info:
            await wiki_tool.execute(action="unknown_action")
        assert "Unknown action" in str(exc_info.value)
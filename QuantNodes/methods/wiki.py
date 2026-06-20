# coding=utf-8
"""
Wiki Method

query_wiki(query, action) -> WikiResult

Provides access to QuantNodes Wiki knowledge base.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from QuantNodes.research.wiki import (
    WikiFactorProxy,
    WikiFactor,
    WikiStrategy,
    FactorSource,
    FactorCategory,
)


@dataclass
class WikiResult:
    status: str
    data: Any = None
    errors: List[str] = field(default_factory=list)


class WikiOperations:
    """Wiki knowledge base operations for external agents."""

    def __init__(self, wiki_path: str = None):
        self.proxy = WikiFactorProxy(wiki_path) if wiki_path else WikiFactorProxy()
        self._logger = logging.getLogger("methods.wiki")

    def ping(self) -> bool:
        """Check if Wiki is available."""
        return self.proxy.ping()

    def status(self) -> Dict[str, Any]:
        """Get Wiki status statistics."""
        return self.proxy.status()

    def search(self, query: str, limit: int = 10) -> WikiResult:
        """Full-text search Wiki."""
        try:
            results = self.proxy.wiki.search(query, limit=limit)
            return WikiResult(status="success", data=results)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def get_factor(self, name: str) -> WikiResult:
        """Get factor details."""
        try:
            factor = self.proxy.get_factor(name)
            if factor:
                return WikiResult(
                    status="success",
                    data=self._factor_to_dict(factor)
                )
            return WikiResult(status="success", data=None)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def search_factors(self, query: str, limit: int = 10) -> WikiResult:
        """Search factors."""
        try:
            factors = self.proxy.search_factors(query, limit=limit)
            return WikiResult(
                status="success",
                data=[self._factor_to_dict(f) for f in factors]
            )
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def list_factors(
        self,
        source: str = None,
        category: str = None,
        tags: List[str] = None,
        limit: int = 50
    ) -> WikiResult:
        """List factors with optional filters."""
        try:
            if source:
                source = FactorSource(source)
            if category:
                category = FactorCategory(category)
            factors = self.proxy.list_factors(
                source=source,
                category=category,
                tags=tags,
                limit=limit
            )
            return WikiResult(
                status="success",
                data=[self._factor_to_dict(f) for f in factors]
            )
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def store_factor(self, **kwargs) -> WikiResult:
        """Store a validated factor in Wiki."""
        try:
            source = FactorSource(kwargs.pop("source"))
            category = FactorCategory(kwargs.pop("category"))
            factor = WikiFactor(source=source, category=category, **kwargs)
            result = self.proxy.store_factor(factor)
            return WikiResult(status="success", data=result)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def get_strategy(self, name: str) -> WikiResult:
        """Get strategy details."""
        try:
            strategy = self.proxy.get_strategy(name)
            if strategy:
                return WikiResult(
                    status="success",
                    data={
                        "name": strategy.name,
                        "description": strategy.description,
                        "category": strategy.category,
                        "tags": strategy.tags,
                        "strategy_yaml": strategy.strategy_yaml,
                        "backtest_result": strategy.backtest_result,
                        "wiki_page_name": strategy.wiki_page_name,
                    }
                )
            return WikiResult(status="success", data=None)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def list_strategies(
        self,
        category: str = None,
        tags: List[str] = None,
        limit: int = 50
    ) -> WikiResult:
        """List strategies."""
        try:
            strategies = self.proxy.list_strategies(
                category=category,
                tags=tags,
                limit=limit
            )
            return WikiResult(
                status="success",
                data=[
                    {
                        "name": s.name,
                        "description": s.description,
                        "category": s.category,
                        "tags": s.tags,
                        "wiki_page_name": s.wiki_page_name,
                    }
                    for s in strategies
                ]
            )
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def store_strategy(self, **kwargs) -> WikiResult:
        """Store a strategy in Wiki."""
        try:
            strategy = WikiStrategy(**kwargs)
            result = self.proxy.store_strategy(strategy)
            return WikiResult(status="success", data=result)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def add_relation(
        self,
        source_name: str,
        target_name: str,
        relation: str
    ) -> WikiResult:
        """Add a knowledge graph relation."""
        try:
            result = self.proxy.add_relation(source_name, target_name, relation)
            return WikiResult(status="success", data=result)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def get_neighbors(self, name: str) -> WikiResult:
        """Get knowledge graph neighbors."""
        try:
            neighbors = self.proxy.get_neighbors(name)
            return WikiResult(status="success", data=neighbors)
        except Exception as e:
            return WikiResult(status="error", errors=[str(e)])

    def _factor_to_dict(self, factor: WikiFactor) -> Dict[str, Any]:
        return {
            "name": factor.name,
            "formula": factor.formula,
            "source": factor.source.value,
            "category": factor.category.value,
            "ic_mean": factor.ic_mean,
            "ic_std": factor.ic_std,
            "icir": factor.icir,
            "rank_ic_mean": factor.rank_ic_mean,
            "tags": factor.tags,
            "wiki_page_name": factor.wiki_page_name,
        }


def query_wiki(query: str, action: str = "search", **kwargs) -> WikiResult:
    """Query Wiki knowledge base.

    Args:
        query: Search query or entity name
        action: Action to perform (search, get_factor, list_factors, etc.)
        **kwargs: Additional parameters for specific actions

    Returns:
        WikiResult with queried data
    """
    wiki = WikiOperations()

    action_map = {
        "search": lambda: wiki.search(query, limit=kwargs.get("limit", 10)),
        "get_factor": lambda: wiki.get_factor(query),
        "list_factors": lambda: wiki.list_factors(
            source=kwargs.get("source"),
            category=kwargs.get("category"),
            tags=kwargs.get("tags"),
            limit=kwargs.get("limit", 50),
        ),
        "get_strategy": lambda: wiki.get_strategy(query),
        "list_strategies": lambda: wiki.list_strategies(
            category=kwargs.get("category"),
            tags=kwargs.get("tags"),
            limit=kwargs.get("limit", 50),
        ),
        "ping": lambda: WikiResult(status="success", data=wiki.ping()),
        "status": lambda: WikiResult(status="success", data=wiki.status()),
    }

    if action not in action_map:
        return WikiResult(
            status="error",
            errors=[f"Unknown action: {action}. Valid actions: {list(action_map.keys())}"]
        )

    return action_map[action]()

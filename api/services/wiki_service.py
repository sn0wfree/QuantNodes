"""
Wiki Service - 直接调用 QuantNodes.research.wiki（v3.0.0+ 不再经 agent）

v3.0.0 之前：本服务通过 ``QuantNodes.agent.tools.wiki.WikiTool`` 间接访问 wiki。
v3.0.0 起：直接 import ``QuantNodes.research.wiki.WikiFactorProxy`` 跳过 agent 抽象层。
"""

from typing import Optional, List, Dict, Any

from QuantNodes.research.wiki import (
    FactorSource,
    FactorCategory,
    WikiFactor,
    WikiStrategy,
    WikiFactorProxy,
)


class WikiService:
    """Wiki service for API layer - thin wrapper around WikiFactorProxy."""

    def __init__(self, data_dir: str = ".agent"):
        self.data_dir = data_dir
        self._proxy: Optional[WikiFactorProxy] = None

    def _get_proxy(self) -> Optional[WikiFactorProxy]:
        """Get or create the WikiFactorProxy singleton."""
        if self._proxy is None:
            try:
                from QuantNodes.research.wiki import init_factor_wiki

                init_factor_wiki(self.data_dir)
                self._proxy = WikiFactorProxy()
            except Exception as e:
                print(f"Failed to initialize WikiFactorProxy: {e}")
                return None
        return self._proxy

    async def get_factors(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort: str = "updated",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get factors from wiki."""
        proxy = self._get_proxy()
        if not proxy:
            return []
        try:
            cat_enum = FactorCategory(category) if category else None
            src_enum = FactorSource(source) if source else None
            factors = proxy.list_factors(
                source=src_enum.value if src_enum else None,
                category=cat_enum.value if cat_enum else None,
                tags=tags,
                limit=limit,
            )
            return [
                {
                    "name": f.name,
                    "formula": getattr(f, "formula", ""),
                    "source": f.source.value if hasattr(f, "source") and f.source else "",
                    "category": f.category.value if hasattr(f, "category") and f.category else "",
                    "ic_mean": getattr(f, "ic_mean", None),
                    "ic_std": getattr(f, "ic_std", None),
                    "icir": getattr(f, "icir", None),
                    "rank_ic_mean": getattr(f, "rank_ic_mean", None),
                    "turnover": getattr(f, "turnover", None),
                    "tags": getattr(f, "tags", []),
                    "description": getattr(f, "description", ""),
                }
                for f in factors
            ]
        except Exception as e:
            print(f"Error getting factors: {e}")
            return []

    async def get_factor(self, name: str) -> Optional[Dict[str, Any]]:
        """Get factor by name."""
        proxy = self._get_proxy()
        if not proxy:
            return None
        try:
            factor = proxy.get_factor(name=name)
            if not factor:
                return None
            return {
                "name": factor.name,
                "formula": getattr(factor, "formula", ""),
                "source": factor.source.value if hasattr(factor, "source") and factor.source else "",
                "category": factor.category.value if hasattr(factor, "category") and factor.category else "",
                "ic_mean": getattr(factor, "ic_mean", None),
                "ic_std": getattr(factor, "ic_std", None),
                "icir": getattr(factor, "icir", None),
                "rank_ic_mean": getattr(factor, "rank_ic_mean", None),
                "turnover": getattr(factor, "turnover", None),
                "tags": getattr(factor, "tags", []),
                "description": getattr(factor, "description", ""),
            }
        except Exception as e:
            print(f"Error getting factor: {e}")
            return None

    async def create_factor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new factor."""
        proxy = self._get_proxy()
        if not proxy:
            return {"error": "WikiFactorProxy not available"}
        try:
            factor = WikiFactor(
                name=data["name"],
                formula=data.get("formula", ""),
                source=FactorSource(data.get("source", "manual")),
                category=FactorCategory(data.get("category", "other")),
                ic_mean=data.get("ic_mean"),
                ic_std=data.get("ic_std"),
                icir=data.get("icir"),
                rank_ic_mean=data.get("rank_ic_mean"),
                turnover=data.get("turnover"),
                tags=data.get("tags", []),
                description=data.get("description", ""),
            )
            proxy.store_factor(factor)
            return {"status": "created", "name": data["name"]}
        except Exception as e:
            print(f"Error creating factor: {e}")
            return {"error": str(e)}

    async def update_factor(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update factor (overwrite via store_factor)."""
        proxy = self._get_proxy()
        if not proxy:
            return {"error": "WikiFactorProxy not available"}
        try:
            existing = proxy.get_factor(name=name)
            if not existing:
                return {"error": f"Factor '{name}' not found"}
            merged = {**{"formula": "", "source": existing.source, "category": existing.category,
                          "tags": existing.tags, "description": existing.description,
                          "ic_mean": existing.ic_mean, "ic_std": existing.ic_std,
                          "icir": existing.icir, "rank_ic_mean": existing.rank_ic_mean,
                          "turnover": existing.turnover}, **data}
            factor = WikiFactor(
                name=name,
                formula=merged.get("formula", ""),
                source=FactorSource(merged["source"].value if hasattr(merged["source"], "value") else merged["source"]),
                category=FactorCategory(merged["category"].value if hasattr(merged["category"], "value") else merged["category"]),
                ic_mean=merged.get("ic_mean"),
                ic_std=merged.get("ic_std"),
                icir=merged.get("icir"),
                rank_ic_mean=merged.get("rank_ic_mean"),
                turnover=merged.get("turnover"),
                tags=merged.get("tags", []),
                description=merged.get("description", ""),
            )
            proxy.store_factor(factor)
            return {"status": "updated", "name": name}
        except Exception as e:
            print(f"Error updating factor: {e}")
            return {"error": str(e)}

    async def delete_factor(self, name: str) -> Dict[str, Any]:
        """Delete factor (no-op stub; wiki proxy has no delete)."""
        return {"status": "deleted", "name": name}

    async def get_strategies(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort: str = "updated",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get strategies from wiki."""
        proxy = self._get_proxy()
        if not proxy:
            return []
        try:
            strategies = proxy.list_strategies(
                category=category, tags=tags, limit=limit,
            )
            return [
                {
                    "name": s.name,
                    "strategy_yaml": getattr(s, "strategy_yaml", ""),
                    "description": getattr(s, "description", ""),
                    "category": getattr(s, "category", "general"),
                    "tags": getattr(s, "tags", []),
                    "backtest_result": getattr(s, "backtest_result", None),
                }
                for s in strategies
            ]
        except Exception as e:
            print(f"Error getting strategies: {e}")
            return []

    async def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get strategy by name."""
        proxy = self._get_proxy()
        if not proxy:
            return None
        try:
            s = proxy.get_strategy(name=name)
            if not s:
                return None
            return {
                "name": s.name,
                "strategy_yaml": getattr(s, "strategy_yaml", ""),
                "description": getattr(s, "description", ""),
                "category": getattr(s, "category", "general"),
                "tags": getattr(s, "tags", []),
                "backtest_result": getattr(s, "backtest_result", None),
            }
        except Exception as e:
            print(f"Error getting strategy: {e}")
            return None

    async def create_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new strategy."""
        proxy = self._get_proxy()
        if not proxy:
            return {"error": "WikiFactorProxy not available"}
        try:
            strategy = WikiStrategy(
                name=data["name"],
                strategy_yaml=data.get("strategy_yaml", ""),
                description=data.get("description", ""),
                category=data.get("category", "general"),
                tags=data.get("tags", []),
                backtest_result=data.get("backtest_result"),
            )
            proxy.store_strategy(strategy)
            return {"status": "created", "name": data["name"]}
        except Exception as e:
            print(f"Error creating strategy: {e}")
            return {"error": str(e)}

    async def update_strategy(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing strategy."""
        proxy = self._get_proxy()
        if not proxy:
            return {"error": "WikiFactorProxy not available"}
        try:
            existing = proxy.get_strategy(name=name)
            if not existing:
                return {"error": f"Strategy '{name}' not found"}
            merged = {**{
                "strategy_yaml": getattr(existing, "strategy_yaml", ""),
                "description": getattr(existing, "description", ""),
                "category": getattr(existing, "category", "general"),
                "tags": getattr(existing, "tags", []),
                "backtest_result": getattr(existing, "backtest_result", None),
            }, **data}
            strategy = WikiStrategy(
                name=name,
                strategy_yaml=merged["strategy_yaml"],
                description=merged["description"],
                category=merged["category"],
                tags=merged["tags"],
                backtest_result=merged["backtest_result"],
            )
            proxy.store_strategy(strategy)
            return {"status": "updated", "name": name}
        except Exception as e:
            print(f"Error updating strategy: {e}")
            return {"error": str(e)}

    async def delete_strategy(self, name: str) -> Dict[str, Any]:
        """Delete strategy (no-op stub)."""
        return {"status": "deleted", "name": name}

    async def search(
        self,
        query: str,
        type: str = "all",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search wiki."""
        proxy = self._get_proxy()
        if not proxy:
            return []
        try:
            results = proxy.search_factors(query=query, limit=limit)
            return [{"name": f.name, "type": "factor"} for f in results]
        except Exception as e:
            print(f"Error searching wiki: {e}")
            return []

    async def get_status(self) -> Dict[str, Any]:
        """Get wiki status."""
        proxy = self._get_proxy()
        if not proxy:
            return {"factors": 0, "strategies": 0, "logics": 0}
        try:
            status = proxy.status()
            return status if isinstance(status, dict) else {"factors": 0, "strategies": 0, "logics": 0}
        except Exception as e:
            print(f"Error getting wiki status: {e}")
            return {"factors": 0, "strategies": 0, "logics": 0}


wiki_service = WikiService()

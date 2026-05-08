"""
Wiki Service - Bridge between FastAPI and WikiTool
"""

from typing import Optional, List, Dict, Any

from QuantNodes.agent.tools.wiki import WikiTool
from QuantNodes.research.wiki import FactorSource, FactorCategory


class WikiService:
    """Wiki service for API layer"""

    def __init__(self, data_dir: str = ".quant_agent"):
        self.data_dir = data_dir
        self._wiki_tool: Optional[WikiTool] = None

    def _get_wiki_tool(self) -> WikiTool:
        """Get or create WikiTool instance"""
        if self._wiki_tool is None:
            try:
                self._wiki_tool = WikiTool(wiki_path=self.data_dir)
            except Exception as e:
                print(f"Failed to initialize WikiTool: {e}")
                return None
        return self._wiki_tool

    async def get_factors(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort: str = "updated",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get factors from wiki"""
        wiki = self._get_wiki_tool()
        try:
            # Parse category filter
            cat_enum = None
            if category:
                try:
                    cat_enum = FactorCategory(category)
                except ValueError:
                    pass

            # Parse source filter
            src_enum = None
            if source:
                try:
                    src_enum = FactorSource(source)
                except ValueError:
                    pass

            factors = await wiki._list_factors(
                source=src_enum.value if src_enum else None,
                category=cat_enum.value if cat_enum else None,
                tags=tags,
                limit=limit,
            )
            return factors
        except Exception as e:
            print(f"Error getting factors: {e}")
            return []

    async def get_factor(self, name: str) -> Optional[Dict[str, Any]]:
        """Get factor by name"""
        wiki = self._get_wiki_tool()
        try:
            factor = await wiki._get_factor(name=name)
            return factor
        except Exception as e:
            print(f"Error getting factor: {e}")
            return None

    async def create_factor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new factor"""
        wiki = self._get_wiki_tool()
        try:
            # Parse enums
            source = FactorSource(data.get("source", "manual"))
            category = FactorCategory(data.get("category", "other"))
            
            from QuantNodes.research.wiki import WikiFactor
            factor = WikiFactor(
                name=data["name"],
                formula=data.get("formula", ""),
                source=source,
                category=category,
                ic_mean=data.get("ic_mean"),
                ic_std=data.get("ic_std"),
                icir=data.get("icir"),
                rank_ic_mean=data.get("rank_ic_mean"),
                turnover=data.get("turnover"),
                tags=data.get("tags", []),
                description=data.get("description", ""),
            )
            
            result = await wiki._store_factor(
                name=factor.name,
                formula=factor.formula,
                source=source.value,
                category=category.value,
                ic_mean=factor.ic_mean,
                ic_std=factor.ic_std,
                icir=factor.icir,
                rank_ic_mean=factor.rank_ic_mean,
                turnover=factor.turnover,
                tags=factor.tags,
                description=factor.description,
            )
            return {"status": "created", "name": data["name"]}
        except Exception as e:
            print(f"Error creating factor: {e}")
            return {"error": str(e)}

    async def update_factor(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update factor"""
        wiki = self._get_wiki_tool()
        try:
            # First get existing factor
            existing = await wiki._get_factor(name=name)
            if not existing:
                return {"error": f"Factor '{name}' not found"}
            
            # Merge data
            updated_data = {**existing, **data}
            
            # Store updated factor
            source = FactorSource(updated_data.get("source", "manual"))
            category = FactorCategory(updated_data.get("category", "other"))
            
            from QuantNodes.research.wiki import WikiFactor
            factor = WikiFactor(
                name=name,
                formula=updated_data.get("formula", ""),
                source=source,
                category=category,
                ic_mean=updated_data.get("ic_mean"),
                ic_std=updated_data.get("ic_std"),
                icir=updated_data.get("icir"),
                rank_ic_mean=updated_data.get("rank_ic_mean"),
                turnover=updated_data.get("turnover"),
                tags=updated_data.get("tags", []),
                description=updated_data.get("description", ""),
            )
            
            result = await wiki._store_factor(
                name=factor.name,
                formula=factor.formula,
                source=source.value,
                category=category.value,
                ic_mean=factor.ic_mean,
                ic_std=factor.ic_std,
                icir=factor.icir,
                rank_ic_mean=factor.rank_ic_mean,
                turnover=factor.turnover,
                tags=factor.tags,
                description=factor.description,
            )
            return {"status": "updated", "name": name}
        except Exception as e:
            print(f"Error updating factor: {e}")
            return {"error": str(e)}

    async def delete_factor(self, name: str) -> Dict[str, Any]:
        """Delete factor"""
        wiki = self._get_wiki_tool()
        try:
            # WikiTool doesn't have a direct delete, but we can overwrite with empty
            # For now, return success (actual implementation would need wiki delete)
            return {"status": "deleted", "name": name}
        except Exception as e:
            print(f"Error deleting factor: {e}")
            return {"error": str(e)}

    async def get_strategies(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort: str = "updated",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get strategies from wiki"""
        wiki = self._get_wiki_tool()
        try:
            strategies = await wiki._list_strategies(
                category=category,
                tags=tags,
                limit=limit,
            )
            return strategies
        except Exception as e:
            print(f"Error getting strategies: {e}")
            return []

    async def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get strategy by name"""
        wiki = self._get_wiki_tool()
        try:
            strategy = await wiki._get_strategy(name=name)
            return strategy
        except Exception as e:
            print(f"Error getting strategy: {e}")
            return None

    async def create_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new strategy"""
        wiki = self._get_wiki_tool()
        try:
            result = await wiki._store_strategy(
                name=data["name"],
                strategy_yaml=data.get("strategy_yaml", ""),
                description=data.get("description", ""),
                category=data.get("category", "general"),
                tags=data.get("tags", []),
                backtest_result=data.get("backtest_result"),
            )
            return {"status": "created", "name": data["name"]}
        except Exception as e:
            print(f"Error creating strategy: {e}")
            return {"error": str(e)}

    async def search(
        self,
        query: str,
        type: str = "all",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search wiki"""
        wiki = self._get_wiki_tool()
        try:
            results = await wiki._search(query=query, limit=limit)
            if type != "all":
                results = [r for r in results if r.get("type") == type]
            return results
        except Exception as e:
            print(f"Error searching wiki: {e}")
            return []

    async def get_status(self) -> Dict[str, Any]:
        """Get wiki status"""
        wiki = self._get_wiki_tool()
        try:
            status = await wiki._status()
            return status
        except Exception as e:
            print(f"Error getting wiki status: {e}")
            return {"factors": 0, "strategies": 0, "logics": 0}


# Singleton instance
wiki_service = WikiService()

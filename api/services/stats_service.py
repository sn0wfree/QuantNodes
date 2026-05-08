"""
Stats Service - Aggregate data from Wiki and other sources
"""

from typing import Optional
from pathlib import Path


class StatsService:
    """Stats service for dashboard metrics"""

    def __init__(self, data_dir: str = ".quant_agent"):
        self.data_dir = data_dir
        self._wiki_tool = None

    def _get_wiki_tool(self):
        """Get or create WikiTool instance"""
        if self._wiki_tool is None:
            try:
                from QuantNodes.agent.tools.wiki import WikiTool
                self._wiki_tool = WikiTool(wiki_path=self.data_dir)
            except Exception:
                return None
        return self._wiki_tool

    async def get_stats(self) -> dict:
        """Get aggregated stats from all sources"""
        factors_count = 0
        strategies_count = 0
        
        try:
            wiki = self._get_wiki_tool()
            if wiki:
                factors = await wiki.execute(action="list_factors", limit=1000)
                factors_count = len(factors) if isinstance(factors, list) else 0
                
                strategies = await wiki.execute(action="list_strategies", limit=1000)
                strategies_count = len(strategies) if isinstance(strategies, list) else 0
        except Exception as e:
            print(f"Error fetching stats: {e}")

        return {
            "factors": factors_count,
            "strategies": strategies_count,
            "backtests": 0,  # TODO: Integrate with backtest history
            "insights": 0,  # TODO: Integrate with dream system
        }

    async def get_recent_activity(self, limit: int = 10) -> list:
        """Get recent activity from wiki"""
        try:
            wiki = self._get_wiki_tool()
            if wiki:
                factors = await wiki.execute(action="list_factors", limit=limit)
                if isinstance(factors, list):
                    return [
                        {
                            "type": "factor",
                            "name": f.get("name", ""),
                            "category": f.get("category", ""),
                            "updated_at": f.get("updated_at", ""),
                        }
                        for f in factors
                    ]
        except Exception:
            pass
        return []


# Singleton instance
stats_service = StatsService()

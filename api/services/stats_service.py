"""
Stats Service - Aggregate data from Wiki (直接调用 research.wiki，v3.0.0+)
"""


class StatsService:
    """Stats service for dashboard metrics."""

    def __init__(self, data_dir: str = ".quant_agent"):
        self.data_dir = data_dir
        self._proxy = None

    def _get_proxy(self):
        """Get or create WikiFactorProxy (skips QuantNodes.agent.tools)."""
        if self._proxy is None:
            try:
                from QuantNodes.research.wiki import init_factor_wiki, WikiFactorProxy
                init_factor_wiki(self.data_dir)
                self._proxy = WikiFactorProxy()
            except Exception:
                return None
        return self._proxy

    async def get_stats(self) -> dict:
        """Get aggregated stats from all sources"""
        factors_count = 0
        strategies_count = 0

        try:
            proxy = self._get_proxy()
            if proxy:
                factors = proxy.list_factors(limit=1000)
                factors_count = len(factors) if isinstance(factors, list) else 0

                strategies = proxy.list_strategies(limit=1000)
                strategies_count = len(strategies) if isinstance(strategies, list) else 0
        except Exception as e:
            print(f"Error fetching stats: {e}")

        return {
            "factors": factors_count,
            "strategies": strategies_count,
            "backtests": 0,
            "insights": 0,
        }

    async def get_recent_activity(self, limit: int = 10) -> list:
        """Get recent activity from wiki"""
        try:
            proxy = self._get_proxy()
            if proxy:
                factors = proxy.list_factors(limit=limit)
                if isinstance(factors, list):
                    return [
                        {
                            "type": "factor",
                            "name": getattr(f, "name", ""),
                            "category": f.category.value if getattr(f, "category", None) else "",
                            "updated_at": getattr(f, "updated_at", ""),
                        }
                        for f in factors
                    ]
        except Exception:
            pass
        return []


stats_service = StatsService()

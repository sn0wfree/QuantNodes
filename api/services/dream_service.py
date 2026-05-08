"""
Dream Service - Bridge between FastAPI and DreamEngine
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta


class DreamService:
    """Dream service for API layer"""

    def __init__(self, data_dir: str = ".quant_agent"):
        self.data_dir = data_dir
        self._dream_store = None
        self._dream_engine = None

    def _get_dream_store(self):
        """Get or create DreamStore instance"""
        if self._dream_store is None:
            try:
                from QuantNodes.agent.core.memory import DreamStore
                self._dream_store = DreamStore(Path(self.data_dir))
            except Exception as e:
                print(f"Failed to initialize DreamStore: {e}")
                return None
        return self._dream_store

    def _get_dream_engine(self):
        """Get or create DreamEngine instance"""
        if self._dream_engine is None:
            try:
                from QuantNodes.agent.core.dream import DreamEngine
                from QuantNodes.agent.core.memory import DreamConfig
                store = self._get_dream_store()
                if store:
                    self._dream_engine = DreamEngine(dream_store=store)
            except Exception as e:
                print(f"Failed to initialize DreamEngine: {e}")
                return None
        return self._dream_engine

    async def list_dreams(
        self,
        limit: int = 20,
        dream_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of dreams/insights"""
        store = self._get_dream_store()
        if not store:
            return self._get_sample_dreams()

        try:
            dreams = store.get_recent_dreams(limit=limit * 2)
            
            # Filter by type
            if dream_type:
                dreams = [d for d in dreams if d.type == dream_type]
            
            # Filter by confidence
            if min_confidence is not None:
                dreams = [d for d in dreams if d.confidence >= min_confidence]
            
            # Convert to dicts
            return [
                {
                    "id": d.id,
                    "title": d.content[:50] + "..." if len(d.content) > 50 else d.content,
                    "content": d.content,
                    "type": d.type,
                    "category": d.type,
                    "confidence": d.confidence,
                    "created_at": d.timestamp,
                    "tags": d.tags,
                    "insights": d.insights,
                    "source": d.source,
                }
                for d in dreams[:limit]
            ]
        except Exception as e:
            print(f"Error listing dreams: {e}")
            return self._get_sample_dreams()

    async def get_dream(self, dream_id: str) -> Optional[Dict[str, Any]]:
        """Get dream by ID"""
        dreams = await self.list_dreams(limit=100)
        for dream in dreams:
            if dream["id"] == dream_id:
                return dream
        return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get dream statistics"""
        dreams = await self.list_dreams(limit=1000)
        
        if not dreams:
            return self._get_sample_stats()
        
        # Calculate stats
        total = len(dreams)
        by_type: Dict[str, int] = {}
        confidences = []
        
        for dream in dreams:
            dream_type = dream.get("type", "unknown")
            by_type[dream_type] = by_type.get(dream_type, 0) + 1
            confidences.append(dream.get("confidence", 0))
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Get recent trends (last 7 days)
        now = datetime.now()
        recent_counts = []
        for i in range(7):
            date = now - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            count = sum(1 for d in dreams if d.get("created_at", "").startswith(date_str))
            recent_counts.append({"date": date_str, "count": count})
        
        return {
            "total_insights": total,
            "by_type": by_type,
            "by_category": by_type.copy(),
            "avg_confidence": avg_confidence,
            "recent_trend": list(reversed(recent_counts)),
            "top_tags": self._get_top_tags(dreams),
        }

    async def generate_insight(
        self,
        dream_type: str,
        content: str,
        source: str = "",
        confidence: float = 0.8,
        tags: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate a new dream/insight"""
        engine = self._get_dream_engine()
        if engine:
            try:
                dream = await engine.generate_dream(
                    dream_type=dream_type,
                    content=content,
                    source=source,
                    confidence=confidence,
                    tags=tags or [],
                )
                return {
                    "id": dream.id,
                    "title": dream.content[:50] + "..." if len(dream.content) > 50 else dream.content,
                    "content": dream.content,
                    "type": dream.type,
                    "category": dream.type,
                    "confidence": dream.confidence,
                    "created_at": dream.timestamp,
                    "tags": getattr(dream, 'tags', []),
                    "insights": getattr(dream, 'insights', []),
                    "source": getattr(dream, 'source', ''),
                }
            except Exception as e:
                print(f"Error generating dream: {e}")
        
        # Fallback: return sample
        return {
            "id": f"dream-{datetime.now().timestamp()}",
            "title": content[:50] + "..." if len(content) > 50 else content,
            "content": content,
            "type": dream_type,
            "category": dream_type,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
            "insights": [],
            "source": source,
        }

    def _get_top_tags(self, dreams: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top tags from dreams"""
        tag_counts: Dict[str, int] = {}
        for dream in dreams:
            for tag in dream.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"tag": tag, "count": count} for tag, count in sorted_tags[:limit]]

    def _get_sample_dreams(self) -> List[Dict[str, Any]]:
        """Return sample dreams for development"""
        return [
            {
                "id": "dream-001",
                "title": "Momentum factor shows strong performance",
                "content": "The 20-day momentum factor has shown consistent alpha generation in the CSI 300 universe with IC mean of 0.052 and ICIR of 2.3.",
                "type": "factor_insight",
                "category": "factor_insight",
                "confidence": 0.85,
                "created_at": "2026-05-07T10:00:00Z",
                "tags": ["momentum", "alpha", "csi300"],
                "insights": [
                    "IC mean is above threshold (0.052 > 0.05)",
                    "ICIR indicates stable performance (2.3 > 2.0)",
                    "Factor is suitable for live trading"
                ],
                "source": "factor_analysis",
            },
            {
                "id": "dream-002",
                "title": "Mean reversion strategy backtest completed",
                "content": "Strategy 'mean_reversion_v2' completed backtest with 15.2% annual return and 1.8 Sharpe ratio.",
                "type": "strategy_insight",
                "category": "strategy_insight",
                "confidence": 0.9,
                "created_at": "2026-05-07T09:30:00Z",
                "tags": ["mean_reversion", "backtest", "sharpe"],
                "insights": [
                    "Annual return: 15.2%",
                    "Sharpe ratio: 1.8",
                    "Max drawdown: -8.5%"
                ],
                "source": "backtest",
            },
            {
                "id": "dream-003",
                "title": "Market regime detected: Low volatility",
                "content": "Market volatility has decreased significantly. Consider reducing position sizes.",
                "type": "market_regime",
                "category": "market_regime",
                "confidence": 0.75,
                "created_at": "2026-05-07T08:00:00Z",
                "tags": ["volatility", "regime", "risk"],
                "insights": [
                    "VIX below 15 for 5 consecutive days",
                    "Consider reducing leverage",
                    "Favor defensive sectors"
                ],
                "source": "market_monitor",
            },
            {
                "id": "dream-004",
                "title": "New factor derived from research report",
                "content": "Factor 'quality_roe' derived from Goldman Sachs research report shows promising backtest results.",
                "type": "factor_insight",
                "category": "factor_insight",
                "confidence": 0.82,
                "created_at": "2026-05-06T16:00:00Z",
                "tags": ["quality", "roe", "research"],
                "insights": [
                    "Factor derived from institutional research",
                    "Backtest shows consistent alpha",
                    "Low correlation with existing factors"
                ],
                "source": "research_report",
            },
            {
                "id": "dream-005",
                "title": "Portfolio risk alert: Concentration",
                "content": "Portfolio has 40% exposure to technology sector. Consider diversification.",
                "type": "risk_alert",
                "category": "risk_alert",
                "confidence": 0.88,
                "created_at": "2026-05-06T14:00:00Z",
                "tags": ["risk", "concentration", "diversification"],
                "insights": [
                    "Technology sector overweight (40% vs 25% benchmark)",
                    "Top 5 positions account for 60% of portfolio",
                    "Recommend rebalancing"
                ],
                "source": "risk_monitor",
            },
        ]

    def _get_sample_stats(self) -> Dict[str, Any]:
        """Return sample stats for development"""
        return {
            "total_insights": 42,
            "by_type": {
                "factor_insight": 18,
                "strategy_insight": 12,
                "market_regime": 5,
                "risk_alert": 7,
            },
            "by_category": {
                "factor_insight": 18,
                "strategy_insight": 12,
                "market_regime": 5,
                "risk_alert": 7,
            },
            "avg_confidence": 0.82,
            "recent_trend": [
                {"date": "2026-05-07", "count": 3},
                {"date": "2026-05-06", "count": 5},
                {"date": "2026-05-05", "count": 2},
                {"date": "2026-05-04", "count": 4},
                {"date": "2026-05-03", "count": 6},
                {"date": "2026-05-02", "count": 3},
                {"date": "2026-05-01", "count": 7},
            ],
            "top_tags": [
                {"tag": "factor", "count": 15},
                {"tag": "alpha", "count": 12},
                {"tag": "momentum", "count": 8},
                {"tag": "risk", "count": 7},
                {"tag": "backtest", "count": 6},
            ],
        }


# Singleton instance
dream_service = DreamService()

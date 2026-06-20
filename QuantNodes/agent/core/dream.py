# coding=utf-8
"""
Dream Engine - Async Insight Generation + Skill Dispatch

Phase 4.2: Dream System
Phase 4.4: Skill Dispatch Integration
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from .memory import Dream, DreamConfig, DreamStore

logger = logging.getLogger(__name__)


class DreamEngine:
    """Dream Engine - Async Insight Generation"""

    def __init__(
        self, dream_store: DreamStore, config: DreamConfig = None
    ):
        self.dream_store = dream_store
        self.config = config or DreamConfig()
        self._pending_dreams: Dict[str, Dream] = {}
        self._running: bool = False
        self._subscribers: List[Callable] = []

    async def generate_dream(
        self,
        dream_type: str,
        content: str,
        source: str = "",
        insights: List[str] = None,
        confidence: float = 0.8,
        tags: List[str] = None,
    ) -> Dream:
        """Generate a dream entry"""
        dream = Dream(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            type=dream_type,
            content=content,
            insights=insights or [],
            source=source,
            confidence=confidence,
            tags=tags or [],
        )
        self.dream_store.save_dream(dream)
        for subscriber in self._subscribers:
            try:
                await subscriber(dream)
            except Exception:
                pass
        return dream

    async def process_wiki_update(
        self, wiki_proxy, page_name: str
    ) -> Optional[Dream]:
        """Process Wiki update and generate insight dream"""
        try:
            page_data = wiki_proxy.wiki.read_page(page_name)
            content = page_data.get("content", "")[:500]
            insights = []
            if "IC" in content:
                insights.append("检测到 IC 相关内容，可能存在有效因子")
            if "回测" in content:
                insights.append("检测到回测记录，策略已验证")
            if "研报" in content:
                insights.append("来源为研报，具备研究价值")
            if insights:
                return await self.generate_dream(
                    dream_type="wiki_insight",
                    content=f"Wiki 页面更新: {page_name}",
                    source="wiki",
                    insights=insights,
                    confidence=0.85,
                    tags=["wiki", "insight"],
                )
        except Exception:
            pass
        return None

    def should_analyze_conversation(
        self, user_message: str, assistant_response: str
    ) -> bool:
        """快速检查对话是否可能包含值得提取的洞察"""
        combined = user_message + assistant_response
        return any(kw in combined for kw in self.config.analysis_keywords)

    async def analyze_conversation(
        self,
        user_message: str,
        assistant_response: str,
        tools_used: List[str] = None,
    ) -> Optional[Dream]:
        """分析对话并生成洞察 Dream（仅在包含新洞察时生成）"""
        insights = []
        confidence = 0.6

        if any(kw in user_message for kw in ["IC", "ICIR", "因子", "factor"]):
            factor_keywords = ["IC均值", "ICIR", "因子有效", "因子无效", "IC"]
            if any(kw in assistant_response for kw in factor_keywords):
                insights.append("对话涉及因子分析")
                confidence += 0.1

        if any(kw in user_message for kw in ["回测", "策略", "收益", "夏普"]):
            if any(kw in assistant_response for kw in ["年化", "夏普", "回撤", "胜率"]):
                insights.append("对话涉及策略回测")
                confidence += 0.1

        if any(kw in user_message for kw in ["记住", "以后", "每次", "偏好"]):
            insights.append("用户表达了偏好或要求记忆")
            confidence += 0.15

        if not insights:
            return None

        confidence = min(max(confidence, 0.3), 1.0)
        return await self.generate_dream(
            dream_type="conversation_insight",
            content=f"对话摘要: {user_message[:100]}",
            source="conversation",
            insights=insights,
            confidence=confidence,
            tags=["conversation", "auto"],
        )

    async def analyze_factor(
        self, factor_name: str, ic_data: Dict[str, Any]
    ) -> Dream:
        """Analyze factor and generate insight"""
        insights = []
        confidence = 0.7
        ic_mean = ic_data.get("ic_mean", 0)
        icir = ic_data.get("icir", 0)
        if ic_mean > 0.05:
            insights.append(f"因子 {factor_name} IC均值较高: {ic_mean:.4f}")
            confidence += 0.1
        if icir > 0.5:
            insights.append(f"因子 {factor_name} ICIR表现良好: {icir:.4f}")
            confidence += 0.1
        if ic_mean < 0:
            insights.append(f"警告: 因子 {factor_name} IC均值为负")
            confidence -= 0.2
        confidence = min(max(confidence, 0.3), 1.0)
        return await self.generate_dream(
            dream_type="factor_insight",
            content=f"因子分析: {factor_name}",
            source="factor_analysis",
            insights=insights,
            confidence=confidence,
            tags=["factor", "analysis", factor_name],
        )

    async def analyze_strategy(
        self, strategy_name: str, performance: Dict[str, Any]
    ) -> Dream:
        """Analyze strategy and generate insight"""
        insights = []
        confidence = 0.7
        returns = performance.get("annual_return", 0)
        sharpe = performance.get("sharpe_ratio", 0)
        max_drawdown = performance.get("max_drawdown", 0)
        if returns > 0.15:
            insights.append(f"策略 {strategy_name} 年化收益较高: {returns:.2%}")
            confidence += 0.1
        if sharpe > 1.5:
            insights.append(f"策略 {strategy_name} 夏普比率优秀: {sharpe:.2f}")
            confidence += 0.1
        if max_drawdown < -0.2:
            insights.append(f"警告: 策略 {strategy_name} 回撤较大: {max_drawdown:.2%}")
            confidence -= 0.15
        confidence = min(max(confidence, 0.3), 1.0)
        return await self.generate_dream(
            dream_type="strategy_insight",
            content=f"策略分析: {strategy_name}",
            source="strategy_analysis",
            insights=insights,
            confidence=confidence,
            tags=["strategy", "analysis", strategy_name],
        )

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to new dreams"""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from dreams"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def start_auto_inject(self, interval_hours: int = 24) -> None:
        """Start auto injection to memory"""
        self._running = True
        while self._running:
            self.dream_store.inject_to_memory(self.config)
            await asyncio.sleep(interval_hours * 3600)

    def stop_auto_inject(self) -> None:
        """Stop auto injection"""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get dream statistics"""
        dreams = self.dream_store.get_recent_dreams(limit=1000)
        by_type: Dict[str, int] = {}
        for dream in dreams:
            by_type[dream.type] = by_type.get(dream.type, 0) + 1
        return {
            "total_dreams": len(dreams),
            "by_type": by_type,
            "config": {
                "max_dreams_per_day": self.config.max_dreams_per_day,
                "min_confidence": self.config.min_confidence,
                "auto_inject": self.config.auto_inject,
            },
        }

    async def dispatch_skills(
        self, query: str, skill_registry=None
    ) -> List:
        """根据查询匹配并执行所有技能，返回 SkillResult 列表"""
        from ..skills.registry import SkillRegistry

        registry = skill_registry or SkillRegistry()
        skills = registry.list_all()
        if not skills:
            return []

        results = []
        for skill in skills:
            try:
                context = {"query": query}
                result = await skill.execute(context)
                results.append(result)
            except Exception as e:
                logger.error("Skill %s failed in dispatch: %s", skill.name, e)
                from ..skills.base import SkillResult
                results.append(
                    SkillResult(success=False, error=f"{skill.name}: {str(e)}")
                )
        return results

    def push_to_agent(self, dream: Dream) -> None:
        """将洞察注入 DreamStore（影响后续 Agent 回复）"""
        self.dream_store.save_dream(dream)
        for subscriber in self._subscribers:
            try:
                asyncio.get_event_loop().create_task(subscriber(dream))
            except RuntimeError:
                pass

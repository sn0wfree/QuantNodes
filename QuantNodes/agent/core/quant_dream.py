# coding=utf-8
"""QuantDreamHook - 量化专属 Dream 钩子（v3.0.0 新增）。

挂在 HKUDS nanobot 的 AgentHook 系统上，扩展通用 Dream 整合（上游
nanobot/agent/memory.py::Dream + Consolidator）以覆盖量化场景：

1. **因子洞察** - 提取 IC 表现好的因子，反思构造逻辑
2. **回测模式** - 记录过拟合/未来函数/手续费过高等问题模式
3. **策略启发** - 跨策略对比，识别共同成功因素
4. **风险事件** - 大回撤/极端行情下的策略表现
5. **代码模式** - LLM 生成的常见代码 bug

输出：``.agent/memory/topic-quant-dream.md``

向后兼容：
- ``QuantNodes.agent.core.dream.DreamEngine`` 仍可用（re-export 自本模块）
- 旧 ``QuantNodes.agent.core.dream.DreamStore`` 由 nanobot upstream 替代
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


FACTOR_KEYWORDS = (
    "alpha", "factor", "ic", "rank", "momentum", "reversal",
    "value", "quality", "volatility",
)
BACKTEST_KEYWORDS = (
    "backtest", "sharpe", "drawdown", "annualized", "win_rate",
    "profit_loss_ratio", "icir",
)
STRATEGY_KEYWORDS = (
    "strategy", "pipeline", "weight", "rebalance", "portfolio",
)


@dataclass
class QuantDreamInsight:
    """Single quant dream entry."""

    type: str
    content: str
    insights: List[str] = field(default_factory=list)
    confidence: float = 0.7
    tags: List[str] = field(default_factory=list)
    timestamp: str = ""
    source: str = "quant_dream"

    def to_markdown(self) -> str:
        lines = [f"### {self.timestamp[:10] or 'unknown'} - {self.type}", f"- {self.content}"]
        for insight in self.insights:
            lines.append(f"  - {insight}")
        if self.tags:
            lines.append(f"  - tags: {', '.join(self.tags)}")
        return "\n".join(lines) + "\n"


class QuantDreamHook:
    """QuantNodes 专属 Dream 钩子，注入到 nanobot 的 _extra_hooks 列表。

    Usage::

        from QuantNodes.agent.core.quant_dream import QuantDreamHook
        from QuantNodes.agent import Agent
        agent = Agent(workspace=".agent")
        agent.loop._extra_hooks.append(QuantDreamHook(agent.workspace))
    """

    DEFAULT_INTERVAL = 10
    DEFAULT_MIN_CONFIDENCE = 0.5

    def __init__(
        self,
        workspace: Path,
        interval: int = DEFAULT_INTERVAL,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.topic_file = self.memory_dir / "topic-quant-dream.md"
        self.interval = interval
        self.min_confidence = min_confidence
        self._counter: Dict[str, int] = {}

    def should_analyze(self, user_content: str, assistant_content: str) -> bool:
        """Return True iff the conversation touches any quant topic."""
        combined = (user_content + " " + assistant_content).lower()
        return any(
            kw in combined
            for kw in FACTOR_KEYWORDS + BACKTEST_KEYWORDS + STRATEGY_KEYWORDS
        )

    def analyze_session(
        self,
        session_key: str,
        user_content: str,
        assistant_content: str,
    ) -> Optional[QuantDreamInsight]:
        """Run quant-specific analysis on a completed session turn.

        Lightweight heuristic — no LLM call. Returns ``None`` if no insight.
        """
        if not self.should_analyze(user_content, assistant_content):
            return None

        kind = self._classify(user_content + " " + assistant_content)
        content, insights = self._extract_insights(user_content, assistant_content, kind)

        if not insights:
            return None

        from datetime import datetime, timezone
        return QuantDreamInsight(
            type=kind,
            content=content,
            insights=insights,
            confidence=0.7,
            tags=[kind, "auto"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="quant_dream",
        )

    def append(self, dream: QuantDreamInsight) -> None:
        """Append an insight to the topic file (idempotent append)."""
        if dream.confidence < self.min_confidence:
            return
        existing = self.topic_file.read_text(encoding="utf-8") if self.topic_file.exists() else "# Quant Dream Insights\n\n"
        self.topic_file.write_text(existing + dream.to_markdown(), encoding="utf-8")
        logger.info("Quant dream appended: type=%s", dream.type)

    def tick(self, session_key: str) -> None:
        """Increment session counter; returns whether to fire analysis this round."""
        n = self._counter.get(session_key, 0) + 1
        self._counter[session_key] = n
        return None  # analysis is triggered externally via analyze_session()

    @staticmethod
    def _classify(text: str) -> str:
        lc = text.lower()
        if any(kw in lc for kw in FACTOR_KEYWORDS):
            return "factor_insight"
        if any(kw in lc for kw in BACKTEST_KEYWORDS):
            return "backtest_pattern"
        if any(kw in lc for kw in STRATEGY_KEYWORDS):
            return "strategy_heuristic"
        return "general"

    @staticmethod
    def _extract_insights(user: str, assistant: str, kind: str) -> tuple:
        first_sentence = re.split(r"[.!?\n]", assistant.strip(), maxsplit=1)[0][:200]
        return (first_sentence or f"{kind} analysis", [assistant[:300]])


class DreamEngine:
    """向后兼容 shim — 旧 ``QuantNodes.agent.core.dream.DreamEngine`` API。

    委托到 QuantDreamHook。保留名称以避免破坏 api/services/dream_service.py。
    """

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)
        self.hook = QuantDreamHook(self.workspace)

    def analyze_conversation(self, user_message: str, assistant_response: str) -> Optional[QuantDreamInsight]:
        return self.hook.analyze_session("default", user_message, assistant_response)

    def generate_dream(self, dream_type: str, content: str, source: str, insights, confidence: float, tags):
        from datetime import datetime, timezone
        dream = QuantDreamInsight(
            type=dream_type, content=content, insights=list(insights),
            confidence=confidence, tags=list(tags), source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.hook.append(dream)
        return dream


__all__ = [
    "QuantDreamHook",
    "QuantDreamInsight",
    "DreamEngine",
    "FACTOR_KEYWORDS",
    "BACKTEST_KEYWORDS",
    "STRATEGY_KEYWORDS",
]

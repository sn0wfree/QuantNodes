"""LLMJudge — LLM 一致性评判 (Week 1 mock 实现)。

Mock 模式: 简单启发式检查 (hypothesis + expression 长度相关)
真实模式 (后续): 调用 deepseek-v3 / Claude / GPT-4o
"""
from __future__ import annotations

import json
from typing import Optional

from .dataclass import ChannelFeedback, FeedbackChannel


class LLMJudge:
    """LLM 一致性评判器 — hypothesis ↔ description ↔ expression。

    Args:
        model: 模型名 (mock/real)
        max_correction_attempts: 解析失败时重试次数
        llm_callable: 真实 LLM 调用函数, 接受 prompt 返回 string
    """

    def __init__(
        self,
        model: str = "mock",
        max_correction_attempts: int = 3,
        llm_callable: Optional[callable] = None,
    ):
        self.model = model
        self.max_correction_attempts = max_correction_attempts
        self._llm_callable = llm_callable

    def judge(
        self,
        hypothesis: str,
        description: str,
        expression: str,
    ) -> ChannelFeedback:
        """评判三者一致性。"""
        prompt = self._build_prompt(hypothesis, description, expression)
        for attempt in range(self.max_correction_attempts + 1):
            try:
                raw = self._call(prompt)
                result = json.loads(raw)
                return ChannelFeedback(
                    channel=FeedbackChannel.LLM,
                    passed=bool(result["consistent"]),
                    detail=str(result.get("reason", "")),
                    score=float(result.get("score", 1.0 if result["consistent"] else 0.0)),
                    metadata={"model": self.model, "attempt": attempt + 1},
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                if attempt == self.max_correction_attempts:
                    return ChannelFeedback(
                        channel=FeedbackChannel.LLM,
                        passed=False,
                        detail=f"LLM 解析失败: {e}",
                        score=0.0,
                        metadata={"model": self.model, "attempt": attempt + 1},
                    )
                continue

    def _build_prompt(self, h: str, d: str, e: str) -> str:
        return (
            "判断以下三者是否逻辑一致:\n"
            f"Hypothesis (研究假设): {h}\n"
            f"Description (因子描述): {d}\n"
            f"Expression (代码表达式): {e}\n\n"
            '返回 JSON: {"consistent": true/false, "reason": "理由", "score": 0-1}'
        )

    def _call(self, prompt: str) -> str:
        """调用 LLM (mock 或真实)。"""
        if self._llm_callable is not None:
            return self._llm_callable(prompt)
        if self.model == "mock":
            return self._mock_call(prompt)
        raise NotImplementedError(
            "真实 LLM 调用未实现, 请提供 llm_callable 或使用 model='mock'"
        )

    @staticmethod
    def _mock_call(prompt: str) -> str:
        """Mock 实现: 启发式检查。

        规则:
            - hypothesis 和 description 都为空 -> 不一致
            - expression 为空 -> 不一致
            - 出现 'momentum' / '反转' 关键词 + 表达式含 'returns' -> 一致
            - 其他: 简单长度匹配
        """
        h = _extract_field(prompt, "Hypothesis")
        d = _extract_field(prompt, "Description")
        e = _extract_field(prompt, "Expression")

        if not e:
            return json.dumps({"consistent": False, "reason": "表达式为空", "score": 0.0})
        if not h and not d:
            return json.dumps({
                "consistent": False,
                "reason": "hypothesis 和 description 都为空",
                "score": 0.0,
            })

        keywords_h = {"momentum", "反转", "反转", "波动", "volume", "量价", "动量"}
        text = (h + " " + d).lower()
        matched = sum(1 for kw in keywords_h if kw.lower() in text)
        if matched > 0 and ("returns" in e or "close" in e or "open" in e):
            return json.dumps({
                "consistent": True,
                "reason": "关键词匹配, 假设与表达式使用相关字段",
                "score": 0.85,
            })
        return json.dumps({
            "consistent": True,
            "reason": "mock 默认通过 (无法判定)",
            "score": 0.7,
        })


def _extract_field(prompt: str, field_name: str) -> str:
    """Extract a single field value from a prompt.

    Format: '<FieldName> ... : <value>\\n<next line>\\n...'.
    Captures the rest of the line after ':' (may be empty).
    """
    lines = prompt.split("\n")
    for line in lines:
        if line.startswith(field_name):
            if ":" not in line:
                return ""
            return line.split(":", 1)[1].strip()
    return ""

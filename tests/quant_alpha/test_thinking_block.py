# coding=utf-8
"""
test_thinking_block.py - parse_thinking_block() 单元测试

测试覆盖:
- 空输入
- 简单结构化 thinking
- 多行 thinking
- SUGGESTED_OPS 列表解析
- 算子提及提取
- 缺失字段的 fallback
- 真实 LLM 输出样例
"""
from __future__ import annotations

import pytest

from QuantNodes.research.quant_alpha.llm.parser import (
    ThinkingRecord,
    parse_thinking_block,
)


class TestParseThinkingBlockBasic:
    """基础场景"""

    def test_empty_string(self):
        result = parse_thinking_block("")
        assert isinstance(result, ThinkingRecord)
        assert result.raw == ""
        assert result.hypothesis == ""
        assert result.mechanism == ""
        assert result.mentioned_ops == []
        assert result.suggested_ops == []

    def test_none_input(self):
        result = parse_thinking_block(None)
        assert result.raw == ""
        assert result.hypothesis == ""

    def test_no_structured_fields(self):
        text = "The user wants me to read a file. Let me think about this."
        result = parse_thinking_block(text)
        assert result.raw == text
        assert result.hypothesis == ""
        assert result.mentioned_ops == []


class TestParseThinkingBlockStructured:
    """结构化字段解析"""

    def test_simple_hypothesis(self):
        text = "HYPOTHESIS: A-share retail overreaction creates reversal opportunity"
        result = parse_thinking_block(text)
        assert result.hypothesis == "A-share retail overreaction creates reversal opportunity"

    def test_all_fields(self):
        text = """HYPOTHESIS: 20-day reversal factor works in A-shares
MECHANISM: Retail overreaction + T+1 settlement amplifies mean reversion
OPERATOR_RATIONALE: rank gives cross-sectional comparison, ts_mean smooths noise
PARAMETER_RATIONALE: 20 days is classic lookback from academic literature
RISK: Regime change in liquidity could break the pattern
SUGGESTED_OPS: rank,ts_mean,sub"""
        result = parse_thinking_block(text)
        assert "20-day reversal" in result.hypothesis
        assert "Retail overreaction" in result.mechanism
        assert "rank" in result.operator_rationale.lower()
        assert "20 days" in result.parameter_rationale
        assert "Regime change" in result.risk
        assert result.suggested_ops == ["rank", "ts_mean", "sub"]

    def test_multiline_field_value(self):
        text = """HYPOTHESIS: Multi-line hypothesis
that spans multiple lines
MECHANISM: Some mechanism"""
        result = parse_thinking_block(text)
        assert "Multi-line hypothesis" in result.hypothesis
        assert "multiple lines" in result.hypothesis
        assert result.mechanism == "Some mechanism"

    def test_partial_fields(self):
        text = "HYPOTHESIS: Only hypothesis, no other fields"
        result = parse_thinking_block(text)
        assert result.hypothesis == "Only hypothesis, no other fields"
        assert result.mechanism == ""
        assert result.suggested_ops == []


class TestParseThinkingBlockOperators:
    """算子提及提取"""

    def test_mentioned_ops_with_vocab(self):
        text = "I will use rank and ts_mean with div for normalization."
        vocab = {"rank", "ts_mean", "ts_std", "div", "mul"}
        result = parse_thinking_block(text, op_vocab=vocab)
        assert "rank" in result.mentioned_ops
        assert "ts_mean" in result.mentioned_ops
        assert "div" in result.mentioned_ops
        assert "mul" not in result.mentioned_ops  # not in text

    def test_mentioned_ops_without_vocab(self):
        text = "rank(ts_mean(close, 20))"
        result = parse_thinking_block(text)  # no vocab
        assert result.mentioned_ops == []  # empty without vocab

    def test_mentioned_ops_with_vocab_excludes_non_ops(self):
        """Only operators in vocab are mentioned_ops, not arbitrary words"""
        text = "I will use rank and ts_mean with div for normalization."
        vocab = {"rank", "ts_mean", "ts_std", "div", "mul"}
        result = parse_thinking_block(text, op_vocab=vocab)
        assert "rank" in result.mentioned_ops
        assert "ts_mean" in result.mentioned_ops
        assert "div" in result.mentioned_ops
        # 'use', 'and', 'with', 'for' 等普通词不在 vocab 中
        assert "use" not in result.mentioned_ops
        assert "and" not in result.mentioned_ops

    def test_mentioned_ops_filters_python_keywords(self):
        """vocab 不应包含 Python 关键字 — 这里 vocab 只有真实算子"""
        text = "I will use rank with if condition and return values"
        vocab = {"rank", "ts_mean"}  # 不含 if/return
        result = parse_thinking_block(text, op_vocab=vocab)
        assert "rank" in result.mentioned_ops
        assert "if" not in result.mentioned_ops
        assert "return" not in result.mentioned_ops

    def test_suggested_ops_parsing(self):
        text = "SUGGESTED_OPS: rank, ts_mean, div,mul"
        result = parse_thinking_block(text)
        assert result.suggested_ops == ["rank", "ts_mean", "div", "mul"]

    def test_suggested_ops_with_spaces(self):
        text = "SUGGESTED_OPS:  rank ,  ts_mean  ,div  "
        result = parse_thinking_block(text)
        assert result.suggested_ops == ["rank", "ts_mean", "div"]


class TestParseThinkingBlockRealistic:
    """真实 LLM 输出样例"""

    def test_realistic_thinking_block(self):
        """模拟 LLM 真实输出（MiniMax M3）"""
        text = """The user wants me to read the agent file and generate alpha ideas. Let me think about A-share reversal patterns.

HYPOTHESIS: Stocks with extreme 20-day declines rebound due to retail overreaction in A-share market
MECHANISM: T+1 settlement + retail dominance (60% of volume) amplifies overreaction; mean reversion follows
OPERATOR_RATIONALE: rank provides cross-sectional comparison; ts_mean smooths daily noise; sub captures difference
PARAMETER_RATIONALE: 20 days matches academic literature (Jegadeesh 1990); 5 days for short-term
RISK: Liquidity regime change; ST stock filter needed
SUGGESTED_OPS: rank,ts_mean,ts_std,sub

Now let me output the JSON..."""
        result = parse_thinking_block(text, op_vocab={"rank", "ts_mean", "ts_std", "sub", "div"})
        assert "20-day declines" in result.hypothesis
        assert "T+1" in result.mechanism
        assert "Jegadeesh" in result.parameter_rationale
        assert "Liquidity regime" in result.risk
        assert result.suggested_ops == ["rank", "ts_mean", "ts_std", "sub"]
        # mentioned_ops: extract all ops with parens, filter by vocab
        assert "rank" in result.mentioned_ops
        assert "ts_mean" in result.mentioned_ops

    def test_no_thinking_at_all(self):
        """LLM 没输出 thinking 块"""
        result = parse_thinking_block(None, op_vocab={"rank"})
        assert result.raw == ""
        assert result.hypothesis == ""


class TestThinkingRecordDataclass:
    """ThinkingRecord dataclass 行为"""

    def test_default_construction(self):
        r = ThinkingRecord()
        assert r.raw == ""
        assert r.hypothesis == ""
        assert r.mentioned_ops == []

    def test_to_dict(self):
        r = ThinkingRecord(
            raw="...",
            hypothesis="h",
            mentioned_ops=["rank", "ts_mean"],
        )
        d = r.to_dict()
        assert d["raw"] == "..."
        assert d["hypothesis"] == "h"
        assert d["mentioned_ops"] == ["rank", "ts_mean"]
        assert "suggested_ops" in d


class TestThinkingBlockIntegration:
    """与 alpha_gpt workflow 集成测试（mock LLM）"""

    def test_idea_record_includes_thinking(self):
        """IdeaRecord 应能容纳 thinking 字段"""
        from QuantNodes.research.quant_alpha.workflow.state import IdeaRecord
        rec = IdeaRecord(
            id="IDEA-1-1",
            name="20日反转",
            category="reversal",
            description="20日跌幅最大反弹",
            thinking="HYPOTHESIS: overreaction",
            hypothesis="overreaction",
            mechanism="T+1",
            mentioned_ops=["rank", "ts_mean"],
        )
        assert rec.thinking == "HYPOTHESIS: overreaction"
        assert rec.hypothesis == "overreaction"
        assert rec.mechanism == "T+1"
        assert rec.mentioned_ops == ["rank", "ts_mean"]
        # to_dict 应包含新字段
        d = rec.to_dict()
        assert d["hypothesis"] == "overreaction"
        assert d["mentioned_ops"] == ["rank", "ts_mean"]

    def test_formula_record_includes_thinking(self):
        """FormulaRecord 应能容纳 thinking 字段"""
        from QuantNodes.research.quant_alpha.workflow.state import FormulaRecord
        rec = FormulaRecord(
            formula_id="F-1-1",
            idea_id="IDEA-1-1",
            formula="rank(ts_mean(close, 20))",
            round_discovered=1,
            thinking="...",
            hypothesis="reversal",
            mentioned_ops=["rank", "ts_mean"],
        )
        assert rec.hypothesis == "reversal"
        d = rec.to_dict()
        assert d["hypothesis"] == "reversal"
        assert d["mentioned_ops"] == ["rank", "ts_mean"]

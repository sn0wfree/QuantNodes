# coding=utf-8
"""
operator_lookup.py - 算子查询工具

让 agent 动态发现 OperatorVocab 中的 162 个可用算子，
获取算子签名、参数、示例，以及校验公式有效性。

用法 (agent 调用)::

    # 列出所有算子
    operator_lookup(action="list_operators")
    operator_lookup(action="list_operators", category="time")

    # 获取算子详情
    operator_lookup(action="get_operator_info", name="ts_mean")

    # 校验公式
    operator_lookup(action="validate_formula", formula="rank(ts_mean(close, 20))")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from .base import Tool

logger = logging.getLogger(__name__)

__all__ = ["OperatorLookupTool"]


class OperatorLookupTool(Tool):
    """算子查询工具 — 让 agent 发现可用算子

    3 个 action:
    - list_operators: 列出所有算子（可按类别过滤）
    - get_operator_info: 获取单个算子详情（签名、参数、示例）
    - validate_formula: 校验公式是否有效
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "operator_lookup"

    @property
    def description(self) -> str:
        return (
            "查询 QuantNodes OperatorVocab 中的可用算子（162 个）。"
            "用于发现算子、获取用法说明、校验公式有效性。"
            "生成 alpha 因子公式前，应先调用此工具获取可用算子列表。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_operators",
                        "get_operator_info",
                        "validate_formula",
                    ],
                    "description": (
                        "list_operators: 列出所有算子（可按 category 过滤）。"
                        "get_operator_info: 获取单个算子详情。"
                        "validate_formula: 校验公式是否有效。"
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": ["time", "point", "section", "multi_section"],
                    "description": (
                        "算子类别过滤（list_operators 时使用）。"
                        "time=时间序列, point=逐点, section=截面, multi_section=多截面"
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "算子名（get_operator_info 时使用）",
                },
                "formula": {
                    "type": "string",
                    "description": "公式字符串（validate_formula 时使用）",
                },
            },
            "required": ["action"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        action: str,
        category: Optional[str] = None,
        name: Optional[str] = None,
        formula: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行算子查询"""
        try:
            if action == "list_operators":
                return self._list_operators(category)
            elif action == "get_operator_info":
                if not name:
                    return {"error": "name 参数必填"}
                return self._get_operator_info(name)
            elif action == "validate_formula":
                if not formula:
                    return {"error": "formula 参数必填"}
                return self._validate_formula(formula)
            else:
                return {"error": f"未知 action: {action}"}
        except Exception as e:
            logger.error("OperatorLookup failed: %s", e)
            return {"error": str(e)}

    def _list_operators(self, category: Optional[str] = None) -> Dict[str, Any]:
        """列出所有算子（可按类别过滤）"""
        from QuantNodes.research.quant_alpha.operator_vocab import (
            list_vocab_operators,
            get_vocab_metadata,
        )

        ops = list_vocab_operators(category=category)
        result = []
        for op_name in ops:
            meta = get_vocab_metadata(op_name)
            if meta:
                result.append({
                    "name": op_name,
                    "category": meta.category,
                    "signature": meta.signature,
                    "doc": meta.doc,
                })
            else:
                result.append({"name": op_name})

        return {
            "operators": result,
            "total": len(result),
            "category_filter": category,
        }

    def _get_operator_info(self, name: str) -> Dict[str, Any]:
        """获取单个算子详情"""
        from QuantNodes.research.quant_alpha.operator_vocab import get_vocab_metadata

        meta = get_vocab_metadata(name)
        if meta is None:
            return {"error": f"算子 '{name}' 不存在"}

        return {
            "name": meta.name,
            "category": meta.category,
            "category_tags": meta.category_tags,
            "signature": meta.signature,
            "parameters": meta.parameters,
            "doc": meta.doc,
            "default_window": meta.default_window,
            "examples": meta.examples,
            "difficulty": meta.difficulty,
            "output_dtype": meta.output_dtype,
            "requires_group_by": meta.requires_group_by,
            "composes_with": meta.composes_with,
        }

    def _validate_formula(self, formula: str) -> Dict[str, Any]:
        """校验公式是否有效"""
        from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
        import polars as pl

        # 构造最小测试数据
        test_data = pl.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"] * 3,
            "code": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "open": [10.0] * 9,
            "high": [10.5] * 9,
            "low": [9.5] * 9,
            "close": [10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 12.0, 12.1, 12.2],
            "vol": [1000.0] * 9,
            "amount": [10000.0] * 9,
        })

        try:
            vocab = OperatorVocab.default()
            result = vocab.evaluate(
                formula=formula,
                data=test_data,
                date_column="date",
                code_column="code",
            )
            if result is not None and len(result) == len(test_data):
                return {
                    "formula": formula,
                    "valid": True,
                    "error": None,
                    "result_length": len(result),
                }
            else:
                return {
                    "formula": formula,
                    "valid": False,
                    "error": "评估结果为空或长度不匹配",
                }
        except Exception as e:
            return {
                "formula": formula,
                "valid": False,
                "error": str(e),
            }

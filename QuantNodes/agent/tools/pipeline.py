# coding=utf-8
"""
Pipeline工具

验证 QuantNodes Pipeline 代码的正确性。
"""

from typing import Any, Dict, List, Optional
import re

from QuantNodes.agent.tools.base import Tool


class PipelineTool(Tool):
    """Pipeline验证工具

    验证 QuantNodes Pipeline 代码的正确性。
    """

    CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pipeline"

    @property
    def description(self) -> str:
        return "验证 QuantNodes Pipeline 代码的正确性，检查语法和结构"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "QuantNodes Pipeline Python代码"
                }
            },
            "required": ["code"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, code: str, **kwargs) -> Dict[str, Any]:
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "nodes": []
        }

        extracted_code = self._extract_code(code)
        if not extracted_code:
            result["is_valid"] = False
            result["errors"].append("No valid code found")
            return result

        try:
            compile(extracted_code, '<string>', 'exec')
        except SyntaxError as e:
            result["is_valid"] = False
            result["errors"].append(f"Syntax error: {e}")
            return result

        result["nodes"] = self._extract_nodes(extracted_code)
        return result

    def _extract_code(self, code: str) -> str:
        match = self.CODE_BLOCK_PATTERN.search(code)
        if match:
            return match.group(1).strip()
        return code.strip()

    def _extract_nodes(self, code: str) -> List[str]:
        nodes = []
        patterns = [
            r'(\w+Node)\s*\(',
            r'FactorPipeline\s*\(',
            r'Pipeline\s*\(',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, code)
            nodes.extend(matches)
        return list(set(nodes))
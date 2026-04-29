# coding=utf-8
"""
沙箱工具

封装 QuantNodes AI 模块的 CodeSandbox，用于代码安全验证。
"""

from typing import Any, Dict, List, Optional

from QuantNodes.ai.sandbox import CodeSandbox as QNCodeSandbox
from QuantNodes.agent.tools.base import Tool


class SandboxTool(Tool):
    """代码安全沙箱工具

    提供代码安全校验，防止执行危险操作。
    """

    def __init__(self, allow_warnings: bool = False, max_code_length: int = 10000):
        self._sandbox = QNCodeSandbox(
            allow_warnings=allow_warnings,
            max_code_length=max_code_length
        )

    @property
    def name(self) -> str:
        return "sandbox"

    @property
    def description(self) -> str:
        return "验证Python代码的安全性，检查危险操作。不执行代码，只返回验证结果"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "待验证的Python代码"
                }
            },
            "required": ["code"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, code: str, **kwargs) -> Dict[str, Any]:
        result = self._sandbox.validate(code)
        return {
            "is_safe": result.is_safe,
            "errors": result.errors,
            "warnings": result.warnings,
            "warnings_only": result.warnings_only
        }
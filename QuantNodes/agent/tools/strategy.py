# coding=utf-8
"""
策略生成工具

封装 QuantNodes AI 模块的 StrategyGenerator。
"""

from typing import Any, Dict
import re

from QuantNodes.agent.tools.base import Tool


class StrategyTool(Tool):
    """策略生成工具

    将自然语言描述转换为 QuantNodes Pipeline 代码。

    需要配置 LLM 客户端才能使用。如果未配置，将返回提示信息。
    """

    CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)

    def __init__(self, llm_client=None):
        if llm_client is None:
            from QuantNodes.ai.llm.gateway import get_llm_gateway
            llm_client = get_llm_gateway()
        self._llm_client = llm_client

    @property
    def name(self) -> str:
        return "strategy"

    @property
    def description(self) -> str:
        return "根据自然语言描述生成 QuantNodes Pipeline 代码"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "策略的自然语言描述，如'生成一个动量因子策略'"
                },
                "validate": {
                    "type": "boolean",
                    "description": "是否验证生成的代码",
                    "default": True
                }
            },
            "required": ["description"]
        }

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, description: str, validate: bool = True, **kwargs) -> Dict[str, Any]:
        result = {
            "code": "",
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "description": description
        }

        if self._llm_client is None:
            result["message"] = (
                "Strategy generation requires LLM client configuration. "
                "Please provide an LLM client when initializing StrategyTool."
            )
            result["status"] = "needs_configuration"
            return result

        try:
            from QuantNodes.ai.strategy_gen import StrategyGenerator
            from QuantNodes.ai.sandbox import CodeSandbox

            generator = StrategyGenerator(
                llm_client=self._llm_client,
                code_sandbox=CodeSandbox()
            )

            gen_result = generator.generate(description, validate=validate)

            result["code"] = gen_result.code
            result["is_valid"] = gen_result.is_valid

            if gen_result.error_message:
                result["errors"].append(gen_result.error_message)

            if gen_result.warnings:
                result["warnings"] = gen_result.warnings

            result["status"] = "success"

        except ImportError as e:
            result["status"] = "error"
            result["errors"] = ["AI module not available: %s" % str(e)]
        except Exception as e:
            result["status"] = "error"
            result["errors"] = [str(e)]

        return result

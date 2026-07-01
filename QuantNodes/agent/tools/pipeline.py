# coding=utf-8
"""
Pipeline工具 - Agent 工具版本

验证 QuantNodes Pipeline 代码的正确性。

实现: 委托给 QuantNodes.methods.pipeline.validate_pipeline，
保持 Tool 接口 (async, dict 返回) 以兼容 Agent 工具注册。
"""

from typing import Any, Dict

from QuantNodes.agent.tools.base import Tool


class PipelineTool(Tool):
    """Pipeline验证工具

    验证 QuantNodes Pipeline 代码的正确性，包括：
    - 语法检查
    - 安全检查（CodeSandbox）
    - 节点提取
    """

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
        from QuantNodes.methods.pipeline import validate_pipeline
        result = validate_pipeline(code=code)
        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "nodes": result.nodes,
            "security_status": result.security_status,
        }
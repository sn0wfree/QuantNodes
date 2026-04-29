# coding=utf-8
"""
Echo测试工具
"""

from .base import Tool


class EchoTool(Tool):
    """Echo测试工具 - 返回输入内容"""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "返回输入的消息内容，用于测试工具调用功能"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "要返回的消息内容"
                }
            },
            "required": ["message"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, message: str, **kwargs) -> str:
        return message

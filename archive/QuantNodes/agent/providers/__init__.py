# coding=utf-8
"""
LLM Provider适配层

LLMProvider基类 / QuantNodes适配器
"""

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .quantnodes import QuantNodesLLMProvider

__all__ = ["LLMProvider", "LLMResponse", "ToolCallRequest", "QuantNodesLLMProvider"]

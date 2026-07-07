"""WikiProxyError (PR6.6 / M4.3 split).

向后兼容: `from QuantNodes.research.wiki import WikiProxyError` 仍可用.
"""
from __future__ import annotations

from typing import Dict

from QuantNodes.core.base import FactorError


class WikiProxyError(FactorError):
    """Wiki proxy 操作错误."""

    code = "WIKI_PROXY_ERROR"

    def __init__(self, message: str, details: Dict = None):
        super().__init__(message)
        self.details = details or {}


__all__ = ["WikiProxyError"]
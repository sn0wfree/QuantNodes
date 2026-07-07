"""Backwards-compat shim for QuantNodes.research.wiki (PR6.6 / M4.3 split).

M4.3 wiki.py 拆分:
  原 `QuantNodes.research.wiki` (1218 行单文件) 已拆分为
  `QuantNodes.research.wiki/` 子包 (8 文件, 各 30-800 行).
  本文件保留为 thin shim, 全部 re-export 自子包.

用法 (向后兼容):
    # 老代码 (依然工作)
    from QuantNodes.research.wiki import WikiFactor, WikiFactorProxy, FactorSource
    from QuantNodes.research.wiki import init_factor_wiki

    # 新代码 (推荐, 显式子包)
    from QuantNodes.research.wiki.factor import WikiFactor
    from QuantNodes.research.wiki.proxy import WikiFactorProxy
    from QuantNodes.research.wiki.enums import FactorSource

未来清理: 下个 PR 可删除本 shim, 全部 caller 改 `from .wiki import` →
         `from .wiki.factor / .proxy / .enums import ...` (mechanical sed).
"""
from __future__ import annotations

from QuantNodes.research.wiki import (  # noqa: F401
    QUANT_RELATION_TYPES,
    FactorCategory,
    FactorSource,
    LogicSource,
    WikiFactor,
    WikiFactorProxy,
    WikiLogic,
    WikiProxyError,
    WikiReproduction,
    WikiStrategy,
    init_factor_wiki,
)

__all__ = [
    "QUANT_RELATION_TYPES",
    "FactorSource",
    "FactorCategory",
    "LogicSource",
    "WikiFactor",
    "WikiLogic",
    "WikiStrategy",
    "WikiReproduction",
    "WikiProxyError",
    "init_factor_wiki",
    "WikiFactorProxy",
]
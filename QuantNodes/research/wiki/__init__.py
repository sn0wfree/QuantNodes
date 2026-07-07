"""QuantNodes.research.wiki — Wiki 因子库子包 (PR6.6 / M4.3 split).

M4.3 wiki.py 拆分:
  原 `QuantNodes.research.wiki` (1218 行单文件) 拆成 8 文件子包:
    - enums.py             FactorSource / FactorCategory / LogicSource + QUANT_RELATION_TYPES
    - factor.py            WikiFactor (V2 23 字段)
    - logic.py             WikiLogic + to_structured_dict/from_structured_dict
    - strategy.py          WikiStrategy
    - reproduction.py      WikiReproduction
    - errors.py            WikiProxyError
    - init_factor_wiki.py  init_factor_wiki() + markdown templates
    - proxy.py             WikiFactorProxy (~795 行)

本 `__init__.py` 全部 re-export — 所有现有
`from QuantNodes.research.wiki import X` 调用保持不变.

向后兼容:
  QuantNodes.research.wiki.py (单文件) 已改为 thin shim, re-export 本子包.
  详见 QuantNodes/research/wiki.py.
"""
from .enums import (
    QUANT_RELATION_TYPES,
    FactorCategory,
    FactorSource,
    LogicSource,
)
from .errors import WikiProxyError
from .factor import WikiFactor
from .init_factor_wiki import init_factor_wiki
from .logic import WikiLogic
from .proxy import WikiFactorProxy
from .reproduction import WikiReproduction
from .strategy import WikiStrategy

__all__ = [
    # Enums + constants
    "QUANT_RELATION_TYPES",
    "FactorSource",
    "FactorCategory",
    "LogicSource",
    # Dataclasses
    "WikiFactor",
    "WikiLogic",
    "WikiStrategy",
    "WikiReproduction",
    # Errors
    "WikiProxyError",
    # Functions
    "init_factor_wiki",
    # Proxy
    "WikiFactorProxy",
]
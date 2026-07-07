"""WikiFactor dataclass (PR6.6 / M4.3 split).

WikiFactor V2 (M3 前置 PR6.5): 23 字段 canonical, 合并自旧 schemas.WikiFactor.
- 21 字段 (身份 + 标签 + 8 指标 + 2 链接 + 3 时间戳 + 1 free-form)
- + factor_params: Dict[str, Any] = field(default_factory=dict)
- + status:        str = "draft"

向后兼容: `from QuantNodes.research.wiki import WikiFactor` 仍可用
(由 `wiki/__init__.py` re-export).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import FactorCategory, FactorSource


@dataclass
class WikiFactor:
    """Wiki 因子定义 (WikiFactor V2: 23 字段).

    与 v1 差异 (PR6.5):
      - + factor_params: Dict[str, Any]  (从 schemas.WikiFactor 合并)
      - + status:        str = "draft"   (从 schemas.WikiFactor 合并)

    Fields:
        name: 因子名 (filename-safe)
        formula: 公式表达式 (math string 或 code reference)
        source: FactorSource 枚举 (5 成员)
        category: FactorCategory 枚举 (7 成员)
        description: 自由文本描述
        tags: 标签列表 (e.g. ["logic-driven", "ir=0.85"])
        factor_params: 运行时参数 (e.g. {"window": 20, "factor_col": "close"}) [V2 NEW]
        status: 生命周期状态 ("draft" | "validated" | "deprecated") [V2 NEW]
        ic_mean ~ turnover: 8 个回测指标
        used_by_strategies: 使用本因子的策略名列表
        strategy_yaml: 嵌入的策略 yaml 字符串
        wiki_page_name: 在 wiki 中的页面路径
        created_at / updated_at: ISO 时间戳
        metadata: 自由扩展 dict
    """

    name: str
    formula: str
    source: FactorSource
    category: FactorCategory
    description: str = ""
    tags: List[str] = field(default_factory=list)
    # V2 NEW: 合并自 schemas.WikiFactor
    factor_params: Dict[str, Any] = field(default_factory=dict)
    status: str = "draft"
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    icir: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    n_dates: Optional[int] = None
    factor_return_corr: Optional[float] = None
    ic_t_stat: Optional[float] = None
    turnover: Optional[float] = None
    group_returns: Optional[List[Dict]] = None
    used_by_strategies: List[str] = field(default_factory=list)
    strategy_yaml: Optional[str] = None
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["WikiFactor"]
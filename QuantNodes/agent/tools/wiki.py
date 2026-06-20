# coding=utf-8
"""
WikiTool - Wiki Knowledge Base Tool for Agent

Phase 3: Wiki Tool Integration
"""

import logging
from typing import Any, Dict, List, Optional

from .base import Tool
from ...research.wiki import (
    WikiFactorProxy,
    WikiFactor,
    WikiLogic,
    WikiStrategy,
    WikiReproduction,
    FactorSource,
    FactorCategory,
    LogicSource,
)


class WikiTool(Tool):
    """Wiki Knowledge Base Operation Tool"""

    name = "wiki"
    description = "QuantNodes Wiki 知识库 - 因子/逻辑/策略的存取与查询"
    read_only = False

    STORE_FACTOR_SCHEMA = {
        "name": "store_factor",
        "description": "将验证通过的因子存储到 Wiki 知识库",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "因子名称（唯一标识）",
                    "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                },
                "formula": {
                    "type": "string",
                    "description": "因子公式（如 ts_mean(close, 20) / ts_mean(close, 60) - 1）",
                },
                "source": {
                    "type": "string",
                    "enum": ["research_report", "auto_research", "manual", "derived", "imported"],
                    "description": "因子来源",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "momentum", "value", "quality",
                        "volatility", "size", "growth", "other",
                    ],
                    "description": "因子分类",
                },
                "ic_mean": {"type": "number", "description": "IC 均值"},
                "ic_std": {"type": "number", "description": "IC 标准差"},
                "icir": {"type": "number", "description": "IC IR"},
                "rank_ic_mean": {"type": "number", "description": "Rank IC 均值"},
                "turnover": {"type": "number", "description": "换手率"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签列表",
                },
                "description": {"type": "string", "description": "因子描述"},
            },
            "required": ["name", "formula", "source", "category"],
        },
    }

    GET_FACTOR_SCHEMA = {
        "name": "get_factor",
        "description": "获取因子详情",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "因子名称"},
            },
            "required": ["name"],
        },
    }

    SEARCH_FACTORS_SCHEMA = {
        "name": "search_factors",
        "description": "全文搜索 Wiki 中的因子",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "default": 10, "description": "返回数量"},
            },
            "required": ["query"],
        },
    }

    LIST_FACTORS_SCHEMA = {
        "name": "list_factors",
        "description": "列举因子（支持过滤）",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "来源过滤"},
                "category": {"type": "string", "description": "分类过滤"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "标签过滤"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    }

    STORE_LOGIC_SCHEMA = {
        "name": "store_logic",
        "description": "存储研报逻辑",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "逻辑名称"},
                "content": {"type": "string", "description": "逻辑内容"},
                "source": {"type": "string", "enum": ["research_report", "manual"]},
                "extracted_formula": {"type": "string", "description": "提取的公式"},
            },
            "required": ["name", "content", "source"],
        },
    }

    STORE_STRATEGY_SCHEMA = {
        "name": "store_strategy",
        "description": "存储策略配置到 Wiki",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "策略名称"},
                "strategy_yaml": {"type": "string", "description": "策略 YAML 配置"},
                "description": {"type": "string", "description": "策略描述"},
                "category": {"type": "string", "default": "general"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "backtest_result": {"type": "object", "description": "回测结果"},
            },
            "required": ["name", "strategy_yaml"],
        },
    }

    GET_STRATEGY_SCHEMA = {
        "name": "get_strategy",
        "description": "获取策略详情",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "策略名称"},
            },
            "required": ["name"],
        },
    }

    LIST_STRATEGIES_SCHEMA = {
        "name": "list_strategies",
        "description": "列举策略",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 50},
            },
        },
    }

    ADD_RELATION_SCHEMA = {
        "name": "add_relation",
        "description": "建立知识图谱关系",
        "parameters": {
            "type": "object",
            "properties": {
                "source_name": {"type": "string", "description": "源节点（如 Factor/xxx）"},
                "target_name": {
                    "type": "string",
                    "description": "目标节点（如 Strategy/yyy）",
                },
                "relation": {
                    "type": "string",
                    "enum": [
                        "uses", "correlates_with", "derived_from",
                        "outperforms", "similar_to",
                    ],
                },
            },
            "required": ["source_name", "target_name", "relation"],
        },
    }

    PING_SCHEMA = {
        "name": "ping",
        "description": "检查 Wiki 可用性",
        "parameters": {"type": "object", "properties": {}},
    }

    STATUS_SCHEMA = {
        "name": "status",
        "description": "获取 Wiki 状态统计",
        "parameters": {"type": "object", "properties": {}},
    }

    SEARCH_SCHEMA = {
        "name": "search",
        "description": "全文搜索 Wiki",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    }

    def __init__(self, wiki_path: str, **kwargs):
        self.proxy = WikiFactorProxy(wiki_path)
        self._logger = logging.getLogger(f"tools.{self.name}")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, action: str, **kwargs: Any) -> Any:
        action_map = {
            "store_factor": self._store_factor,
            "get_factor": self._get_factor,
            "search_factors": self._search_factors,
            "list_factors": self._list_factors,
            "store_logic": self._store_logic,
            "get_logic": self._get_logic,
            "store_strategy": self._store_strategy,
            "get_strategy": self._get_strategy,
            "list_strategies": self._list_strategies,
            "store_reproduction": self._store_reproduction,
            "add_relation": self._add_relation,
            "get_neighbors": self._get_neighbors,
            "ping": self._ping,
            "status": self._status,
            "search": self._search,
        }
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")
        self._logger.info(f"[WikiTool] executing action={action}")
        return await action_map[action](**kwargs)

    def _factor_to_dict(self, factor: WikiFactor) -> Dict[str, Any]:
        return {
            "name": factor.name,
            "formula": factor.formula,
            "source": factor.source.value,
            "category": factor.category.value,
            "ic_mean": factor.ic_mean,
            "ic_std": factor.ic_std,
            "icir": factor.icir,
            "rank_ic_mean": factor.rank_ic_mean,
            "tags": factor.tags,
            "wiki_page_name": factor.wiki_page_name,
        }

    async def _store_factor(self, **kwargs) -> str:
        source = FactorSource(kwargs.pop("source"))
        category = FactorCategory(kwargs.pop("category"))
        factor = WikiFactor(source=source, category=category, **kwargs)
        result = self.proxy.store_factor(factor)
        self._logger.info(f"[WikiTool] stored factor: {result}")
        return result

    async def _get_factor(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        factor = self.proxy.get_factor(name)
        if factor:
            return self._factor_to_dict(factor)
        return None

    async def _search_factors(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        factors = self.proxy.search_factors(query, limit=limit)
        return [self._factor_to_dict(f) for f in factors]

    async def _list_factors(
        self, source=None, category=None, tags=None, limit=50, **kwargs,
    ) -> List[Dict[str, Any]]:
        if source:
            source = FactorSource(source)
        if category:
            category = FactorCategory(category)
        factors = self.proxy.list_factors(source=source, category=category, tags=tags, limit=limit)
        return [self._factor_to_dict(f) for f in factors]

    async def _store_logic(self, **kwargs) -> str:
        source = LogicSource(kwargs.pop("source"))
        logic = WikiLogic(source=source, **kwargs)
        result = self.proxy.store_logic(logic)
        self._logger.info(f"[WikiTool] stored logic: {result}")
        return result

    async def _get_logic(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        logic = self.proxy.get_logic(name)
        if logic:
            return {
                "name": logic.name,
                "content": logic.content,
                "source": logic.source.value,
                "extracted_formula": logic.extracted_formula,
                "validation_status": logic.validation_status,
                "wiki_page_name": logic.wiki_page_name,
            }
        return None

    async def _store_strategy(self, **kwargs) -> str:
        strategy = WikiStrategy(**kwargs)
        result = self.proxy.store_strategy(strategy)
        self._logger.info(f"[WikiTool] stored strategy: {result}")
        return result

    async def _get_strategy(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        strategy = self.proxy.get_strategy(name)
        if strategy:
            return {
                "name": strategy.name,
                "description": strategy.description,
                "category": strategy.category,
                "tags": strategy.tags,
                "strategy_yaml": strategy.strategy_yaml,
                "backtest_result": strategy.backtest_result,
                "wiki_page_name": strategy.wiki_page_name,
            }
        return None

    async def _list_strategies(
        self, category=None, tags=None, limit=50, **kwargs,
    ) -> List[Dict[str, Any]]:
        strategies = self.proxy.list_strategies(category=category, tags=tags, limit=limit)
        return [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "tags": s.tags,
                "wiki_page_name": s.wiki_page_name,
            }
            for s in strategies
        ]

    async def _store_reproduction(self, **kwargs) -> str:
        reproduction = WikiReproduction(**kwargs)
        result = self.proxy.store_reproduction(reproduction)
        self._logger.info(f"[WikiTool] stored reproduction: {result}")
        return result

    async def _add_relation(
        self, source_name: str, target_name: str, relation: str, **kwargs,
    ) -> bool:
        result = self.proxy.add_relation(source_name, target_name, relation)
        self._logger.info(f"[WikiTool] added relation: {source_name} -> {target_name}")
        return result

    async def _get_neighbors(self, name: str, **kwargs) -> List[Dict[str, Any]]:
        return self.proxy.get_neighbors(name)

    async def _ping(self, **kwargs) -> bool:
        return self.proxy.ping()

    async def _status(self, **kwargs) -> Dict[str, Any]:
        return self.proxy.status()

    async def _search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        return self.proxy.wiki.search(query, limit=limit)

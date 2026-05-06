# coding=utf-8
"""
QuantNodes.research - Wiki 因子库代理层 + 自动因子研究

功能3A: WikiFactorProxy (因子库基础设施)
功能3C: AutoResearcher (自动因子挖掘)
"""

from QuantNodes.research.wiki import (
    WikiFactorProxy,
    WikiFactor,
    WikiLogic,
    FactorSource,
    FactorCategory,
    LogicSource,
    WikiProxyError,
    QUANT_RELATION_TYPES,
    init_factor_wiki,
)

from QuantNodes.research.factor_miner import (
    FactorMiner,
    FactorCandidate,
)

from QuantNodes.research.factor_evaluator import (
    FactorEvaluator,
    FactorEvaluationResult,
    EvalConfig,
)

from QuantNodes.research.auto_researcher import (
    AutoResearcher,
    AutoResearchResult,
)

from QuantNodes.research.mcts_search import (
    MCTSSearch,
    MCTSNode,
)

__all__ = [
    # 功能3A
    "WikiFactorProxy",
    "WikiFactor",
    "WikiLogic",
    "FactorSource",
    "FactorCategory",
    "LogicSource",
    "WikiProxyError",
    "QUANT_RELATION_TYPES",
    "init_factor_wiki",
    # 功能3C
    "AutoResearcher",
    "AutoResearchResult",
    "FactorMiner",
    "FactorCandidate",
    "FactorEvaluator",
    "FactorEvaluationResult",
    "EvalConfig",
    "MCTSSearch",
    "MCTSNode",
]

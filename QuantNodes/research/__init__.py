# coding=utf-8
"""
QuantNodes.research - Wiki 因子库代理层 + 研报复现

功能3A: WikiFactorProxy (因子库基础设施)
功能3B: ResearchReportReproducer (研报复现)

**功能3C 已归档（v2.7.0+ Phase C）**：
原 `factor_miner` / `factor_evaluator` / `auto_researcher` / `mcts_search`
四个模块已移至 `QuantNodes.research._legacy_3c`，通过下方 shim 仍可导入
但带强 DeprecationWarning。

**新实现**：
- 算子统一 → `QuantNodes.research.quant_alpha.operator_vocab.OperatorVocab` (162 算子)
- MCTS 搜索 → `QuantNodes.research.quant_alpha.mcts.MCTSSearch` (5 通道反馈 + 谱系)
- Alpha-GPT 工作流 → `QuantNodes.research.quant_alpha.workflow.AlphaGptWorkflow`
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

# 功能3C — Phase C 归档（v2.7.0+）：4 个模块已移至 _legacy_3c/
# 下面 4 个 import 通过 re-export 保持向后兼容，但每次导入触发 DeprecationWarning
import warnings as _warnings

_LEGACY3C_MIGRATION = (
    "QuantNodes.research.{name} 已归档到 _legacy_3c/（Phase C, v2.7.0+）。"
    "请迁移到 QuantNodes.research.quant_alpha.{target}。"
)

_LEGACY3C_TARGETS = {
    "FactorMiner": "operator_vocab.OperatorVocab",
    "FactorCandidate": "operator_vocab.OperatorVocab",
    "FactorEvaluator": "operator_vocab.OperatorVocab",
    "FactorEvaluationResult": "operator_vocab.OperatorVocab",
    "EvalConfig": "operator_vocab.OperatorVocab",
    "AutoResearcher": "workflow.AlphaGptWorkflow",
    "AutoResearchResult": "workflow.AlphaGptResult",
    "MCTSSearch": "mcts.MCTSSearch",
    "MCTSNode": "mcts.MCTSNode",
}


class _LegacyShim:
    """Lazy shim: import 时触发 DeprecationWarning，访问属性时再 import"""

    def __init__(self, module_name: str, attrs: list, target: str):
        self._module_name = module_name
        self._attrs = attrs
        self._target = target
        self._warned = False
        self._module = None

    def _warn_once(self):
        if not self._warned:
            _warnings.warn(
                _LEGACY3C_MIGRATION.format(
                    name=self._module_name.split(".")[-1],
                    target=self._target,
                ),
                DeprecationWarning,
                stacklevel=3,
            )
            self._warned = True

    def __getattr__(self, name: str):
        if name not in self._attrs:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            )
        self._warn_once()
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)


import sys as _sys
_legacy_factor_evaluator = _LegacyShim(
    "QuantNodes.research._legacy_3c.factor_evaluator",
    ["EvalConfig", "FactorEvaluationResult", "FactorEvaluator"],
    "operator_vocab.OperatorVocab",
)
_legacy_factor_miner = _LegacyShim(
    "QuantNodes.research._legacy_3c.factor_miner",
    ["FactorMiner", "FactorCandidate", "TEMPLATES", "DEFAULT_WINDOWS"],
    "operator_vocab.OperatorVocab",
)
_legacy_auto_researcher = _LegacyShim(
    "QuantNodes.research._legacy_3c.auto_researcher",
    ["AutoResearcher", "AutoResearchResult"],
    "workflow.AlphaGptWorkflow",
)
_legacy_mcts_search = _LegacyShim(
    "QuantNodes.research._legacy_3c.mcts_search",
    ["MCTSSearch", "MCTSNode"],
    "mcts.MCTSSearch",
)

# Register in sys.modules for import resolution
_sys.modules[__name__ + ".factor_evaluator"] = _legacy_factor_evaluator
_sys.modules[__name__ + ".factor_miner"] = _legacy_factor_miner
_sys.modules[__name__ + ".auto_researcher"] = _legacy_auto_researcher
_sys.modules[__name__ + ".mcts_search"] = _legacy_mcts_search

# Also set as attributes on the parent module for attribute access
factor_evaluator = _legacy_factor_evaluator
factor_miner = _legacy_factor_miner
auto_researcher = _legacy_auto_researcher
mcts_search = _legacy_mcts_search

from QuantNodes.research.report_reproducer import (
    ResearchReportReproducer,
    ExtractedLogic,
    ReproductionResult,
    ReproductionReport,
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
    # 功能3B
    "ResearchReportReproducer",
    "ExtractedLogic",
    "ReproductionResult",
    "ReproductionReport",
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

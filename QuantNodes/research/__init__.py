# coding=utf-8
"""
QuantNodes.research - Wiki 因子库代理层 + 研报复现

功能3A: WikiFactorProxy (因子库基础设施)
功能3B: ResearchReportReproducer (研报复现)

**功能3C 在 v3.0.0 已完全归档（M2 重构删除了 _legacy_3c/ 包）**：
直接使用 `QuantNodes.research.quant_alpha`：
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

from QuantNodes.research.report_reproducer import (
    ResearchReportReproducer,
    ExtractedLogic,
    ReproductionResult,
    ReproductionReport,
)

# Phase C+ (v4.0.0 reproduction merge): re-export run_backtest / BacktestResult
# M3 (PR4): run_backtest moved from backtest_pkg to backtest; keep public API stable.
from QuantNodes.research.backtest import run_backtest
from QuantNodes.research.paper_understanding.schemas import BacktestResult

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
    # Reproduction merge (v4.0.0)
    "run_backtest",
    "BacktestResult",
]

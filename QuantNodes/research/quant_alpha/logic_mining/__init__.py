# coding=utf-8
"""
logic_mining - 市场逻辑驱动的因子挖掘模块

基于 AlphaLogics 论文 (arXiv 2603.20247) 实现。

核心组件:
- WikiLogicStructured: 逻辑结构化表示
- CompiledConstraint (Γ): 编译后的可执行约束
- compile_to_constraint(): 逻辑 → Γ 编译器
- LogicMiningPipeline: Logic Mining 三段式 Agent
- mine_logic_from_formula(): 从公式抽取逻辑
- build_initial_logic_library(): 构建初始逻辑库
"""

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
    LogicPerformanceEvidence,
    LogicAbstractionResult,
)
from QuantNodes.research.quant_alpha.logic_mining.compiler import (
    CompiledConstraint,
    compile_to_constraint,
)
from QuantNodes.research.quant_alpha.logic_mining.parser import (
    parse_formula_structure,
    parse_financial_semantics,
    parse_market_logic,
)
from QuantNodes.research.quant_alpha.logic_mining.pipelines import (
    LogicMiningPipeline,
    mine_logic_from_formula,
    build_initial_logic_library,
)
from QuantNodes.research.quant_alpha.logic_mining.sources import (
    SOURCES,
    get_formulas_from_source,
    list_available_sources,
)
from QuantNodes.research.quant_alpha.logic_mining.generator import (
    MarketLogicGenerator,
    MarketLogicRefinementDirection,
    generate_logic_name,
)
from QuantNodes.research.quant_alpha.logic_mining.metrics import (
    PipelineMetrics,
    StrictConfig,
    LogicMiningStrictError,
)
from QuantNodes.research.quant_alpha.logic_mining.batch import (
    mine_logic_library_v2,
    ThreadSafeMetrics,
    LogicMiningBatchResult,
)
from QuantNodes.research.quant_alpha.logic_mining.report import (
    MetricsReportBuilder,
)

__all__ = [
    # Models
    "LogicCondition",
    "LogicBehavior",
    "WikiLogicStructured",
    "LogicPerformanceEvidence",
    "LogicAbstractionResult",
    # Compiler (Γ)
    "CompiledConstraint",
    "compile_to_constraint",
    # Parser
    "parse_formula_structure",
    "parse_financial_semantics",
    "parse_market_logic",
    # Pipelines
    "LogicMiningPipeline",
    "mine_logic_from_formula",
    "build_initial_logic_library",
    # Sources
    "SOURCES",
    "get_formulas_from_source",
    "list_available_sources",
    # Generator (外层循环 PR-4)
    "MarketLogicGenerator",
    "MarketLogicRefinementDirection",
    "generate_logic_name",
    # Metrics (v3.0.1 Phase 2)
    "PipelineMetrics",
    "StrictConfig",
    "LogicMiningStrictError",
    # Batch (v3.0.2)
    "mine_logic_library_v2",
    "ThreadSafeMetrics",
    "LogicMiningBatchResult",
    "MetricsReportBuilder",
]

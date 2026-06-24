# coding=utf-8
"""
_legacy_3c - 功能 3C 归档包（Phase C, v2.7.0+）

包含原 v2.x 的 4 个自动因子挖掘模块（已弃用）：
- factor_evaluator: 12-lambda namespace 评估器
- factor_miner: 4 模板族（momentum/reversal/volume/volatility）
- auto_researcher: AutoResearcher 主类
- mcts_search: 7 硬编码 EXTENSION_OPS 的 MCTS

**迁移指南**：

| v2.x (此处) | v2.7.0+ (quant_alpha) |
|-------------|----------------------|
| `factor_evaluator.FactorEvaluator` | `quant_alpha.operator_vocab.OperatorVocab` |
| `factor_miner.FactorMiner` | `quant_alpha.operator_vocab.OperatorVocab` (162 算子) |
| `mcts_search.MCTSSearch` | `quant_alpha.mcts.MCTSSearch` (26 动态操作 + 5 通道反馈) |
| `auto_researcher.AutoResearcher` | `quant_alpha.workflow.AlphaGptWorkflow` |

修复的 3 个 latent bug（vs 旧实现）：
1. `ts_corr/ts_cov` 之前调用不存在的 `Series.rolling_corr` → 用 `pl.rolling_corr`
2. `rank/zscore` 之前全局而非 per-date → 默认 per-date 截面
3. 之前异常被静默吞掉 → 现在显式 raise

这些模块在 v3.0 (Phase C 完全归档) 后将被移除。
"""

from .factor_evaluator import (
    FactorEvaluator,
    FactorEvaluationResult,
    EvalConfig,
)
from .factor_miner import (
    FactorMiner,
    FactorCandidate,
    TEMPLATES,
    DEFAULT_WINDOWS,
)
from .mcts_search import (
    MCTSSearch,
    MCTSNode,
)
from .auto_researcher import (
    AutoResearcher,
    AutoResearchResult,
)


__all__ = [
    "FactorEvaluator",
    "FactorEvaluationResult",
    "EvalConfig",
    "FactorMiner",
    "FactorCandidate",
    "TEMPLATES",
    "DEFAULT_WINDOWS",
    "MCTSSearch",
    "MCTSNode",
    "AutoResearcher",
    "AutoResearchResult",
]

# coding=utf-8
"""
operator_vocab - 算子词表层

把分散在 3 处的 285 个算子整合成统一接口：
- L0 Built-in: factor_node/factor_functions/ (157 算子)
- L0 TA-Lib: factor_node/factor_functions/talib_ops.py (109 算子，需显式 import)
- L1 Composite: operators/composite_dag.py (20 算子)

主要功能：
- 统一查询：list_operators / get_operator / get_metadata
- 算子元数据：12 字段（含 7 个 LLM 友好）
- Namespace 构建：build_namespace（per-date over() 修复）
- 端到端评估：evaluate（formula + data → result）

修复 3 个 latent bug：
1. ts_corr/ts_cov 的 Series.rolling_corr 不存在
2. rank/zscore 全局而非 per-date
3. 异常被静默吞掉
"""

from __future__ import annotations

from QuantNodes.research.quant_alpha.operator_vocab.metadata import (
    OperatorMetadata,
    OperatorCategory,
)
from QuantNodes.research.quant_alpha.operator_vocab.config import (
    OperatorVocabConfig,
)
from QuantNodes.research.quant_alpha.operator_vocab.vocabulary import (
    OperatorVocab,
    build_namespace,
    list_vocab_operators,
    get_vocab_operator,
    get_vocab_metadata,
)

__all__ = [
    # 主类
    "OperatorVocab",
    "OperatorVocabConfig",
    "OperatorMetadata",
    "OperatorCategory",
    # 便捷函数
    "build_namespace",
    "list_vocab_operators",
    "get_vocab_operator",
    "get_vocab_metadata",
]

# coding=utf-8
"""
QuantAlpha - 自动化因子挖掘引擎

从"人工设计因子"到"机器发现因子"的范式升级。

参考 4 大因子库演进链（见 docs/quant_alpha/PROJECT_PLAN.md）：
- Alpha 101 (WorldQuant 2015) - 公式化因子范式
- Alpha 158/360 (Qlib 2020) - ML 友好特征集
- AutoAlpha (清华 2020) - 层次化进化
- AlphaGen/Alpha-GPT (2023+) - RL/LLM 驱动

M1 PR 范围（v2.7.0+）：
- 算子词表 OperatorVocab：统一 285 个算子的查询/调用接口
- 5 个新算子：signedpower / ts_decay_linear / IndNeutralize / ts_skew / ts_kurt
- per-date over() 语义修复：修复旧 12-lambda namespace 的 3 个 latent bug
- 旧 4 文件 DeprecationWarning：标记进入 deprecation 周期

M2+ 路线：MCTS+5通道反馈 / Alpha 101+158/360 借鉴 / PolarsAlphaCalculator / Alpha-GPT。
"""

from __future__ import annotations

# M1 公开 API
from QuantNodes.research.quant_alpha.operator_vocab import (
    OperatorVocab,
    OperatorVocabConfig,
    OperatorMetadata,
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
    # 便捷函数
    "build_namespace",
    "list_vocab_operators",
    "get_vocab_operator",
    "get_vocab_metadata",
]

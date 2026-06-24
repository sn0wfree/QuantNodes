# coding=utf-8
"""Table 4 复现的 evaluation 子包

Stage 1 mock + Stage 2 real 复用同一接口契约（contracts.py）。

主要组件：
- contracts.py：4 dataclass + 4 ABC（接口契约）
- mock_data_loader.py：Stage 1 500 票 GBM 数据生成
- evaluators/polars_evaluator.py：基于 alpha_evaluate tool 的 PolarsEvaluator
- baselines/g1_handcrafted.py：动态从 OperatorVocab 生成 100 公式
- baselines/g2_llm_only.py：mock LLM 直接生成 50 字符串
- baselines/g3_alpha_gpt.py：包 AlphaGptWorkflow（M5）
- runner.py：Table4Runner 主入口
"""

from .contracts import (
    Baseline,
    DataLoader,
    Evaluator,
    FactorMetrics,
    FactorSpec,
    Table4GroupResult,
    Table4Report,
    Table4Runner,
)
from .evaluators import PolarsAlphaCalculatorEvaluator
from .mock_data_loader import MOCK_INDUSTRIES, MockDataLoader
from .runner import MockTable4Runner

__all__ = [
    "DataLoader",
    "Evaluator",
    "Baseline",
    "Table4Runner",
    "FactorSpec",
    "FactorMetrics",
    "Table4GroupResult",
    "Table4Report",
    "PolarsAlphaCalculatorEvaluator",
    "MockDataLoader",
    "MockTable4Runner",
    "MOCK_INDUSTRIES",
]
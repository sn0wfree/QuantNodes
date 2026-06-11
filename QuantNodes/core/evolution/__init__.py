"""EvolutionLoop — 多轮演化主循环。

公开 API:
    - EvolutionSetting: Pydantic 配置
    - FactorCandidate: 候选因子 dataclass
    - Hypothesizer / Mutator / Crosser: 3 个 LLM-based operators
    - EvolutionLoop: 主控循环
    - EvolutionResult: 结果 dataclass
"""
from .settings import EvolutionSetting
from .operators import Crosser, FactorCandidate, Hypothesizer, Mutator
from .loop import EvolutionLoop, EvolutionResult

__all__ = [
    "EvolutionSetting",
    "FactorCandidate",
    "Hypothesizer",
    "Mutator",
    "Crosser",
    "EvolutionLoop",
    "EvolutionResult",
]

"""QualityGate — 3 门质量检查 (Complexity / Redundancy / Consistency)。

公开 API:
    - QualityGateSetting / ComplexitySetting / RedundancySetting / ConsistencySetting
    - FactorZoo: AST hash 库
    - ComplexityChecker / RedundancyChecker / ConsistencyChecker
    - QualityGateNode: 集成 3 门的节点
"""
from .settings import (
    ComplexitySetting,
    ConsistencySetting,
    QualityGateSetting,
    RedundancySetting,
)
from .zoo import FactorZoo, ast_hash
from .complexity import ComplexityChecker
from .redundancy import RedundancyChecker
from .consistency import ConsistencyChecker
from .node import QualityGateNode

__all__ = [
    "ComplexitySetting",
    "ConsistencySetting",
    "QualityGateSetting",
    "RedundancySetting",
    "FactorZoo",
    "ast_hash",
    "ComplexityChecker",
    "RedundancyChecker",
    "ConsistencyChecker",
    "QualityGateNode",
]

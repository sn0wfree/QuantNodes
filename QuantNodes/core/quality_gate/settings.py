"""QualityGate 配置 — 3 个独立可配门 + 总配置。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ComplexitySetting(BaseModel):
    """COMPLEXITY 门配置 — 防过拟合 (AST 静态检查)。"""
    enabled: bool = Field(default=True, description="启用复杂度门")
    symbol_length_threshold: int = Field(default=200, description="表达式字符串长度上限")
    base_features_threshold: int = Field(default=5, description="基础特征数上限")
    free_args_ratio_threshold: float = Field(default=0.5, description="自由参数占比上限")


class RedundancySetting(BaseModel):
    """REDUNDANCY 门配置 — 防重复 (AST hash + 汉明距离)。"""
    enabled: bool = Field(default=True, description="启用冗余门")
    threshold: int = Field(default=5, description="最小汉明距离阈值")
    zoo_path: Optional[str] = Field(default=None, description="因子 Zoo 路径 (None=内存)")


class ConsistencySetting(BaseModel):
    """CONSISTENCY 门配置 — LLM 验证 hypothesis ↔ description ↔ expression。"""
    enabled: bool = Field(default=False, description="启用一致性门 (需 LLM, 默认关闭)")
    model: str = Field(default="mock", description="LLM 模型名")
    max_correction_attempts: int = Field(default=3, description="解析失败最大重试次数")


class QualityGateSetting(BaseModel):
    """质量门总配置。"""
    complexity: ComplexitySetting = Field(default_factory=ComplexitySetting)
    redundancy: RedundancySetting = Field(default_factory=RedundancySetting)
    consistency: ConsistencySetting = Field(default_factory=ConsistencySetting)

    def any_enabled(self) -> bool:
        """是否有任何门启用。"""
        return (
            self.complexity.enabled
            or self.redundancy.enabled
            or self.consistency.enabled
        )

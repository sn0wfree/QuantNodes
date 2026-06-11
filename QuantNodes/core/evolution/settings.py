"""Evolution 配置 — 演化主循环相关设置。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OperatorSetting(BaseModel):
    """LLM Operator 配置 (hypothesize / mutate / crossover)。"""
    enabled: bool = Field(default=True, description="启用该 operator")
    model: str = Field(default="mock", description="LLM 模型 (mock/deepseek-v3/...)")
    max_correction_attempts: int = Field(default=3, description="LLM 解析失败最大重试")
    seed: int = Field(default=42, description="mock 模式随机种子")


class EvolutionSetting(BaseModel):
    """演化主循环配置 (集成到 SingleFactorTestConfig)。"""
    enabled: bool = Field(default=False, description="是否启用演化模式")
    max_rounds: int = Field(default=3, description="演化总轮数 (不含 round 0 原始)")
    parents_per_round: int = Field(default=1, description="每轮选几个 parent (crossover=2)")
    parent_selection_strategy: str = Field(
        default="top_percent_plus_random",
        description="选择策略 (best/random/weighted/weighted_inverse/top_percent_plus_random)",
    )
    top_percent_threshold: float = Field(default=0.3, description="top_percent_plus_random 的 top 比例")
    metric: str = Field(default="sharpe", description="用于排序/加权的指标")
    pool_dir: Optional[str] = Field(default=None, description="TrajectoryPool 目录 (None=output.dir/trajectory)")
    early_stop_patience: int = Field(default=0, description="连续 N 轮无改善则停 (0=不启用)")
    hypothesizer: OperatorSetting = Field(default_factory=OperatorSetting)
    mutator: OperatorSetting = Field(default_factory=OperatorSetting)
    crosser: OperatorSetting = Field(default_factory=OperatorSetting)

    def any_operator_enabled(self) -> bool:
        return (
            self.hypothesizer.enabled
            or self.mutator.enabled
            or self.crosser.enabled
        )

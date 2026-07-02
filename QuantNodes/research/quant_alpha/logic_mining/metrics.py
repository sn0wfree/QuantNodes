# coding=utf-8
"""
metrics.py - Logic Mining 可观测性 & 严格模式

Phase 2 (v3.0.1) 引入。为 silent-fallback 提供可观测指标，
允许外部代码在 strict=True 时升级异常。

设计要点:
- 6 类计数 (call_failures / parse_failures / parse_layer_reached /
  structured_failures / wiki_failures / inner_loop_failures)
- PipelineMetrics 实例默认字段为空, 跨阶段共享
- StrictConfig: 三挡开关 (call/parse/structured) 默认全 False
- LogicMiningStrictError: strict=True 下静默升级为异常

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.metrics import (
        PipelineMetrics, StrictConfig, LogicMiningStrictError,
    )

    metrics = PipelineMetrics()
    strict = StrictConfig(parse=True)

    pipeline = LogicMiningPipeline(llm_client=client, metrics=metrics, strict=strict)
    ...
    print(metrics.to_dict())
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Optional


@dataclass
class StrictConfig:
    """严格模式配置: 控制哪些 silent fallback 升级为异常

    Attributes:
        call:       True 时 LLM 调用失败抛 LogicMiningStrictError
        parse:      True 时 JSON parse 失败抛 LogicMiningStrictError
        structured: True 时 WikiLogicStructured 构建失败抛 LogicMiningStrictError
    """
    call: bool = False
    parse: bool = False
    structured: bool = False

    @classmethod
    def all_off(cls) -> "StrictConfig":
        return cls()


@dataclass
class PipelineMetrics:
    """Logic Mining 可观测性指标

    全部为弱默认, 调用方无须设置即可使用。线程不安全 (单线程 pipeline)
    """
    call_failures: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    parse_failures: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    parse_layer_reached: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    structured_failures: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    wiki_failures: int = 0
    inner_loop_failures: int = 0

    def record_call_failure(self, agent_id: str) -> None:
        """记录 LLM 调用失败"""
        self.call_failures[agent_id] += 1

    def record_parse_failure(self, agent_id: str, layer_reached: int) -> None:
        """记录 JSON 解析失败 + 最远触达层"""
        self.parse_failures[agent_id] += 1
        cur = self.parse_layer_reached[agent_id]
        if layer_reached > cur:
            self.parse_layer_reached[agent_id] = layer_reached

    def record_structured_failure(self, agent_id: str) -> None:
        """记录 WikiLogicStructured 数据解析失败"""
        self.structured_failures[agent_id] += 1

    def record_wiki_failure(self) -> None:
        """记录 Wiki persistence 失败"""
        self.wiki_failures += 1

    def record_inner_loop_failure(self) -> None:
        """记录内层 AlphaGptWorkflow 失败"""
        self.inner_loop_failures += 1

    def to_dict(self) -> Dict[str, Any]:
        """导出为可序列化字典"""
        return {
            "call_failures": dict(self.call_failures),
            "parse_failures": dict(self.parse_failures),
            "parse_layer_reached": dict(self.parse_layer_reached),
            "structured_failures": dict(self.structured_failures),
            "wiki_failures": self.wiki_failures,
            "inner_loop_failures": self.inner_loop_failures,
        }

    def total_failures(self) -> int:
        """总失败计数"""
        return (
            sum(self.call_failures.values())
            + sum(self.parse_failures.values())
            + sum(self.structured_failures.values())
            + self.wiki_failures
            + self.inner_loop_failures
        )


class LogicMiningStrictError(RuntimeError):
    """strict=True 模式下静默失败升级为异常

    异常 message 包含:
    - kind: 'call' / 'parse' / 'structured' / 'wiki' / 'inner_loop'
    - agent_id / layer / 等上下文
    """

    def __init__(self, message: str, *, kind: str, **context: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.context: Dict[str, Any] = dict(context)

    def __repr__(self) -> str:
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"LogicMiningStrictError({self.kind}, {ctx}): {self.args[0]}"


__all__ = [
    "PipelineMetrics",
    "StrictConfig",
    "LogicMiningStrictError",
]

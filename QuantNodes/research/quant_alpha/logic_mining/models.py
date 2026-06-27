# coding=utf-8
"""
models.py - 逻辑结构化数据模型

基于 AlphaLogics 论文的逻辑表示 H = ⟨𝒞, ℬ⟩。

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.models import (
        LogicCondition, LogicBehavior, WikiLogicStructured,
    )

    logic = WikiLogicStructured(
        predicates=[
            LogicCondition(variable="open", op="rank", threshold=0),
            LogicCondition(variable="volume", op="ts_corr", threshold=-0.5, window=10),
        ],
        behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        operator_whitelist=["rank", "ts_corr", "sign"],
        parameter_ranges={"ts_corr": (5, 30)},
        sign_constraint=-1,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LogicCondition",
    "LogicBehavior",
    "WikiLogicStructured",
    "LogicPerformanceEvidence",
    "LogicAbstractionResult",
]


@dataclass
class LogicCondition:
    """单条谓词 (v, op, θ, w)

    对应论文中的条件谓词: variable op threshold
    """

    variable: str           # 市场变量名: "close", "volume", "open", "high", "low"
    op: str                 # 算子名: "ts_corr", "ts_mean", "rank"
    threshold: float        # 阈值 θ
    window: Optional[int] = None  # 时序算子的窗口 d
    weight: float = 1.0     # 权重 w
    second_variable: Optional[str] = None  # 双变量算子的第二个变量

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = {
            "variable": self.variable,
            "op": self.op,
            "threshold": self.threshold,
            "weight": self.weight,
        }
        if self.window is not None:
            d["window"] = self.window
        if self.second_variable is not None:
            d["second_variable"] = self.second_variable
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogicCondition:
        """从字典创建"""
        return cls(
            variable=data["variable"],
            op=data["op"],
            threshold=data.get("threshold", 0.0),
            window=data.get("window"),
            weight=data.get("weight", 1.0),
            second_variable=data.get("second_variable"),
        )


@dataclass
class LogicBehavior:
    """ℬ = (y, d, h): 行为三元组

    - target: 预测目标 (forward_return_1/5/20)
    - direction: 信号方向 (+1/-1)
    - horizon: 持有期天数
    """

    target: str             # "forward_return_1" / "forward_return_5" / "forward_return_20"
    direction: int          # +1(信号方向与目标一致)/ -1(反向)
    horizon: int            # 持有期天数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "target": self.target,
            "direction": self.direction,
            "horizon": self.horizon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogicBehavior:
        """从字典创建"""
        return cls(
            target=data["target"],
            direction=data["direction"],
            horizon=data["horizon"],
        )


@dataclass
class WikiLogicStructured:
    """H_struct: 规范化后的结构化逻辑

    对应论文中的逻辑表示 H = ⟨𝒞, ℬ⟩。
    可通过 compile_to_constraint() 编译为 Γ 约束。
    """

    predicates: List[LogicCondition]  # 条件谓词列表 (𝒞)
    behavior: LogicBehavior           # 行为目标 (ℬ)
    operator_whitelist: Optional[List[str]] = None  # 允许的算子族
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None  # 参数范围
    sign_constraint: Optional[int] = None  # +1 / -1 / None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（便于 JSON 序列化）"""
        d: Dict[str, Any] = {
            "predicates": [p.to_dict() for p in self.predicates],
            "behavior": self.behavior.to_dict(),
        }
        if self.operator_whitelist is not None:
            d["operator_whitelist"] = self.operator_whitelist
        if self.parameter_ranges is not None:
            d["parameter_ranges"] = {
                k: list(v) for k, v in self.parameter_ranges.items()
            }
        if self.sign_constraint is not None:
            d["sign_constraint"] = self.sign_constraint
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WikiLogicStructured:
        """从字典创建"""
        predicates = [LogicCondition.from_dict(p) for p in data.get("predicates", [])]
        behavior = LogicBehavior.from_dict(data["behavior"])

        param_ranges = None
        if "parameter_ranges" in data and data["parameter_ranges"] is not None:
            param_ranges = {
                k: tuple(v) for k, v in data["parameter_ranges"].items()
            }

        return cls(
            predicates=predicates,
            behavior=behavior,
            operator_whitelist=data.get("operator_whitelist"),
            parameter_ranges=param_ranges,
            sign_constraint=data.get("sign_constraint"),
        )

    def get_operators(self) -> List[str]:
        """从谓词中提取所有使用的算子"""
        ops = set()
        for p in self.predicates:
            ops.add(p.op)
        return sorted(ops)

    def get_variables(self) -> List[str]:
        """从谓词中提取所有使用的变量"""
        vars_ = set()
        for p in self.predicates:
            vars_.add(p.variable)
            if p.second_variable:
                vars_.add(p.second_variable)
        return sorted(vars_)


@dataclass
class LogicPerformanceEvidence:
    """聚合的 per-logic 回测证据 E^Logic

    对应论文中每条逻辑下所有因子的回测证据聚合。

    Attributes:
        n_factors_explored: 探索过的因子数
        best_ir: 最佳 IR
        best_ic: 最佳 IC
        best_factor_id: 最佳因子 ID
        mean_ir: 平均 IR
        refinement_round: 第几轮聚合的证据
        timestamp: 时间戳
    """
    n_factors_explored: int = 0
    best_ir: float = 0.0
    best_ic: float = 0.0
    best_factor_id: Optional[str] = None
    mean_ir: float = 0.0
    refinement_round: int = 0
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "n_factors_explored": self.n_factors_explored,
            "best_ir": self.best_ir,
            "best_ic": self.best_ic,
            "best_factor_id": self.best_factor_id,
            "mean_ir": self.mean_ir,
            "refinement_round": self.refinement_round,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogicPerformanceEvidence:
        """从字典创建"""
        return cls(
            n_factors_explored=data.get("n_factors_explored", 0),
            best_ir=data.get("best_ir", 0.0),
            best_ic=data.get("best_ic", 0.0),
            best_factor_id=data.get("best_factor_id"),
            mean_ir=data.get("mean_ir", 0.0),
            refinement_round=data.get("refinement_round", 0),
            timestamp=data.get("timestamp"),
        )


@dataclass
class LogicAbstractionResult:
    """Logic Mining 三段式 Agent 的输出

    Attributes:
        formula_structure: FormulaStructureAgent 输出
        financial_semantics: FinancialSemanticsMappingAgent 输出
        structured_logic: MarketLogicAbstractionAgent 输出
        source_formula: 原始公式
        source_lib: 来源库 ("alpha101" / "alpha158" / "alpha191")
    """
    formula_structure: Dict[str, Any] = field(default_factory=dict)
    financial_semantics: Dict[str, Any] = field(default_factory=dict)
    structured_logic: Optional[WikiLogicStructured] = None
    source_formula: str = ""
    source_lib: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "formula_structure": self.formula_structure,
            "financial_semantics": self.financial_semantics,
            "structured_logic": self.structured_logic.to_dict() if self.structured_logic else None,
            "source_formula": self.source_formula,
            "source_lib": self.source_lib,
        }

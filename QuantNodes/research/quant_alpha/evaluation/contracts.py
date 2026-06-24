# coding=utf-8
"""
contracts.py - Table 4 复现的接口契约

定义 Stage 1（mock）与 Stage 2（real）共用的接口与数据 schema，
确保 mock pipeline 与真实 pipeline 仅需替换 DataLoader / LLMClient 实现，
其余代码可零修改复用。

设计原则：
1. dataclass 风格对齐 workflow/state.py（M5），便于 Alpha-GPT 生态复用
2. ABC 接口仅规定最小契约（method signature + docstring）
3. schema 与 alpha_evaluate tool（M5）保持一致（{status, metrics: {ic_mean, ic_std, ir, ic_decay}}）
4. 借鉴 factor_test/utils/data_loader.py 接口模式（Stage 1 Mock → Stage 2 IFinD 无缝替换）

复用：
- workflow/state.py：IdeaRecord / FormulaRecord 字段命名风格
- alpha_evaluate.py：metrics dict schema
- factor_test/utils/data_loader.py：DataLoader 接口风格

Stage 1 mock：
    MockDataLoader → PolarsAlphaCalculatorEvaluator → Table4Runner
                                                ↓
    G1Handcrafted / G2LlmOnly / G3AlphaGpt 三个 baseline

Stage 2 real（接口同构，仅替换实现）：
    IFinDDataLoader → PolarsAlphaCalculatorEvaluator → Table4Runner
    MiniMaxClient    ↑                                  ↓
    NanobotLLMWrapper                                  G3AlphaGpt (注入 NanobotLLMWrapper)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


__all__ = [
    # dataclass（4）
    "FactorSpec",
    "FactorMetrics",
    "Table4GroupResult",
    "Table4Report",
    # ABC（4）
    "DataLoader",
    "Evaluator",
    "Baseline",
    "Table4Runner",
]


# ---------------------------------------------------------------------------
# Dataclasses（4 个）
# ---------------------------------------------------------------------------


@dataclass
class FactorSpec:
    """单个因子的描述：公式 + 元信息

    对齐 alpha_evaluate tool 输入 schema：
        - formula: polars 表达式字符串（如 "rank(-ts_mean(returns, 20))"）
        - category: 因子类别（momentum / reversal / volatility / volume / value）
        - source: 来源标记（"g1_handcrafted" / "g2_llm_only" / "g3_alpha_gpt"）
        - complexity: 复杂度（算子数，用于 G1 动态生成）
    """

    formula_id: str
    formula: str
    source: str  # "g1_handcrafted" | "g2_llm_only" | "g3_alpha_gpt"
    category: str = "unknown"
    complexity: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "formula": self.formula,
            "source": self.source,
            "category": self.category,
            "complexity": self.complexity,
            "meta": self.meta,
        }


@dataclass
class FactorMetrics:
    """单个因子的评估指标

    对齐 alpha_evaluate tool 输出 schema：
        {
            "status": "success" | "failed",
            "metrics": {ic_mean, ic_std, ir, ic_decay: {1: x, 5: y, 20: z}},
            "error_msg": None | "..."
        }
    """

    formula_id: str
    status: str  # "success" | "failed"
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    ic_decay: Dict[int, float] = field(default_factory=dict)
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "status": self.status,
            "metrics": {
                "ic_mean": self.ic_mean,
                "ic_std": self.ic_std,
                "ir": self.ir,
                "ic_decay": {str(k): v for k, v in self.ic_decay.items()},
            },
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_alpha_evaluate(cls, formula_id: str, eval_dict: Dict[str, Any]) -> "FactorMetrics":
        """从 alpha_evaluate tool 输出 dict 构造（Stage 1/2 通用）"""
        metrics = eval_dict.get("metrics", {})
        decay_raw = metrics.get("ic_decay", {})
        return cls(
            formula_id=formula_id,
            status=eval_dict.get("status", "failed"),
            ic_mean=float(metrics.get("ic_mean", 0.0)),
            ic_std=float(metrics.get("ic_std", 0.0)),
            ir=float(metrics.get("ir", 0.0)),
            ic_decay={int(k): float(v) for k, v in decay_raw.items()},
            error_msg=eval_dict.get("error_msg"),
        )


@dataclass
class Table4GroupResult:
    """单个 baseline 组（G1 / G2 / G3）的汇总结果"""

    group_name: str  # "G1_Handcrafted" | "G2_LlmOnly" | "G3_AlphaGpt"
    factors: List[FactorSpec] = field(default_factory=list)
    metrics: List[FactorMetrics] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for m in self.metrics if m.status == "success")

    @property
    def failed_count(self) -> int:
        return sum(1 for m in self.metrics if m.status == "failed")

    @property
    def avg_ir(self) -> float:
        success_irs = [m.ir for m in self.metrics if m.status == "success"]
        return float(sum(success_irs) / len(success_irs)) if success_irs else 0.0

    @property
    def best_ir(self) -> float:
        success_irs = [m.ir for m in self.metrics if m.status == "success"]
        return float(max(success_irs)) if success_irs else 0.0

    @property
    def avg_ic(self) -> float:
        success_ics = [m.ic_mean for m in self.metrics if m.status == "success"]
        return float(sum(success_ics) / len(success_ics)) if success_ics else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group_name,
            "n_factors": len(self.factors),
            "n_success": self.success_count,
            "n_failed": self.failed_count,
            "avg_ic_mean": self.avg_ic,
            "avg_ir": self.avg_ir,
            "best_ir": self.best_ir,
            "elapsed_sec": self.elapsed_sec,
            "factors": [f.to_dict() for f in self.factors[:10]],  # 仅 top-10
            "metrics": [m.to_dict() for m in self.metrics[:10]],  # 仅 top-10
        }


@dataclass
class Table4Report:
    """Table 4 复现的最终报告：3 组对比 + 论文对比（Stage 2 填）"""

    timestamp: str
    stage: str  # "mock" | "real"
    groups: List[Table4GroupResult] = field(default_factory=list)
    paper_comparison: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def add_group(self, group: Table4GroupResult) -> None:
        self.groups.append(group)

    def rank_groups_by_ir(self) -> List[Table4GroupResult]:
        """按 avg_ir 降序排列（用于验证 G3 > G1 > G2 趋势）"""
        return sorted(self.groups, key=lambda g: g.avg_ir, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "summary": {
                "n_groups": len(self.groups),
                "groups": [
                    {
                        "group": g.group_name,
                        "avg_ir": g.avg_ir,
                        "avg_ic_mean": g.avg_ic,
                        "n_success": g.success_count,
                        "n_failed": g.failed_count,
                    }
                    for g in self.groups
                ],
            },
            "groups": [g.to_dict() for g in self.groups],
            "paper_comparison": self.paper_comparison,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# ABC（4 个）
# ---------------------------------------------------------------------------


class DataLoader(abc.ABC):
    """数据加载器抽象接口

    Stage 1：MockDataLoader（GBM 模拟价格）
    Stage 2：IFinDDataLoader（拉取全 A 5 年数据）

    借鉴 factor_test/utils/data_loader.py 接口风格。
    """

    @abc.abstractmethod
    def load(self) -> Any:
        """加载全市场数据，返回 polars.DataFrame

        必需字段：date, code, open, high, low, close, vol, amount, industry
        可选字段：vwap, adj_factor, float_share, is_st
        """


class Evaluator(abc.ABC):
    """评估器抽象接口

    Stage 1 + Stage 2 共用：PolarsAlphaCalculatorEvaluator
    内部调用 alpha_evaluate tool（M5）批量评估因子。
    """

    @abc.abstractmethod
    def evaluate(
        self,
        factors: List[FactorSpec],
        data: Any,
        forward_returns: Optional[List[int]] = None,
    ) -> List[FactorMetrics]:
        """批量评估因子列表，返回 FactorMetrics 列表

        Args:
            factors: 待评估因子（FactorSpec 列表）
            data: 全市场数据（polars.DataFrame）
            forward_returns: 前瞻期列表（默认 [1]）

        Returns:
            FactorMetrics 列表（顺序与 factors 一一对应）
        """


class Baseline(abc.ABC):
    """Baseline 抽象接口（G1 / G2 / G3）

    Stage 1：
        G1Handcrafted：动态从 OperatorVocab 生成 100 公式
        G2LlmOnly：mock LLM 直接生成 50 字符串
        G3AlphaGpt：包 AlphaGptWorkflow（M5）
    """

    @property
    @abc.abstractmethod
    def group_name(self) -> str:
        """Baseline 组名（如 "G1_Handcrafted"）"""

    @abc.abstractmethod
    def generate_factors(self, n: int = 100) -> List[FactorSpec]:
        """生成 n 个因子（formula_id 唯一）"""


class Table4Runner(abc.ABC):
    """Table 4 主入口抽象接口

    串联 DataLoader + Baseline + Evaluator，输出 Table4Report。
    """

    @abc.abstractmethod
    def run(self) -> Table4Report:
        """执行完整 pipeline，返回 Table4Report

        流程：
            1. data = loader.load()
            2. for baseline in [G1, G2, G3]:
                factors = baseline.generate_factors(n)
                metrics = evaluator.evaluate(factors, data)
                groups.append(Table4GroupResult(...))
            3. report = Table4Report(groups=...)
            4. report.save(json_path)
            5. report.to_markdown(md_path)
            6. return report
        """
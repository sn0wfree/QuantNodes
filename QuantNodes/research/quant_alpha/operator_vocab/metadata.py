# coding=utf-8
"""
算子元数据 schema - 12 字段（含 7 个 LLM 友好）

扩展自 factor_node/factor_functions/_helpers.py::register_operator 的 5 字段，
新增 7 个对 LLM 关键的字段。

字段定义：
- 基础字段（5 个，对应现有 _OPERATOR_REGISTRY）：
    name, category, func, doc, signature, parameters
- LLM 友好字段（7 个，本文件新增）：
    difficulty, category_tags, default_window,
    requires_group_by, output_dtype, examples, composes_with
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# 与 factor_node.factor_functions._helpers.OperatorCategory 保持一致
class OperatorCategory:
    """算子类别常量"""
    POINT = "point"
    TIME = "time"
    SECTION = "section"
    MULTI_SECTION = "multi_section"
    TALIB = "talib"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.POINT, cls.TIME, cls.SECTION, cls.MULTI_SECTION, cls.TALIB]

    @classmethod
    def is_valid(cls, category: str) -> bool:
        return category in cls.all()


@dataclass
class OperatorMetadata:
    """算子元数据 — 12 字段 schema

    用于：
    - LLM prompt 注入（Alpha-GPT 路线 6）
    - 算子文档自动生成
    - 算子查询/筛选
    - 算子推荐（composes_with）

    字段分组：
    - 基础（5）：name, category, func, doc, signature, parameters
    - LLM 友好（7）：difficulty, category_tags, default_window,
                requires_group_by, output_dtype, examples, composes_with
    """

    # === 基础字段（5/6） ===
    name: str
    category: str
    func: Optional[Callable] = None
    doc: str = ""
    signature: str = ""
    parameters: List[str] = field(default_factory=list)

    # === LLM 友好字段（7） ===
    # 难度等级：1=简单（无参/单参），2=中等（2-3 参），3=复杂（4+ 参或嵌套）
    difficulty: int = 1
    # 语义角色标签：central_tendency / dispersion / quantile / polarity /
    #              position / transformation / aggregation / groupby
    category_tags: List[str] = field(default_factory=list)
    # 默认窗口建议：时间序列算子的推荐窗口
    default_window: List[int] = field(default_factory=list)
    # 是否需要 group_by(date) 上下文（用于 per-date over() 修复判断）
    requires_group_by: bool = False
    # 输出数据类型：float64 / int64 / bool / object
    output_dtype: str = "float64"
    # few-shot 示例：每项 {"input": ..., "output": ...}
    examples: List[Dict[str, Any]] = field(default_factory=list)
    # 推荐组合的算子名列表
    composes_with: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict（用于持久化 / LLM 注入）"""
        return {
            "name": self.name,
            "category": self.category,
            "doc": self.doc,
            "signature": self.signature,
            "parameters": list(self.parameters),
            "difficulty": self.difficulty,
            "category_tags": list(self.category_tags),
            "default_window": list(self.default_window),
            "requires_group_by": self.requires_group_by,
            "output_dtype": self.output_dtype,
            "examples": list(self.examples),
            "composes_with": list(self.composes_with),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorMetadata":
        """从 dict 构造"""
        return cls(
            name=data["name"],
            category=data["category"],
            doc=data.get("doc", ""),
            signature=data.get("signature", ""),
            parameters=data.get("parameters", []),
            difficulty=data.get("difficulty", 1),
            category_tags=data.get("category_tags", []),
            default_window=data.get("default_window", []),
            requires_group_by=data.get("requires_group_by", False),
            output_dtype=data.get("output_dtype", "float64"),
            examples=data.get("examples", []),
            composes_with=data.get("composes_with", []),
        )

    @classmethod
    def from_registry_entry(cls, entry: Dict[str, Any]) -> "OperatorMetadata":
        """从 factor_node 的 _OPERATOR_REGISTRY 条目构造（自动推断 7 字段）

        Args:
            entry: _OPERATOR_REGISTRY[category][name] 的值，含
                name, category, func, doc, signature, parameters

        Returns:
            自动推断 7 字段的 OperatorMetadata
        """
        params = entry.get("parameters", [])
        sig = entry.get("signature", "")
        name = entry["name"]
        category = entry["category"]

        # 推断 difficulty
        difficulty = _infer_difficulty(params, sig)

        # 推断 category_tags
        category_tags = _infer_category_tags(name, category, params)

        # 推断 default_window
        default_window = _infer_default_window(name, params)

        # 推断 requires_group_by
        requires_group_by = _infer_requires_group_by(name, category)

        # 推断 output_dtype
        output_dtype = _infer_output_dtype(name, category, params)

        return cls(
            name=name,
            category=category,
            func=entry.get("func"),
            doc=entry.get("doc", ""),
            signature=sig,
            parameters=params,
            difficulty=difficulty,
            category_tags=category_tags,
            default_window=default_window,
            requires_group_by=requires_group_by,
            output_dtype=output_dtype,
            examples=[],
            composes_with=[],
        )


def _infer_difficulty(params: List[str], signature: str) -> int:
    """推断难度等级

    - 1：0-1 参数
    - 2：2-3 参数
    - 3：4+ 参数 或嵌套调用
    """
    n = len([p for p in params if p not in ("self", "kwargs", "args")])
    if n <= 1:
        return 1
    if n <= 3:
        return 2
    return 3


def _infer_category_tags(name: str, category: str, params: List[str]) -> List[str]:
    """推断语义角色标签

    基于算子名前缀和类别：
    - mean/median/sum → central_tendency
    - std/var/skew/kurt → dispersion
    - quantile/percentile → quantile
    - sign/abs → polarity
    - argmax/argmin → position
    - rank/zscore/winsorize → transformation
    """
    tags = []
    name_lower = name.lower()

    if any(k in name_lower for k in ("mean", "median", "sum", "sma", "ema")):
        tags.append("central_tendency")
    if any(k in name_lower for k in ("std", "var", "skew", "kurt", "vol")):
        tags.append("dispersion")
    if any(k in name_lower for k in ("quantile", "percentile", "qtl")):
        tags.append("quantile")
    if any(k in name_lower for k in ("sign", "abs", "signedpower")):
        tags.append("polarity")
    if any(k in name_lower for k in ("argmax", "argmin", "idxmax", "idxmin")):
        tags.append("position")
    if any(k in name_lower for k in ("rank", "zscore", "winsorize", "scale", "normalize")):
        tags.append("transformation")
    if any(k in name_lower for k in ("corr", "cov", "regress", "beta")):
        tags.append("correlation")
    if any(k in name_lower for k in ("delta", "diff", "pct_change", "return", "roc")):
        tags.append("momentum")
    if any(k in name_lower for k in ("lag", "lead", "shift", "delay", "ref")):
        tags.append("shift")

    if category == OperatorCategory.SECTION:
        tags.append("cross_sectional")
    elif category == OperatorCategory.MULTI_SECTION:
        tags.append("groupby")
    elif category == OperatorCategory.TIME:
        tags.append("time_series")
    elif category == OperatorCategory.POINT:
        tags.append("pointwise")

    return list(set(tags))


def _infer_default_window(name: str, params: List[str]) -> List[int]:
    """推断默认窗口建议

    时序算子默认 [5, 10, 20, 60]
    """
    name_lower = name.lower()
    if any(k in name_lower for k in (
        "ts_", "rolling_", "ewm_", "expanding_", "decay_", "lag", "shift", "delta"
    )):
        return [5, 10, 20, 60]
    return []


def _infer_requires_group_by(name: str, category: str) -> bool:
    """推断是否需要 group_by(date) 上下文

    截面算子（rank/zscore/winsorize/normalize/IndNeutralize/aggregate）需要。
    """
    name_lower = name.lower()
    if category == OperatorCategory.SECTION:
        return True
    if category == OperatorCategory.MULTI_SECTION:
        return True
    if any(k in name_lower for k in ("rank", "zscore", "winsorize", "normalize", "neutralize")):
        return True
    return False


def _infer_output_dtype(name: str, category: str, params: List[str]) -> str:
    """推断输出数据类型

    - argmax/argmin/sign/condition → int64 或 bool
    - 其它 → float64
    """
    name_lower = name.lower()
    if any(k in name_lower for k in ("argmax", "argmin", "idxmax", "idxmin", "row_number")):
        return "int64"
    if any(k in name_lower for k in ("sign", "is_", "where", "cond", "isnull", "notnull")):
        return "bool"
    return "float64"

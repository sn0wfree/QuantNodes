# coding=utf-8
"""
compiler.py - Logic→Γ 编译器

把 WikiLogicStructured 编译为 CompiledConstraint (Γ)，
用于约束因子生成过程。

基于 AlphaLogics 论文 (arXiv 2603.20247) 的核心发现：
- Γ 约束生成 > 自由生成 (Figure 3)
- 逻辑库越大，因子质量单调提升 (Figure 5)

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.compiler import (
        CompiledConstraint, compile_to_constraint,
    )
    from QuantNodes.research.quant_alpha.logic_mining.models import (
        WikiLogicStructured, LogicCondition, LogicBehavior,
    )

    # 定义逻辑
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

    # 编译为 Γ 约束
    gamma = compile_to_constraint(logic)

    # 校验公式
    passed, reason = gamma.validate("sign(-ts_corr(rank(open), rank(volume), 10))")
    assert passed == True

    # 生成 prompt 注入文本
    prompt_text = gamma.render_for_prompt()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from QuantNodes.research.quant_alpha.logic_mining.models import WikiLogicStructured

logger = logging.getLogger(__name__)

__all__ = [
    "CompiledConstraint",
    "compile_to_constraint",
    "extract_operators",
    "extract_variables",
    "parse_op_args",
]


# ==============================================================================
# 默认配置
# ==============================================================================

# 默认算子白名单（OperatorVocab 中常用的量价算子）
DEFAULT_OPERATOR_WHITELIST: Set[str] = {
    # 截面算子
    "rank", "zscore",
    # 时序算子
    "ts_mean", "ts_std", "ts_corr", "ts_cov", "ts_delta", "ts_rank",
    "ts_min", "ts_max", "ts_argmin", "ts_argmax", "ts_sum", "ts_count",
    "ts_skew", "ts_kurt",
    # 点算子
    "abs", "sign", "log", "sqrt", "signedpower",
    # 算术算子
    "add", "sub", "mul", "div",
    # 比较算子
    "max", "min",
}

# 默认变量白名单（OHLCV 量价变量）
DEFAULT_VARIABLE_WHITELIST: Set[str] = {
    "open", "high", "low", "close", "vol", "amount",
    "returns", "volume",
}

# 默认参数范围
DEFAULT_PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
    "ts_mean": (2, 120),
    "ts_std": (2, 120),
    "ts_corr": (2, 60),
    "ts_cov": (2, 60),
    "ts_delta": (1, 60),
    "ts_rank": (2, 120),
    "ts_min": (2, 120),
    "ts_max": (2, 120),
    "ts_argmin": (2, 120),
    "ts_argmax": (2, 120),
    "ts_sum": (2, 120),
    "ts_count": (2, 120),
    "ts_skew": (2, 120),
    "ts_kurt": (2, 120),
}


# ==============================================================================
# 辅助函数
# ==============================================================================


def extract_operators(formula: str) -> Set[str]:
    """从公式中提取使用的算子名

    Args:
        formula: 因子公式字符串

    Returns:
        使用的算子名集合
    """
    # 匹配函数调用: word(
    pattern = r'\b([a-zA-Z_]\w*)\s*\('
    matches = re.findall(pattern, formula)

    # 过滤掉常见的非算子关键字
    keywords = {"if", "else", "for", "while", "return", "def", "class", "None", "True", "False"}
    ops = {m for m in matches if m not in keywords}

    return ops


def extract_variables(formula: str) -> Set[str]:
    """从公式中提取使用的市场变量名

    Args:
        formula: 因子公式字符串

    Returns:
        使用的变量名集合
    """
    # 匹配独立的变量名（不在函数调用中的标识符）
    # 简单实现：匹配已知的 OHLCV 变量
    known_vars = {"open", "high", "low", "close", "vol", "amount", "returns", "volume"}
    found = set()

    for var in known_vars:
        # 使用单词边界匹配
        if re.search(r'\b' + var + r'\b', formula):
            found.add(var)

    return found


def parse_op_args(formula: str) -> List[Tuple[str, List[float]]]:
    """解析公式中算子的数值参数

    Args:
        formula: 因子公式字符串

    Returns:
        [(算子名, [参数值列表]), ...]
    """
    results = []

    # 匹配函数调用及其参数
    # 使用更简单的方法：找到所有 func( 的位置，然后提取参数
    # 例如: ts_mean(close, 20) -> ("ts_mean", ["20"])

    # 先找到所有函数调用
    func_pattern = r'(\w+)\s*\('
    for match in re.finditer(func_pattern, formula):
        op_name = match.group(1)
        start_pos = match.end()

        # 找到匹配的右括号
        depth = 1
        pos = start_pos
        while pos < len(formula) and depth > 0:
            if formula[pos] == '(':
                depth += 1
            elif formula[pos] == ')':
                depth -= 1
            pos += 1

        # 提取括号内的内容
        args_str = formula[start_pos:pos-1]

        # 提取数值参数
        nums = []
        for arg in args_str.split(','):
            arg = arg.strip()
            try:
                nums.append(float(arg))
            except ValueError:
                pass

        if nums:
            results.append((op_name, nums))

    return results


def check_sign_hint(formula: str, direction: int) -> bool:
    """检查公式是否符合指定的符号方向

    启发式检查：
    - 如果 direction == -1，检查是否有负号或 sign 取反
    - 如果 direction == +1，检查是否有正向结构

    Args:
        formula: 因子公式字符串
        direction: 期望的方向 (+1/-1)

    Returns:
        是否符合

    Note:
        V8 修复 (test/expand-coverage-2x Phase 1):
        修复前 direction=-1 时无负向标记的公式被宽松接受 (return True 兜底)
        修复后 direction=-1 严格: 必须有 - / sign(- / sub(0, ...) 才接受
        这与 sign_constraint=-1 的语义一致 (期望负 IR, 必须有显式负向)
    """
    formula_lower = formula.lower().strip()

    if direction == -1:
        # 检查是否有负号
        if formula_lower.startswith('-'):
            return True
        # 检查是否有 sign(-...) 结构
        if 'sign(-' in formula_lower or 'sign( -' in formula_lower:
            return True
        # 检查是否有 sub(0, ...) 结构
        if 'sub(0' in formula_lower:
            return True
        # 无负向标记 → 严格拒绝 (V8 修复: 去掉宽松兜底)
        return False

    elif direction == +1:
        # 正向约束通常不需要特殊标记
        return True

    return True


# ==============================================================================
# 核心数据结构
# ==============================================================================


@dataclass
class CompiledConstraint:
    """Γ: 编译后的可执行约束

    对应论文中的 Γ 约束，用于约束因子生成过程。
    """

    operator_whitelist: Set[str]           # 允许的算子名集合
    operator_blacklist: Set[str] = field(default_factory=set)  # 显式禁止的算子
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # 参数范围
    sign_constraint: Optional[int] = None  # +1 / -1 / None
    variable_whitelist: Optional[Set[str]] = None  # 允许的市场变量
    source_logic: Optional[str] = None     # 来源逻辑名（用于追溯）

    def validate(self, formula: str) -> Tuple[bool, Optional[str]]:
        """校验 formula 是否满足所有约束

        Args:
            formula: 因子公式字符串

        Returns:
            (passed, reason): passed=True 表示通过，reason 为失败原因
        """
        # 1. 提取使用的算子
        used_ops = extract_operators(formula)

        # 2. 黑名单检查
        for op in used_ops:
            if op in self.operator_blacklist:
                return False, f"operator '{op}' is blacklisted"

        # 3. 白名单检查
        for op in used_ops:
            if op not in self.operator_whitelist:
                return False, f"operator '{op}' not in whitelist"

        # 4. 提取使用的变量
        used_vars = extract_variables(formula)

        # 5. 变量白名单检查
        if self.variable_whitelist is not None:
            for v in used_vars:
                if v not in self.variable_whitelist:
                    return False, f"variable '{v}' not in whitelist"

        # 6. 参数范围检查
        for op, args in parse_op_args(formula):
            if op in self.parameter_ranges:
                lo, hi = self.parameter_ranges[op]
                for arg in args:
                    if not (lo <= arg <= hi):
                        return False, f"{op} arg {arg} not in [{lo}, {hi}]"

        # 7. 符号方向检查（启发式）
        if self.sign_constraint is not None:
            if not check_sign_hint(formula, self.sign_constraint):
                return False, f"sign_constraint {self.sign_constraint} not satisfied"

        return True, None

    def render_for_prompt(self) -> str:
        """生成可注入 LLM prompt 的人类可读描述

        Returns:
            格式化的约束描述文本
        """
        lines = ["## Γ 约束（必须遵守）", ""]

        # 算子白名单
        if self.operator_whitelist:
            ops = ", ".join(sorted(self.operator_whitelist))
            lines.append("### 允许使用的算子")
            lines.append(f"只使用以下算子: {ops}")
            lines.append("")

        # 变量白名单
        if self.variable_whitelist is not None:
            vars_ = ", ".join(sorted(self.variable_whitelist))
            lines.append("### 允许使用的变量")
            lines.append(f"只使用以下变量: {vars_}")
            lines.append("")

        # 参数范围
        if self.parameter_ranges:
            lines.append("### 参数范围")
            for op, (lo, hi) in sorted(self.parameter_ranges.items()):
                lines.append(f"- {op}: [{lo}, {hi}]")
            lines.append("")

        # 符号约束
        if self.sign_constraint is not None:
            direction = "正向" if self.sign_constraint > 0 else "反向"
            lines.append("### 符号方向")
            lines.append(f"因子整体方向: {direction} ({self.sign_constraint:+d})")
            lines.append("")

        # 来源逻辑
        if self.source_logic:
            lines.append(f"### 来源逻辑")
            lines.append(f"基于市场逻辑: {self.source_logic}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operator_whitelist": sorted(self.operator_whitelist),
            "operator_blacklist": sorted(self.operator_blacklist),
            "parameter_ranges": {k: list(v) for k, v in self.parameter_ranges.items()},
            "sign_constraint": self.sign_constraint,
            "variable_whitelist": sorted(self.variable_whitelist) if self.variable_whitelist else None,
            "source_logic": self.source_logic,
        }


# ==============================================================================
# 编译器
# ==============================================================================


def compile_to_constraint(
    logic: WikiLogicStructured,
    source_logic: Optional[str] = None,
) -> CompiledConstraint:
    """主入口: H_struct → Γ

    把 WikiLogicStructured 编译为 CompiledConstraint。

    Args:
        logic: 结构化逻辑
        source_logic: 来源逻辑名（用于追溯）

    Returns:
        CompiledConstraint (Γ 约束)
    """
    # 1. 从谓词中提取算子和变量
    ops_from_predicates = set(logic.get_operators())
    vars_from_predicates = set(logic.get_variables())

    # 2. 构建算子白名单
    if logic.operator_whitelist is not None:
        # 使用显式指定的白名单
        op_whitelist = set(logic.operator_whitelist)
    else:
        # 使用谓词中的算子 + 默认算子
        op_whitelist = ops_from_predicates.copy()
        # 添加常用的基础算子
        op_whitelist.update({"add", "sub", "mul", "div", "abs", "sign"})
        # 只保留默认白名单中存在的算子
        op_whitelist = op_whitelist.intersection(DEFAULT_OPERATOR_WHITELIST)

    # 3. 构建变量白名单
    if vars_from_predicates:
        var_whitelist = vars_from_predicates.copy()
    else:
        var_whitelist = DEFAULT_VARIABLE_WHITELIST.copy()

    # 4. 构建参数范围
    param_ranges: Dict[str, Tuple[float, float]] = {}

    # 从默认范围中获取
    for op in op_whitelist:
        if op in DEFAULT_PARAMETER_RANGES:
            param_ranges[op] = DEFAULT_PARAMETER_RANGES[op]

    # 从逻辑中获取显式范围
    if logic.parameter_ranges is not None:
        param_ranges.update(logic.parameter_ranges)

    # 从谓词窗口中获取固定范围
    for p in logic.predicates:
        if p.window is not None:
            op = p.op
            if op in param_ranges:
                # 收缩范围以包含窗口值
                lo, hi = param_ranges[op]
                param_ranges[op] = (min(lo, p.window), max(hi, p.window))
            else:
                # 固定窗口
                param_ranges[op] = (p.window, p.window)

    # 5. 构建符号约束
    sign_constraint = logic.sign_constraint
    if sign_constraint is None:
        # 从行为方向推断
        sign_constraint = logic.behavior.direction

    # 6. 创建 CompiledConstraint
    gamma = CompiledConstraint(
        operator_whitelist=op_whitelist,
        operator_blacklist=set(),
        parameter_ranges=param_ranges,
        sign_constraint=sign_constraint,
        variable_whitelist=var_whitelist,
        source_logic=source_logic,
    )

    logger.info(
        "编译逻辑 → Γ: %d 算子, %d 变量, %d 参数范围",
        len(op_whitelist),
        len(var_whitelist),
        len(param_ranges),
    )

    return gamma

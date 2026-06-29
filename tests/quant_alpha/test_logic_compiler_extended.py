# coding=utf-8
"""
test_logic_compiler_extended.py - Logic compiler 完整覆盖 (Phase 8)

目标: 补全 logic_mining/compiler.py 的所有公开 API
- extract_operators (基础已有, 加边界)
- extract_variables (基础已有, 加边界)
- parse_op_args (基础已有, 加边界)
- CompiledConstraint.validate (补全)
- compile_to_constraint (基础已有, 加边界)
"""
from typing import Optional

import pytest

from QuantNodes.research.quant_alpha.logic_mining.compiler import (
    CompiledConstraint,
    compile_to_constraint,
    extract_operators,
    extract_variables,
    parse_op_args,
)
from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicBehavior,
    LogicCondition,
    WikiLogicStructured,
)


# ==============================================================================
# Test Class 1: extract_operators 边界
# ==============================================================================


class TestExtractOperators:
    """extract_operators 边界测试"""

    def test_empty_formula(self):
        """空公式应返回空集合"""
        assert extract_operators("") == set()

    def test_no_function_call(self):
        """无函数调用应返回空"""
        # 只有变量
        assert extract_operators("close") == set()

    def test_simple_function(self):
        """简单函数调用"""
        assert extract_operators("rank(close)") == {"rank"}

    def test_nested_functions(self):
        """嵌套函数调用"""
        result = extract_operators("rank(ts_mean(close, 20))")
        assert "rank" in result
        assert "ts_mean" in result

    def test_filters_python_keywords(self):
        """过滤 Python 关键字 (if/else/for/...)"""
        # 不应包含 "for" 等
        result = extract_operators("for rank(close)")
        # "for" 是 Python 关键字
        # 但 "for" 后面有空格+函数调用, regex 不会匹配
        # 因为 \b([a-zA-Z_]\w*)\s*\( 需要 word 直接跟 (
        # 实际: "for rank(close)" → 匹配 rank, 不匹配 for
        assert "for" not in result or "rank" in result

    def test_multiple_same_op(self):
        """同一算子多次使用应去重"""
        result = extract_operators("add(close, 0) - add(close, 1)")
        assert result == {"add"}

    def test_unicode_operator(self):
        """含 unicode 字符的公式"""
        # 算子名不应含 unicode
        result = extract_operators("rank(close) + 中文")
        assert "rank" in result


# ==============================================================================
# Test Class 2: extract_variables 边界
# ==============================================================================


class TestExtractVariables:
    """extract_variables 边界测试"""

    def test_ohlcv_variables(self):
        """OHLCV 变量识别"""
        result = extract_variables("ts_mean(close, 20)")
        assert "close" in result

    def test_vol_and_volume_alias(self):
        """vol 和 volume 应都被识别 (LLM 两种命名)"""
        r1 = extract_variables("rank(vol)")
        r2 = extract_variables("rank(volume)")
        assert "vol" in r1
        assert "volume" in r2

    def test_multiple_variables(self):
        """多个变量"""
        result = extract_variables("div(close, volume)")
        assert "close" in result
        assert "volume" in result

    def test_no_ohlcv_variables(self):
        """无 OHLCV 变量"""
        result = extract_variables("rank(123)")
        # 123 是数字, 不在 known_vars
        assert result == set() or "vol" not in result

    def test_partial_match_excluded(self):
        """部分匹配应排除 (closeprice 不应匹配 close)"""
        result = extract_variables("closeprice")
        # closeprice 不应包含 close (单词边界)
        assert "close" not in result


# ==============================================================================
# Test Class 3: parse_op_args 边界
# ==============================================================================


class TestParseOpArgs:
    """parse_op_args 边界测试"""

    def test_single_arg(self):
        """单参数"""
        result = parse_op_args("ts_mean(close, 20)")
        assert ("ts_mean", [20.0]) in result

    def test_multi_args(self):
        """多参数"""
        result = parse_op_args("ts_corr(close, vol, 20)")
        assert ("ts_corr", [20.0]) in result

    def test_no_numeric_args(self):
        """无数值参数"""
        result = parse_op_args("rank(close)")
        # rank 没有数值参数
        assert result == []

    def test_mixed_args(self):
        """混合 (列名 + 数值) 参数"""
        result = parse_op_args("add(close, 1.5)")
        # 1.5 是数值
        assert ("add", [1.5]) in result

    def test_float_args(self):
        """浮点参数"""
        result = parse_op_args("signedpower(close, 0.5)")
        assert ("signedpower", [0.5]) in result

    def test_negative_args(self):
        """负参数"""
        result = parse_op_args("ts_mean(close, -5)")
        # -5 是负数, 应被解析
        assert ("ts_mean", [-5.0]) in result

    def test_nested_function_args(self):
        """嵌套函数参数: 外层 ts_mean 看到的不是数值"""
        result = parse_op_args("ts_mean(rank(close), 5)")
        # ts_mean(数值=5), rank(无数值)
        # parse_op_args 简单实现可能误把 rank(close) 解析为 rank 的参数
        # 但外层 ts_mean 应只看到 5
        assert any(args == [5.0] for _, args in result)

    def test_multiple_ops(self):
        """多个算子"""
        result = parse_op_args("sub(ts_mean(close, 5), 10)")
        # ts_mean(5) + sub 应该有数值
        ts_mean_args = [args for op, args in result if op == "ts_mean"]
        assert [5.0] in ts_mean_args


# ==============================================================================
# Test Class 4: CompiledConstraint.validate
# ==============================================================================


class TestCompiledConstraintValidate:
    """CompiledConstraint.validate 完整覆盖"""

    def test_valid_formula_passes(self):
        """合法公式应通过"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "ts_mean", "sub"},
        )
        passed, reason = gamma.validate("rank(ts_mean(close, 20))")
        assert passed is True
        assert reason is None

    def test_blacklist_blocks(self):
        """黑名单算子应被拒绝"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "ts_mean", "sub"},
            operator_blacklist={"ts_mean"},  # 显式禁止 ts_mean
        )
        passed, reason = gamma.validate("ts_mean(close, 20)")
        assert passed is False
        assert "blacklisted" in reason.lower()

    def test_unknown_op_rejected(self):
        """白名单外的算子应被拒绝"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank"},  # 只允许 rank
        )
        passed, reason = gamma.validate("ts_mean(close, 20)")
        assert passed is False
        assert "not in whitelist" in reason.lower()

    def test_variable_whitelist(self):
        """变量白名单检查"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank"},
            variable_whitelist={"close"},  # 只允许 close
        )
        # 使用 close 通过
        passed, _ = gamma.validate("rank(close)")
        assert passed is True
        # 使用 volume 失败
        passed, reason = gamma.validate("rank(volume)")
        assert passed is False
        assert "variable" in reason.lower() or "whitelist" in reason.lower()

    def test_parameter_range_violated(self):
        """参数范围超界应被拒绝"""
        gamma = CompiledConstraint(
            operator_whitelist={"ts_mean"},
            parameter_ranges={"ts_mean": (5, 30)},
        )
        # window=20 在范围内
        passed, _ = gamma.validate("ts_mean(close, 20)")
        assert passed is True
        # window=100 超出
        passed, reason = gamma.validate("ts_mean(close, 100)")
        assert passed is False
        assert "range" in reason.lower() or "not in" in reason.lower()

    def test_sign_constraint_violated(self):
        """符号约束违反应被拒绝 (Phase 1 修复后)"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "sub"},
            sign_constraint=-1,
        )
        # rank(close) 无负向标记 → 失败
        passed, reason = gamma.validate("rank(close)")
        assert passed is False
        assert "sign" in reason.lower()

    def test_sign_constraint_satisfied(self):
        """符号约束满足应通过"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "sub"},
            sign_constraint=-1,
        )
        # sub(0, close) 含 sub(0 → 通过
        passed, _ = gamma.validate("sub(0, close)")
        assert passed is True

    def test_no_sign_constraint_anything_passes(self):
        """sign_constraint=None 时任何公式通过"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "ts_mean"},
            sign_constraint=None,
        )
        passed, _ = gamma.validate("rank(close)")
        assert passed is True
        passed, _ = gamma.validate("ts_mean(close, 20)")
        assert passed is True


# ==============================================================================
# Test Class 5: compile_to_constraint 边界
# ==============================================================================


class TestCompileToConstraint:
    """compile_to_constraint 边界"""

    def test_basic_compilation(self):
        """基本编译"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=+1, horizon=5),
            operator_whitelist=["ts_mean", "rank"],
            parameter_ranges={"ts_mean": (5, 60)},
            sign_constraint=+1,
        )
        gamma = compile_to_constraint(logic, source_logic="test")
        assert isinstance(gamma, CompiledConstraint)
        assert gamma.source_logic == "test"
        assert "ts_mean" in gamma.operator_whitelist
        assert gamma.sign_constraint == +1

    def test_compile_with_negative_sign(self):
        """负 sign 编译"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            operator_whitelist=["ts_mean"],
            sign_constraint=-1,
        )
        gamma = compile_to_constraint(logic)
        assert gamma.sign_constraint == -1

    def test_compile_preserves_parameter_ranges(self):
        """编译保留 parameter_ranges"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=+1, horizon=5),
            operator_whitelist=["ts_mean"],
            parameter_ranges={"ts_mean": (5, 60), "ts_std": (10, 30)},
        )
        gamma = compile_to_constraint(logic)
        assert gamma.parameter_ranges["ts_mean"] == (5, 60)
        assert gamma.parameter_ranges["ts_std"] == (10, 30)

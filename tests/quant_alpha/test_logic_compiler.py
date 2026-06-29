# coding=utf-8
"""
test_logic_compiler.py - Γ 编译器单元测试

测试：
- WikiLogicStructured 序列化/反序列化
- compile_to_constraint() 编译功能
- CompiledConstraint.validate() 校验逻辑
- CompiledConstraint.render_for_prompt() Prompt 注入
"""

import pytest

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.logic_mining.compiler import (
    CompiledConstraint,
    check_sign_hint,
    compile_to_constraint,
    extract_operators,
    extract_variables,
    parse_op_args,
)


# ==============================================================================
# 辅助函数测试
# ==============================================================================


class TestExtractOperators:
    """extract_operators 测试"""

    def test_simple_function(self):
        """简单函数调用"""
        ops = extract_operators("rank(close)")
        assert ops == {"rank"}

    def test_nested_functions(self):
        """嵌套函数调用"""
        ops = extract_operators("ts_mean(rank(close), 20)")
        assert ops == {"ts_mean", "rank"}

    def test_complex_formula(self):
        """复杂公式"""
        ops = extract_operators("sign(-ts_corr(rank(open), rank(volume), 10))")
        assert "sign" in ops
        assert "ts_corr" in ops
        assert "rank" in ops

    def test_arithmetic_operators(self):
        """算术运算符（非函数调用）"""
        ops = extract_operators("add(close, sub(close, open))")
        assert ops == {"add", "sub"}


class TestExtractVariables:
    """extract_variables 测试"""

    def test_single_variable(self):
        """单个变量"""
        vars_ = extract_variables("rank(close)")
        assert "close" in vars_

    def test_multiple_variables(self):
        """多个变量"""
        vars_ = extract_variables("ts_corr(open, volume, 10)")
        assert "open" in vars_
        assert "volume" in vars_

    def test_no_variables(self):
        """无变量"""
        vars_ = extract_variables("ts_mean(1, 2, 3)")
        assert len(vars_) == 0


class TestParseOpArgs:
    """parse_op_args 测试"""

    def test_numeric_args(self):
        """数值参数"""
        args = parse_op_args("ts_mean(close, 20)")
        assert len(args) == 1
        assert args[0][0] == "ts_mean"
        assert 20.0 in args[0][1]

    def test_multiple_ops(self):
        """多个算子"""
        args = parse_op_args("ts_corr(ts_mean(close, 10), volume, 20)")
        ops = [a[0] for a in args]
        assert "ts_corr" in ops
        assert "ts_mean" in ops


# ==============================================================================
# 数据结构测试
# ==============================================================================


class TestLogicCondition:
    """LogicCondition 测试"""

    def test_basic(self):
        """基本创建"""
        cond = LogicCondition(variable="close", op="ts_mean", threshold=0.5, window=20)
        assert cond.variable == "close"
        assert cond.op == "ts_mean"
        assert cond.threshold == 0.5
        assert cond.window == 20

    def test_to_dict(self):
        """序列化"""
        cond = LogicCondition(variable="close", op="ts_mean", threshold=0.5, window=20)
        d = cond.to_dict()
        assert d["variable"] == "close"
        assert d["op"] == "ts_mean"
        assert d["window"] == 20

    def test_from_dict(self):
        """反序列化"""
        d = {"variable": "close", "op": "ts_mean", "threshold": 0.5, "window": 20}
        cond = LogicCondition.from_dict(d)
        assert cond.variable == "close"
        assert cond.window == 20


class TestLogicBehavior:
    """LogicBehavior 测试"""

    def test_basic(self):
        """基本创建"""
        beh = LogicBehavior(target="forward_return_5", direction=-1, horizon=5)
        assert beh.target == "forward_return_5"
        assert beh.direction == -1
        assert beh.horizon == 5

    def test_to_dict(self):
        """序列化"""
        beh = LogicBehavior(target="forward_return_5", direction=-1, horizon=5)
        d = beh.to_dict()
        assert d["direction"] == -1

    def test_from_dict(self):
        """反序列化"""
        d = {"target": "forward_return_5", "direction": -1, "horizon": 5}
        beh = LogicBehavior.from_dict(d)
        assert beh.direction == -1


class TestWikiLogicStructured:
    """WikiLogicStructured 测试"""

    @pytest.fixture
    def sample_logic(self):
        """示例逻辑"""
        return WikiLogicStructured(
            predicates=[
                LogicCondition(variable="open", op="rank", threshold=0),
                LogicCondition(variable="volume", op="rank", threshold=0),
                LogicCondition(
                    variable="open", op="ts_corr", threshold=-0.5, window=10,
                    second_variable="volume",
                ),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            operator_whitelist=["rank", "ts_corr", "sign"],
            parameter_ranges={"ts_corr": (5, 30)},
            sign_constraint=-1,
        )

    def test_get_operators(self, sample_logic):
        """提取算子"""
        ops = sample_logic.get_operators()
        assert "rank" in ops
        assert "ts_corr" in ops

    def test_get_variables(self, sample_logic):
        """提取变量"""
        vars_ = sample_logic.get_variables()
        assert "open" in vars_
        assert "volume" in vars_

    def test_to_dict(self, sample_logic):
        """序列化"""
        d = sample_logic.to_dict()
        assert "predicates" in d
        assert "behavior" in d
        assert d["sign_constraint"] == -1

    def test_from_dict(self, sample_logic):
        """反序列化"""
        d = sample_logic.to_dict()
        logic = WikiLogicStructured.from_dict(d)
        assert len(logic.predicates) == 3
        assert logic.sign_constraint == -1


# ==============================================================================
# 编译器测试
# ==============================================================================


class TestCompileToConstraint:
    """compile_to_constraint 测试"""

    @pytest.fixture
    def price_volume_logic(self):
        """量价背离逻辑（论文示例）"""
        return WikiLogicStructured(
            predicates=[
                LogicCondition(variable="open", op="rank", threshold=0),
                LogicCondition(variable="volume", op="rank", threshold=0),
                LogicCondition(
                    variable="open", op="ts_corr", threshold=-0.5, window=10,
                    second_variable="volume",
                ),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            operator_whitelist=["rank", "ts_corr", "sign"],
            parameter_ranges={"ts_corr": (5, 30)},
            sign_constraint=-1,
        )

    def test_basic_compilation(self, price_volume_logic):
        """基本编译"""
        gamma = compile_to_constraint(price_volume_logic)
        assert isinstance(gamma, CompiledConstraint)
        assert "rank" in gamma.operator_whitelist
        assert "ts_corr" in gamma.operator_whitelist
        assert gamma.sign_constraint == -1

    def test_operator_whitelist(self, price_volume_logic):
        """算子白名单"""
        gamma = compile_to_constraint(price_volume_logic)
        assert gamma.operator_whitelist == {"rank", "ts_corr", "sign"}

    def test_variable_whitelist(self, price_volume_logic):
        """变量白名单"""
        gamma = compile_to_constraint(price_volume_logic)
        assert "open" in gamma.variable_whitelist
        assert "volume" in gamma.variable_whitelist

    def test_parameter_ranges(self, price_volume_logic):
        """参数范围"""
        gamma = compile_to_constraint(price_volume_logic)
        assert "ts_corr" in gamma.parameter_ranges
        lo, hi = gamma.parameter_ranges["ts_corr"]
        assert lo <= 10 <= hi  # 窗口 10 应该在范围内

    def test_sign_constraint(self, price_volume_logic):
        """符号约束"""
        gamma = compile_to_constraint(price_volume_logic)
        assert gamma.sign_constraint == -1

    def test_source_logic(self, price_volume_logic):
        """来源逻辑"""
        gamma = compile_to_constraint(price_volume_logic, source_logic="price_volume_divergence")
        assert gamma.source_logic == "price_volume_divergence"


class TestCompiledConstraintValidate:
    """CompiledConstraint.validate 测试"""

    @pytest.fixture
    def gamma(self):
        """示例 Γ 约束"""
        return CompiledConstraint(
            operator_whitelist={"rank", "ts_corr", "sign", "sub", "mul", "div"},
            operator_blacklist=set(),
            parameter_ranges={"ts_corr": (5, 30)},
            sign_constraint=-1,
            variable_whitelist={"open", "volume"},
        )

    def test_valid_formula(self, gamma):
        """有效公式"""
        passed, reason = gamma.validate("sign(-ts_corr(rank(open), rank(volume), 10))")
        assert passed == True
        assert reason is None

    def test_operator_not_in_whitelist(self, gamma):
        """算子不在白名单"""
        passed, reason = gamma.validate("ts_argmax(close, 5)")
        assert passed == False
        assert "not in whitelist" in reason

    def test_variable_not_in_whitelist(self, gamma):
        """变量不在白名单"""
        passed, reason = gamma.validate("rank(close)")
        assert passed == False
        assert "not in whitelist" in reason

    def test_parameter_out_of_range(self, gamma):
        """参数超出范围"""
        passed, reason = gamma.validate("ts_corr(rank(open), rank(volume), 100)")
        assert passed == False
        assert "not in" in reason

    def test_blacklisted_operator(self):
        """黑名单算子"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "ts_corr"},
            operator_blacklist={"ts_argmax"},
        )
        passed, reason = gamma.validate("ts_argmax(close, 5)")
        assert passed == False
        assert "blacklisted" in reason


class TestRenderForPrompt:
    """CompiledConstraint.render_for_prompt 测试"""

    def test_basic_render(self):
        """基本渲染"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "ts_corr"},
            variable_whitelist={"open", "volume"},
            parameter_ranges={"ts_corr": (5, 30)},
            sign_constraint=-1,
        )
        text = gamma.render_for_prompt()
        assert "Γ 约束" in text
        assert "rank" in text
        assert "ts_corr" in text
        assert "open" in text
        assert "反向" in text

    def test_with_source_logic(self):
        """带来源逻辑"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank"},
            source_logic="test_logic",
        )
        text = gamma.render_for_prompt()
        assert "test_logic" in text


# ==============================================================================
# 集成测试
# ==============================================================================


class TestIntegration:
    """集成测试"""

    def test_paper_example(self):
        """论文示例: -TS_CORR(RANK(open), RANK(volume), 10)"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="open", op="rank", threshold=0),
                LogicCondition(variable="volume", op="rank", threshold=0),
                LogicCondition(
                    variable="open", op="ts_corr", threshold=-0.5, window=10,
                    second_variable="volume",
                ),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            operator_whitelist=["rank", "ts_corr", "sign"],
            parameter_ranges={"ts_corr": (5, 30)},
            sign_constraint=-1,
        )

        gamma = compile_to_constraint(logic)

        # 验证论文示例公式
        passed, _ = gamma.validate("sign(-ts_corr(rank(open), rank(volume), 10))")
        assert passed == True

        # 验证非法算子被拒绝
        passed, _ = gamma.validate("ts_argmax(close, 5)")
        assert passed == False

        # 验证 Prompt 注入文本
        prompt_text = gamma.render_for_prompt()
        assert "rank" in prompt_text
        assert "ts_corr" in prompt_text

    def test_roundtrip_serialization(self):
        """往返序列化"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
            ],
            behavior=LogicBehavior(target="forward_return_1", direction=+1, horizon=1),
            operator_whitelist=["ts_mean", "rank"],
            parameter_ranges={"ts_mean": (5, 60)},
            sign_constraint=+1,
        )

        # 序列化
        d = logic.to_dict()

        # 反序列化
        logic2 = WikiLogicStructured.from_dict(d)

        # 验证
        assert len(logic2.predicates) == len(logic.predicates)
        assert logic2.sign_constraint == logic.sign_constraint
        assert logic2.operator_whitelist == logic.operator_whitelist

    def test_gamma_to_dict(self):
        """Γ 序列化"""
        gamma = CompiledConstraint(
            operator_whitelist={"rank", "ts_corr"},
            parameter_ranges={"ts_corr": (5, 30)},
            sign_constraint=-1,
        )
        d = gamma.to_dict()
        assert "rank" in d["operator_whitelist"]
        assert d["sign_constraint"] == -1


# ==============================================================================
# Test Class: check_sign_hint (Phase 1 红→绿回归测试)
# ==============================================================================


class TestCheckSignHint:
    """check_sign_hint 单元测试

    V8 暴露 bug: direction=-1 时, 无负向标记的公式仍被接受 (宽松兜底)。
    这导致 sign_constraint=-1 接受全正 IR 公式 (e.g. intraday_reversal)。

    修复后, direction=-1 必须有负向标记 (- / sign(- / sub(0) 才接受。
    红→绿对照: test_positive_rejected_for_direction_minus1
    """

    def test_negative_prefix_dash_accepted_for_direction_minus1(self):
        """formula 以 - 开头 + direction=-1 → True"""
        assert check_sign_hint("-ts_mean(close, 20)", -1) is True

    def test_sign_neg_accepted_for_direction_minus1(self):
        """formula 包含 sign(-...) + direction=-1 → True"""
        assert check_sign_hint("sign(-ts_corr(close, vol, 10))", -1) is True

    def test_sub_zero_accepted_for_direction_minus1(self):
        """formula 包含 sub(0, ...) + direction=-1 → True"""
        assert check_sign_hint("sub(0, ts_mean(close, 20))", -1) is True

    def test_positive_rejected_for_direction_minus1(self):
        """formula 无负向标记 + direction=-1 → False (V8 回归保护)

        修复前: 宽松兜底 return True (bug)
        修复后: return False
        """
        # rank(close) - 纯正向, 无任何负向标记
        assert check_sign_hint("rank(close)", -1) is False
        # ts_mean(close, 20) - 也是纯正向
        assert check_sign_hint("ts_mean(close, 20)", -1) is False
        # div(sub(a, b), c) - 嵌套但顶层无负向
        assert check_sign_hint("div(sub(close, ts_mean(close, 20)), std)", -1) is False

    def test_direction_plus1_always_accepted(self):
        """direction=+1 宽松模式, 任何 formula 都接受"""
        assert check_sign_hint("rank(close)", 1) is True
        assert check_sign_hint("-rank(close)", 1) is True
        assert check_sign_hint("ts_mean(close, 20)", 1) is True

    def test_direction_none_always_accepted(self):
        """direction=None 表示不约束, 任何 formula 都接受"""
        assert check_sign_hint("rank(close)", None) is True
        assert check_sign_hint("-rank(close)", None) is True
        assert check_sign_hint("any_formula(...)", None) is True

    def test_whitespace_around_neg_prefix_accepted(self):
        """formula 以 - 开头, 含空格也接受"""
        assert check_sign_hint("  -ts_mean(close, 20)", -1) is True

    def test_sign_with_space_accepted(self):
        """sign( -...) 含空格的变体也接受"""
        assert check_sign_hint("sign( -ts_corr(close, vol, 10))", -1) is True

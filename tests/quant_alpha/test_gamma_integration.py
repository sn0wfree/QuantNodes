#!/usr/bin/env python3
# coding=utf-8
"""
test_gamma_integration.py - Γ 编译器集成测试

测试 Γ 约束是否正确注入到 Alpha-GPT 工作流中。

Usage:
    python3.11 tests/quant_alpha/test_gamma_integration.py
"""

import polars as pl
from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.logic_mining.compiler import (
    compile_to_constraint,
)
from QuantNodes.research.quant_alpha.workflow import (
    AlphaGptConfig,
    AlphaGptWorkflow,
)


def test_gamma_prompt_injection():
    """测试 Γ 约束是否正确注入到 prompt 中"""
    print("=== 测试 Γ 约束 Prompt 注入 ===")

    # 定义逻辑
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

    # 编译为 Γ 约束
    gamma = compile_to_constraint(logic, source_logic="price_volume_divergence")

    # 创建配置
    config = AlphaGptConfig(
        objective="capture A-share reversal effect",
        iterations=1,
        pool_size=5,
        gamma=gamma,
    )

    # 测试 prompt 注入
    workflow = AlphaGptWorkflow(config=config)
    ideas = [{"id": "IDEA-1", "name": "test", "description": "test idea"}]
    prompt = workflow._build_formula_prompt(1, ideas, ["rank", "ts_corr"], ["open", "volume"])

    print(f"Prompt 包含 Γ 约束: {'Γ 约束' in prompt}")
    print(f"Prompt 包含 rank: {'rank' in prompt}")
    print(f"Prompt 包含 ts_corr: {'ts_corr' in prompt}")
    print(f"Prompt 包含 open: {'open' in prompt}")
    print()

    return True


def test_gamma_validation():
    """测试 Γ 约束校验"""
    print("=== 测试 Γ 约束校验 ===")

    # 定义逻辑
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

    # 编译为 Γ 约束
    gamma = compile_to_constraint(logic)

    # 测试公式
    test_cases = [
        ("sign(-ts_corr(rank(open), rank(volume), 10))", True),
        ("rank(ts_corr(open, volume, 10))", True),
        ("ts_argmax(close, 5)", False),  # ts_argmax 不在白名单
        ("rank(close)", False),  # close 不在变量白名单
    ]

    for formula, expected in test_cases:
        passed, reason = gamma.validate(formula)
        status = "✓" if passed == expected else "✗"
        print(f"  {status} {formula[:50]:50s} -> passed={passed}, expected={expected}")
        if passed != expected:
            print(f"     原因: {reason}")

    print()
    return True


def test_pipeline_config_gamma():
    """测试 PipelineConfig 中的 gamma 参数"""
    print("=== 测试 PipelineConfig gamma 参数 ===")

    from QuantNodes.research.quant_alpha.pipeline import PipelineConfig

    # 定义逻辑
    logic = WikiLogicStructured(
        predicates=[
            LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
        ],
        behavior=LogicBehavior(target="forward_return_1", direction=+1, horizon=1),
        operator_whitelist=["ts_mean", "rank"],
        parameter_ranges={"ts_mean": (5, 60)},
        sign_constraint=+1,
    )

    # 编译为 Γ 约束
    gamma = compile_to_constraint(logic)

    # 创建配置
    config = PipelineConfig(
        objective="capture momentum effect",
        gamma=gamma,
    )

    print(f"PipelineConfig.gamma 存在: {config.gamma is not None}")
    print(f"PipelineConfig.gamma 类型: {type(config.gamma).__name__}")
    print()

    return True


def test_gamma_render_for_prompt():
    """测试 render_for_prompt 输出"""
    print("=== 测试 render_for_prompt 输出 ===")

    # 定义逻辑
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

    # 编译为 Γ 约束
    gamma = compile_to_constraint(logic, source_logic="price_volume_divergence")

    # 渲染 prompt
    prompt_text = gamma.render_for_prompt()

    print("渲染的 Prompt 文本:")
    print("-" * 60)
    print(prompt_text)
    print("-" * 60)
    print()

    return True


if __name__ == "__main__":
    print("Γ 编译器集成测试")
    print("=" * 60)
    print()

    test_gamma_prompt_injection()
    test_gamma_validation()
    test_pipeline_config_gamma()
    test_gamma_render_for_prompt()

    print("所有测试完成!")

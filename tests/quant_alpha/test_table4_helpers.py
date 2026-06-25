# coding=utf-8
"""
test_table4_helpers.py - 内部辅助函数与脚本测试
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from QuantNodes.research.quant_alpha.evaluation import (
    FactorSpec,
    MockDataLoader,
    MockTable4Runner,
    PolarsAlphaCalculatorEvaluator,
)
from QuantNodes.research.quant_alpha.evaluation.baselines import (
    G1Handcrafted,
    G2LlmOnly,
    G3AlphaGpt,
)
from QuantNodes.research.quant_alpha.evaluation.baselines.g1_handcrafted import (
    _gen_formula,
    FIELDS,
    WINDOWS,
    TIME_OPS,
    CROSS_OPS,
    BINARY_OPS,
)
from QuantNodes.research.quant_alpha.evaluation.baselines.g2_llm_only import (
    INVALID_LLM_TOKENS,
)
from QuantNodes.research.quant_alpha.evaluation.contracts import (
    Baseline,
    DataLoader,
    Evaluator,
    FactorMetrics,
    Table4GroupResult,
    Table4Report,
    Table4Runner,
)


class TestGenFormulaHelper:
    """测试 _gen_formula 内部辅助函数"""

    def test_gen_formula_returns_string(self):
        rng = random.Random(42)
        formula = _gen_formula(rng)
        assert isinstance(formula, str)
        assert len(formula) > 0

    def test_gen_formula_uses_valid_chars(self):
        """生成的公式应仅含合法字符"""
        import re as _re
        rng = random.Random(42)
        for _ in range(50):
            f = _gen_formula(rng)
            # 允许的字符: 字母数字下划线括号逗号点号负号 + 空格
            assert _re.match(r"^[A-Za-z0-9_\-(),.\s]+$", f), f"Invalid chars: {f}"

    def test_gen_formula_max_depth_1(self):
        """max_depth=1 生成简单公式（无双层 binary 嵌套）"""
        rng = random.Random(42)
        for _ in range(50):
            f = _gen_formula(rng, max_depth=1)
            # max_depth=1 时不应有 binary op 嵌套（如 Add(Add(...), ...)）
            # 简单判定：没有两个 binary op 串接
            binary_count = sum(1 for op in BINARY_OPS if op in f)
            # 简单公式中最多 1 个 binary op（外层）
            assert binary_count <= 1, f"Too many binary ops: {f}"

    def test_gen_formula_max_depth_2(self):
        """max_depth=2 可生成嵌套"""
        rng = random.Random(42)
        has_nested = False
        for _ in range(50):
            f = _gen_formula(rng, max_depth=2)
            if f.count("(") >= 3:  # 双层嵌套至少 3 个 (
                has_nested = True
                break
        assert has_nested

    def test_constants_defined(self):
        """基础常量已定义"""
        assert "close" in FIELDS
        assert "vol" in FIELDS
        assert 5 in WINDOWS
        assert 20 in WINDOWS
        assert "ts_mean" in TIME_OPS
        assert "Add" in BINARY_OPS


class TestInvalidLLMTokens:
    """测试 G2 invalid token 列表"""

    def test_invalid_tokens_defined(self):
        assert len(INVALID_LLM_TOKENS) > 0
        # 至少 5 个 invalid tokens
        assert len(INVALID_LLM_TOKENS) >= 5

    def test_invalid_tokens_contain_known_unsupported(self):
        """确认包含已知不被 parser 支持的算子"""
        all_invalid = " ".join(INVALID_LLM_TOKENS)
        assert "rank(" in all_invalid  # 跨截面 rank 不支持
        assert "IndNeutralize" in all_invalid
        assert "log(" in all_invalid  # log(vol) 单变量 log 不支持


# ---------------------------------------------------------------------------
# 测试 __init__.py 完整导出
# ---------------------------------------------------------------------------


class TestPackageExports:
    """验证 evaluation 子包对外暴露的接口完整"""

    def test_all_4_baselines_exported(self):
        from QuantNodes.research.quant_alpha.evaluation.baselines import (
            G1Handcrafted,
            G2LlmOnly,
            G3AlphaGpt,
        )
        assert G1Handcrafted is not None
        assert G2LlmOnly is not None
        assert G3AlphaGpt is not None

    def test_all_4_contracts_exported(self):
        from QuantNodes.research.quant_alpha.evaluation import (
            DataLoader,
            Evaluator,
            Baseline,
            Table4Runner,
        )
        for cls in (DataLoader, Evaluator, Baseline, Table4Runner):
            assert hasattr(cls, "__abstractmethods__")

    def test_all_4_dataclasses_exported(self):
        from QuantNodes.research.quant_alpha.evaluation import (
            FactorSpec,
            FactorMetrics,
            Table4GroupResult,
            Table4Report,
        )
        for cls in (FactorSpec, FactorMetrics, Table4GroupResult, Table4Report):
            assert hasattr(cls, "__dataclass_fields__")

    def test_main_evaluation_exports(self):
        from QuantNodes.research.quant_alpha.evaluation import (
            MockDataLoader,
            MockTable4Runner,
            PolarsAlphaCalculatorEvaluator,
            MOCK_INDUSTRIES,
        )
        assert MockDataLoader is not None
        assert MockTable4Runner is not None
        assert PolarsAlphaCalculatorEvaluator is not None
        assert isinstance(MOCK_INDUSTRIES, list)
        assert len(MOCK_INDUSTRIES) >= 5


# ---------------------------------------------------------------------------
# Runner 多 baseline 一致性
# ---------------------------------------------------------------------------


class TestRunnerMultiBaseline:
    def test_three_baselines_different_groups(self, tmp_path):
        """3 个 baseline 输出 3 个不同 group"""
        from QuantNodes.research.quant_alpha.evaluation import (
            MockDataLoader,
            MockTable4Runner,
            PolarsAlphaCalculatorEvaluator,
        )

        loader = MockDataLoader(n_stocks=10, n_days=20, seed=42)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[
                G1Handcrafted(n=3, seed=42),
                G2LlmOnly(n=3, seed=42),
                G3AlphaGpt(n=3, iterations=1, pool_size=2, seed=42),
            ],
            output_dir=tmp_path,
        )
        report = runner.run()
        assert len(report.groups) == 3
        group_names = [g.group_name for g in report.groups]
        assert group_names == ["G1_Handcrafted", "G2_LlmOnly", "G3_AlphaGpt"]

    def test_report_timestamp_iso_format(self, tmp_path):
        """timestamp 应为 ISO 8601 格式"""
        import re
        from QuantNodes.research.quant_alpha.evaluation import (
            MockDataLoader,
            MockTable4Runner,
            PolarsAlphaCalculatorEvaluator,
        )

        loader = MockDataLoader(n_stocks=5, n_days=10)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[G2LlmOnly(n=2, seed=42)],
            output_dir=tmp_path,
        )
        report = runner.run()
        # ISO 8601: 2024-01-01T00:00:00+00:00
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            report.timestamp,
        ), f"Bad timestamp: {report.timestamp}"


# ---------------------------------------------------------------------------
# FactorSpec / FactorMetrics 边界
# ---------------------------------------------------------------------------


class TestFactorSpecEdges:
    def test_formula_with_special_chars(self):
        """公式含特殊字符（负号, 下划线）"""
        fs = FactorSpec(
            formula_id="f1",
            formula="Sub(close, ts_mean(vol_5, 10))",
            source="g1",
        )
        d = fs.to_dict()
        assert d["formula"] == "Sub(close, ts_mean(vol_5, 10))"

    def test_empty_meta(self):
        """meta 默认空 dict"""
        fs = FactorSpec(formula_id="f1", formula="close", source="g1")
        assert fs.meta == {}
        # to_dict 应保留
        d = fs.to_dict()
        assert d["meta"] == {}

    def test_meta_with_complex_value(self):
        """meta 支持复杂值"""
        fs = FactorSpec(
            formula_id="f1",
            formula="close",
            source="g1",
            meta={"rank": 3, "tags": ["a", "b"], "nested": {"k": 1}},
        )
        d = fs.to_dict()
        assert d["meta"]["rank"] == 3
        assert d["meta"]["tags"] == ["a", "b"]
        assert d["meta"]["nested"] == {"k": 1}


class TestFactorMetricsEdges:
    def test_ic_decay_with_int_keys(self):
        """ic_decay 接受 int key"""
        from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
        fm = FactorMetrics(
            formula_id="f1",
            status="success",
            ic_decay={1: 0.05, 5: 0.03, 20: 0.01},
        )
        d = fm.to_dict()
        assert d["metrics"]["ic_decay"]["1"] == 0.05
        assert d["metrics"]["ic_decay"]["5"] == 0.03

    def test_factor_metrics_to_dict_keeps_error_msg(self):
        """error_msg 保留"""
        from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
        fm = FactorMetrics(
            formula_id="f1",
            status="failed",
            error_msg="Cannot parse: rank(close)",
        )
        d = fm.to_dict()
        assert d["error_msg"] == "Cannot parse: rank(close)"

    def test_factor_metrics_from_alpha_evaluate_partial(self):
        """from_alpha_evaluate 处理部分字段缺失"""
        from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
        eval_dict = {"status": "success"}  # 无 metrics 字段
        fm = FactorMetrics.from_alpha_evaluate("f1", eval_dict)
        assert fm.status == "success"
        assert fm.ic_mean == 0.0
        assert fm.ir == 0.0
        assert fm.ic_decay == {}
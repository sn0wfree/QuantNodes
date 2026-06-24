# coding=utf-8
"""
test_table4_contracts.py - 接口契约 dataclass + ABC 测试
"""

from __future__ import annotations

import pytest

from QuantNodes.research.quant_alpha.evaluation import (
    Baseline,
    DataLoader,
    Evaluator,
    FactorMetrics,
    FactorSpec,
    Table4GroupResult,
    Table4Report,
    Table4Runner,
)


class TestFactorSpec:
    def test_minimal_construct(self):
        fs = FactorSpec(formula_id="f1", formula="rank(close)", source="g1")
        assert fs.formula_id == "f1"
        assert fs.source == "g1"
        assert fs.category == "unknown"
        assert fs.complexity == 0
        assert fs.meta == {}

    def test_to_dict_roundtrip(self):
        fs = FactorSpec(
            formula_id="f1",
            formula="ts_mean(close, 5)",
            source="g1",
            category="momentum",
            complexity=2,
        )
        d = fs.to_dict()
        assert d["formula_id"] == "f1"
        assert d["category"] == "momentum"
        assert d["complexity"] == 2


class TestFactorMetrics:
    def test_default_values(self):
        fm = FactorMetrics(formula_id="f1", status="success")
        assert fm.ic_mean == 0.0
        assert fm.ir == 0.0
        assert fm.ic_decay == {}
        assert fm.error_msg is None

    def test_from_alpha_evaluate(self):
        eval_dict = {
            "status": "success",
            "metrics": {
                "ic_mean": 0.05,
                "ic_std": 0.1,
                "ir": 0.5,
                "ic_decay": {"1": 0.05, "5": 0.03},
            },
        }
        fm = FactorMetrics.from_alpha_evaluate("f1", eval_dict)
        assert fm.status == "success"
        assert abs(fm.ic_mean - 0.05) < 1e-9
        assert fm.ir == 0.5
        assert fm.ic_decay[1] == 0.05
        assert fm.ic_decay[5] == 0.03

    def test_from_alpha_evaluate_failed(self):
        eval_dict = {"status": "failed", "error_msg": "parse error"}
        fm = FactorMetrics.from_alpha_evaluate("f1", eval_dict)
        assert fm.status == "failed"
        assert fm.error_msg == "parse error"


class TestTable4GroupResult:
    def test_empty_group(self):
        g = Table4GroupResult(group_name="G1")
        assert g.success_count == 0
        assert g.avg_ir == 0.0
        assert g.best_ir == 0.0

    def test_avg_ir_calculation(self):
        from QuantNodes.research.quant_alpha.evaluation.contracts import (
            FactorMetrics,
            FactorSpec,
        )
        fs1 = FactorSpec(formula_id="f1", formula="x", source="g1")
        fs2 = FactorSpec(formula_id="f2", formula="y", source="g1")
        fm1 = FactorMetrics(formula_id="f1", status="success", ir=0.4)
        fm2 = FactorMetrics(formula_id="f2", status="success", ir=0.6)
        fm_failed = FactorMetrics(formula_id="f3", status="failed", ir=-1.0)
        g = Table4GroupResult(
            group_name="G1",
            factors=[fs1, fs2],
            metrics=[fm1, fm2, fm_failed],
        )
        assert g.avg_ir == pytest.approx(0.5)  # (0.4 + 0.6) / 2
        assert g.best_ir == 0.6
        assert g.success_count == 2
        assert g.failed_count == 1


class TestTable4Report:
    def test_rank_groups_by_ir(self):
        g1 = Table4GroupResult(
            group_name="G1",
            metrics=[FactorMetrics(formula_id="f1", status="success", ir=0.3)],
        )
        g3 = Table4GroupResult(
            group_name="G3",
            metrics=[FactorMetrics(formula_id="f1", status="success", ir=0.6)],
        )
        g2 = Table4GroupResult(
            group_name="G2",
            metrics=[FactorMetrics(formula_id="f1", status="success", ir=0.4)],
        )
        report = Table4Report(timestamp="2024-01-01", stage="mock")
        report.add_group(g1)
        report.add_group(g3)
        report.add_group(g2)
        ranked = report.rank_groups_by_ir()
        assert [g.group_name for g in ranked] == ["G3", "G2", "G1"]


class TestABCs:
    """验证 4 个 ABC 均为抽象类"""

    @pytest.mark.parametrize(
        "abc_cls",
        [DataLoader, Evaluator, Baseline, Table4Runner],
    )
    def test_abc_is_abstract(self, abc_cls):
        assert abc_cls.__abstractmethods__
        with pytest.raises(TypeError):
            abc_cls()  # 无法实例化抽象类
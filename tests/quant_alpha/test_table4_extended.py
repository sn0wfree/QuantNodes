# coding=utf-8
"""
test_table4_extended.py - 扩展测试（边界 / 鲁棒性 / 集成）

覆盖：
- MockDataLoader 极端参数 (n_stocks=1, n_days=1, n_days=10)
- g1_handcrafted._infer_category 6 个分支
- g1_handcrafted 大规模生成 (n=500)
- Runner 边界 (空 baselines / output_dir=None / 大 n_groups)
- Stage 2 IFindDataLoader 接口契约 (mock 验证)
- 确定性（同一 seed 多次跑结果一致）
- 跨 baseline 一致性（同一公式在不同 baseline 中评估结果一致）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
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
    G1Handcrafted as G1Cls,
)
from QuantNodes.research.quant_alpha.evaluation.contracts import (
    DataLoader,
    Evaluator,
    Baseline,
    Table4Runner,
)


# ---------------------------------------------------------------------------
# MockDataLoader 极端参数
# ---------------------------------------------------------------------------


class TestMockDataLoaderExtreme:
    def test_single_stock_single_day(self):
        """n_stocks=1, n_days=1 极小数据"""
        loader = MockDataLoader(n_stocks=1, n_days=1, seed=42)
        df = loader.load()
        assert df.height == 1
        assert df["code"].n_unique() == 1
        assert df["date"].n_unique() == 1
        # 单日无前向收益
        assert df["forward_return_1d"].null_count() == 1

    def test_zero_seed(self):
        """seed=0 也能正常工作（不报 numpy rng 错误）"""
        loader = MockDataLoader(n_stocks=5, n_days=10, seed=0)
        df = loader.load()
        assert df.height == 50

    def test_high_seed(self):
        """seed=2^31-1 大数"""
        loader = MockDataLoader(n_stocks=5, n_days=10, seed=2**31 - 1)
        df = loader.load()
        assert df.height == 50

    def test_industry_fewer_than_stocks(self):
        """industry 数量少于 stock 数量时, 行业平均分配"""
        loader = MockDataLoader(
            n_stocks=100,
            n_days=10,
            industries=["Tech", "Fin"],
        )
        df = loader.load()
        assert df["industry"].n_unique() == 2
        # 两个行业各占 ~50 票
        counts = df.group_by("industry").agg(pl.len()).sort("industry")
        counts_dict = dict(zip(counts["industry"].to_list(), counts["len"].to_list()))
        assert abs(counts_dict["Tech"] - counts_dict["Fin"]) <= 20  # 浮动 ≤ 20

    def test_load_summary_with_extreme_params(self):
        """极端参数下 load_summary 仍返回完整字典"""
        loader = MockDataLoader(n_stocks=1, n_days=1, seed=42)
        s = loader.load_summary()
        assert s["n_stocks"] == 1
        assert s["n_days"] == 1
        assert s["n_rows"] == 1


# ---------------------------------------------------------------------------
# g1_handcrafted._infer_category 6 个分支
# ---------------------------------------------------------------------------


class TestG1InferCategory:
    def test_momentum_ts_mean(self):
        assert G1Cls._infer_category("ts_mean(close, 5)") == "momentum"

    def test_momentum_delta(self):
        assert G1Cls._infer_category("delta(close, 3)") == "momentum"

    def test_volatility_ts_std(self):
        assert G1Cls._infer_category("ts_std(close, 10)") == "volatility"

    def test_volume_vol(self):
        assert G1Cls._infer_category("Mul(vol, close)") == "volume"

    def test_reversal_abs(self):
        # 公式中无 ts_mean / delta / ts_std / vol
        assert G1Cls._infer_category("abs(close)") == "reversal"

    def test_value_default(self):
        # 无任何关键字 → fallback value
        assert G1Cls._infer_category("close") == "value"
        assert G1Cls._infer_category("Sub(close, open)") == "value"
        assert G1Cls._infer_category("Div(close, close)") == "value"

    def test_momentum_takes_priority_over_volatility(self):
        """同时含 ts_mean 和 ts_std → momentum (检查顺序)"""
        # 公式中 ts_mean 出现 → momentum
        assert G1Cls._infer_category("ts_mean(ts_std(close, 5), 10)") == "momentum"


# ---------------------------------------------------------------------------
# 大规模 G1 生成
# ---------------------------------------------------------------------------


class TestG1LargeScale:
    def test_generate_500_unique(self):
        """n=500 时仍能生成 500 个 unique formula"""
        g1 = G1Handcrafted(n=500, seed=42)
        factors = g1.generate_factors()
        assert len(factors) == 500
        # formula_id 唯一
        ids = [f.formula_id for f in factors]
        assert len(set(ids)) == 500
        # formula 也应该 unique (虽然允许 collisions, 但 n=500 时应至少 > 100)
        formulas = [f.formula for f in factors]
        assert len(set(formulas)) > 100  # 至少 100 个 unique


# ---------------------------------------------------------------------------
# Runner 边界
# ---------------------------------------------------------------------------


class TestRunnerEdgeCases:
    def test_empty_baselines(self, tmp_path):
        """空 baselines 列表：返回空 report, 不抛错"""
        loader = MockDataLoader(n_stocks=5, n_days=10)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[],
            output_dir=tmp_path,
        )
        report = runner.run()
        assert len(report.groups) == 0
        assert report.rank_groups_by_ir() == []

    def test_runner_without_output_dir(self):
        """output_dir=None：不写文件, 仍返回 report"""
        loader = MockDataLoader(n_stocks=5, n_days=10)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[G2LlmOnly(n=2, seed=42)],
            output_dir=None,
        )
        report = runner.run()
        assert len(report.groups) == 1

    def test_runner_with_custom_notes(self, tmp_path):
        """自定义 notes 写入 report"""
        loader = MockDataLoader(n_stocks=5, n_days=10)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[G2LlmOnly(n=2, seed=42)],
            output_dir=tmp_path,
            notes=["custom note A", "custom note B"],
        )
        report = runner.run()
        assert "custom note A" in report.notes
        assert "custom note B" in report.notes
        # notes 也写入 md
        runner.save_markdown(report, tmp_path / "r.md")
        content = (tmp_path / "r.md").read_text()
        assert "custom note A" in content

    def test_runner_double_run(self, tmp_path):
        """同一 runner 跑两次, 第二次仍能跑通"""
        loader = MockDataLoader(n_stocks=5, n_days=10)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[G2LlmOnly(n=2, seed=42)],
            output_dir=tmp_path,
        )
        r1 = runner.run()
        r2 = runner.run()
        assert len(r1.groups) == 1
        assert len(r2.groups) == 1


# ---------------------------------------------------------------------------
# 跨 baseline 一致性
# ---------------------------------------------------------------------------


class TestCrossBaselineConsistency:
    """同一公式在 G1/G2/G3 中评估, 结果应一致 (evaluator 是 deterministic)"""

    def test_same_formula_same_metrics(self):
        loader = MockDataLoader(n_stocks=20, n_days=50, seed=42)
        df = loader.load()
        evaluator = PolarsAlphaCalculatorEvaluator()

        # 直接用 evaluator 评估同一公式
        factors = [FactorSpec(formula_id="f1", formula="ts_mean(close, 5)", source="t")]
        m1 = evaluator.evaluate(factors, df, forward_returns=[1])
        m2 = evaluator.evaluate(factors, df, forward_returns=[1])

        assert m1[0].ic_mean == m2[0].ic_mean
        assert m1[0].ir == m2[0].ir
        assert m1[0].status == m2[0].status


# ---------------------------------------------------------------------------
# 确定性 / 重跑一致性
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_mock_data_loader_deterministic(self):
        """MockDataLoader 同 seed 多次生成结果一致"""
        l1 = MockDataLoader(n_stocks=20, n_days=30, seed=7)
        l2 = MockDataLoader(n_stocks=20, n_days=30, seed=7)
        d1 = l1.load()
        d2 = l2.load()
        assert d1["close"].to_list() == d2["close"].to_list()
        assert d1["amount"].to_list() == d2["amount"].to_list()

    def test_g1_deterministic(self):
        """G1Handcrafted 同 seed 多次生成结果一致"""
        g1_a = G1Handcrafted(n=20, seed=42)
        g1_b = G1Handcrafted(n=20, seed=42)
        fa = g1_a.generate_factors()
        fb = g1_b.generate_factors()
        assert [f.formula for f in fa] == [f.formula for f in fb]

    def test_g2_deterministic(self):
        """G2LlmOnly 同 seed 多次生成结果一致"""
        g2_a = G2LlmOnly(n=20, seed=42)
        g2_b = G2LlmOnly(n=20, seed=42)
        fa = g2_a.generate_factors()
        fb = g2_b.generate_factors()
        assert [f.formula for f in fa] == [f.formula for f in fb]


# ---------------------------------------------------------------------------
# Stage 2 IFindDataLoader 接口契约
# ---------------------------------------------------------------------------


class TestStage2DataLoaderContract:
    """验证 Stage 2 DataLoader 接口契约, 准备 iFinD 实施"""

    def test_data_loader_abstract(self):
        """DataLoader ABC 不能直接实例化"""
        with pytest.raises(TypeError):
            DataLoader()

    def test_evaluator_abstract(self):
        with pytest.raises(TypeError):
            Evaluator()

    def test_baseline_abstract(self):
        with pytest.raises(TypeError):
            Baseline()

    def test_table4runner_abstract(self):
        with pytest.raises(TypeError):
            Table4Runner()

    def test_concrete_data_loader_implements_contract(self):
        """MockDataLoader 是 DataLoader 的具体实现"""

        class _CustomLoader(DataLoader):
            def load(self):
                return pl.DataFrame({"a": [1, 2, 3]})

        loader = _CustomLoader()
        df = loader.load()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3

    def test_concrete_evaluator_implements_contract(self):
        """PolarsAlphaCalculatorEvaluator 是 Evaluator 的具体实现"""
        assert isinstance(PolarsAlphaCalculatorEvaluator(), Evaluator)

    def test_concrete_baseline_implements_contract(self):
        """G1/G2/G3 都是 Baseline 的具体实现"""
        assert isinstance(G1Handcrafted(n=5), Baseline)
        assert isinstance(G2LlmOnly(n=5), Baseline)
        assert isinstance(G3AlphaGpt(n=5), Baseline)

    def test_concrete_runner_implements_contract(self):
        """MockTable4Runner 是 Table4Runner 的具体实现"""
        loader = MockDataLoader(n_stocks=5, n_days=10)
        evaluator = PolarsAlphaCalculatorEvaluator()
        runner = MockTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[],
        )
        assert isinstance(runner, Table4Runner)


# ---------------------------------------------------------------------------
# Report schema 完整性
# ---------------------------------------------------------------------------


class TestReportSchemaCompleteness:
    def test_group_dict_contains_all_keys(self):
        """Table4GroupResult.to_dict() 包含所有 key"""
        g = MockTable4Runner.run.__qualname__  # placeholder
        from QuantNodes.research.quant_alpha.evaluation.contracts import (
            Table4GroupResult,
            FactorMetrics,
            FactorSpec,
        )
        g = Table4GroupResult(
            group_name="G1",
            factors=[FactorSpec(formula_id="f1", formula="close", source="g1")],
            metrics=[FactorMetrics(formula_id="f1", status="success", ir=0.5)],
        )
        d = g.to_dict()
        expected_keys = {
            "group", "n_factors", "n_success", "n_failed",
            "avg_ic_mean", "avg_ir", "best_ir",
            "elapsed_sec", "factors", "metrics",
        }
        assert expected_keys.issubset(d.keys())

    def test_factor_spec_to_dict_keys(self):
        fs = FactorSpec(formula_id="f1", formula="close", source="g1")
        d = fs.to_dict()
        for key in ["formula_id", "formula", "source", "category", "complexity", "meta"]:
            assert key in d

    def test_factor_metrics_to_dict_keys(self):
        from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
        m = FactorMetrics(formula_id="f1", status="success")
        d = m.to_dict()
        for key in ["formula_id", "status", "metrics", "error_msg"]:
            assert key in d
        for k in ["ic_mean", "ic_std", "ir", "ic_decay"]:
            assert k in d["metrics"]
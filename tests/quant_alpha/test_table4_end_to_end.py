# coding=utf-8
"""
test_table4_end_to_end.py - Table 4 pipeline 端到端测试
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from QuantNodes.research.quant_alpha.evaluation import (
    MockDataLoader,
    MockTable4Runner,
    PolarsAlphaCalculatorEvaluator,
)
from QuantNodes.research.quant_alpha.evaluation.baselines import (
    G1Handcrafted,
    G2LlmOnly,
    G3AlphaGpt,
)


@pytest.fixture(scope="module")
def tiny_runner(tmp_path_factory):
    """生成一个 20 票 × 50 日的小 pipeline，端到端跑通"""
    out_dir = tmp_path_factory.mktemp("table4_tiny")
    loader = MockDataLoader(n_stocks=20, n_days=50, seed=42)
    evaluator = PolarsAlphaCalculatorEvaluator()
    baselines = [
        G1Handcrafted(n=5, seed=42),
        G2LlmOnly(n=5, seed=43),
        G3AlphaGpt(n=5, iterations=1, pool_size=3, seed=44),
    ]
    return MockTable4Runner(
        loader=loader,
        evaluator=evaluator,
        baselines=baselines,
        output_dir=out_dir,
        stage="test",
    )


class TestMockTable4Runner:
    def test_runner_returns_report(self, tiny_runner):
        report = tiny_runner.run()
        assert report.stage == "test"
        assert len(report.groups) == 3

    def test_group_names(self, tiny_runner):
        report = tiny_runner.run()
        names = {g.group_name for g in report.groups}
        assert names == {"G1_Handcrafted", "G2_LlmOnly", "G3_AlphaGpt"}

    def test_each_group_has_factors(self, tiny_runner):
        report = tiny_runner.run()
        for g in report.groups:
            assert len(g.factors) > 0
            assert len(g.metrics) == len(g.factors)

    def test_save_json(self, tiny_runner):
        report = tiny_runner.run()
        json_path = tiny_runner.output_dir / "test_report.json"
        tiny_runner.save_json(report, json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["stage"] == "test"
        assert len(data["groups"]) == 3

    def test_save_markdown(self, tiny_runner):
        report = tiny_runner.run()
        md_path = tiny_runner.output_dir / "test_report.md"
        tiny_runner.save_markdown(report, md_path)
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "# Table 4 复现报告" in content
        assert "G1_Handcrafted" in content
        assert "G3_AlphaGpt" in content

    def test_rank_groups_method(self, tiny_runner):
        report = tiny_runner.run()
        ranked = report.rank_groups_by_ir()
        assert len(ranked) == 3
        # 排名按 avg_ir 降序
        for i in range(len(ranked) - 1):
            assert ranked[i].avg_ir >= ranked[i + 1].avg_ir


class TestTable4ReportSchema:
    """验证 Table4Report 输出 schema 稳定（外部依赖如 nanobot WebUI 依赖此）"""

    def test_to_dict_structure(self, tiny_runner):
        report = tiny_runner.run()
        d = report.to_dict()
        assert "timestamp" in d
        assert "stage" in d
        assert "summary" in d
        assert "groups" in d
        assert "paper_comparison" in d
        assert "notes" in d

    def test_group_summary_aggregation(self, tiny_runner):
        report = tiny_runner.run()
        for g in report.groups:
            d = g.to_dict()
            assert "group" in d
            assert "n_factors" in d
            assert "n_success" in d
            assert "n_failed" in d
            assert "avg_ic_mean" in d
            assert "avg_ir" in d
            assert "best_ir" in d
            assert "elapsed_sec" in d
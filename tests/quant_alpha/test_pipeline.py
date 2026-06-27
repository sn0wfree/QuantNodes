# coding=utf-8
"""Tests for AlphaPipeline - 端到端因子挖掘流水线。

覆盖：
- PipelineConfig: 配置初始化
- PipelineResult: 结果结构
- AlphaPipeline: 流水线运行
- CLI: 命令注册
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.pipeline import (
    AlphaPipeline,
    PipelineConfig,
    PipelineResult,
)
from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成测试数据"""
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            close = float(np.random.randn() * 5 + 100)
            rows.append({
                "date": date,
                "code": code,
                "close": close,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "vol": 1000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """测试用 PipelineConfig"""
    return PipelineConfig(
        objective="test reversal",
        wiki_path=str(tmp_path / "wiki"),
        alphagpt_iterations=1,
        alphagpt_pool_size=2,
        mcts_iterations=10,
        mcts_max_depth=3,
        max_mutual_ic=0.7,
        top_k=5,
    )


@pytest.fixture
def pipeline(pipeline_config: PipelineConfig) -> AlphaPipeline:
    """AlphaPipeline 实例"""
    return AlphaPipeline(pipeline_config)


# ==============================================================================
# TestPipelineConfig
# ==============================================================================


class TestPipelineConfig:
    """PipelineConfig 测试"""

    def test_default_values(self):
        """默认值"""
        config = PipelineConfig(objective="test")
        assert config.objective == "test"
        assert config.wiki_path == "wiki/"
        assert config.alphagpt_iterations == 3
        assert config.alphagpt_pool_size == 10
        assert config.mcts_iterations == 50
        assert config.mcts_max_depth == 5
        assert config.max_mutual_ic == 0.7
        assert config.top_k == 10
        assert config.date_column == "date"
        assert config.code_column == "code"
        assert config.forward_returns == (1, 5, 20)
        assert config.llm_provider == "minimax"
        assert config.temperature == 0.7

    def test_custom_values(self, tmp_path: Path):
        """自定义值"""
        config = PipelineConfig(
            objective="test",
            wiki_path=str(tmp_path / "custom"),
            alphagpt_iterations=5,
            mcts_iterations=100,
            top_k=20,
        )
        assert config.alphagpt_iterations == 5
        assert config.mcts_iterations == 100
        assert config.top_k == 20


# ==============================================================================
# TestPipelineResult
# ==============================================================================


class TestPipelineResult:
    """PipelineResult 测试"""

    def test_initial_state(self):
        """初始状态"""
        result = PipelineResult()
        assert result.rounds == []
        assert result.final_pool == []
        assert result.wiki_pages == []
        assert result.all_mcts_nodes == []
        assert result.global_correlation_matrix == {}
        assert result.elapsed_seconds == 0.0
        assert result.summary == {}

    def test_field_mutation(self):
        """字段可变"""
        result = PipelineResult()
        result.final_pool = [
            FactorMetrics(formula_id="f1", status="success", ir=0.8),
        ]
        result.wiki_pages = ["page1"]
        result.elapsed_seconds = 10.5
        result.summary = {"final_factors": 1}

        assert len(result.final_pool) == 1
        assert result.wiki_pages == ["page1"]
        assert result.elapsed_seconds == 10.5
        assert result.summary["final_factors"] == 1


# ==============================================================================
# TestAlphaPipeline
# ==============================================================================


class TestAlphaPipeline:
    """AlphaPipeline 测试"""

    def test_initialization(self, pipeline_config):
        """初始化"""
        pipeline = AlphaPipeline(pipeline_config)
        assert pipeline.config == pipeline_config
        assert pipeline.wiki is not None

    def test_extract_seed_formulas_none(self, pipeline):
        """提取种子公式 - None 输入"""
        result = pipeline._extract_seed_formulas(None)
        assert result is None

    def test_extract_seed_formulas_empty(self, pipeline):
        """提取种子公式 - 空结果"""
        from QuantNodes.research.quant_alpha.workflow import AlphaGptResult

        alphagpt_result = AlphaGptResult(
            objective="test",
            iterations_completed=1,
            total_formulas=0,
            final_pool=[],
            summary={},
            elapsed_seconds=0.0,
        )
        result = pipeline._extract_seed_formulas(alphagpt_result)
        assert result == []

    def test_merge_and_dedup_empty(self, pipeline, sample_data):
        """合并去重 - 空输入"""
        result = pipeline._merge_and_dedup(None, None, sample_data)
        assert result == []

    def test_to_wiki_factor(self, pipeline):
        """转换为 WikiFactor"""
        from QuantNodes.research.wiki import FactorSource

        metrics = FactorMetrics(
            formula_id="test_formula",
            status="success",
            ic_mean=0.05,
            ir=0.8,
            rank_ic_mean=0.04,
        )
        wiki_factor = pipeline._to_wiki_factor(metrics)
        assert wiki_factor.name == "test_formula"
        assert wiki_factor.source == FactorSource.AUTO_RESEARCH
        assert wiki_factor.ic_mean == 0.05
        assert wiki_factor.icir == 0.8

    def test_build_summary(self, pipeline):
        """构建摘要"""
        result = PipelineResult()
        result.final_pool = [
            FactorMetrics(formula_id="f1", status="success", ir=0.8),
            FactorMetrics(formula_id="f2", status="success", ir=0.6),
        ]
        result.wiki_pages = ["page1"]
        result.elapsed_seconds = 10.0

        summary = pipeline._build_summary(result)
        assert summary["objective"] == "test reversal"
        assert summary["final_factors"] == 2
        assert summary["wiki_pages"] == 1
        assert summary["best_ir"] == 0.8
        assert summary["avg_ir"] == 0.7


# ==============================================================================
# TestCLI
# ==============================================================================


class TestCLI:
    """CLI 测试"""

    def test_command_registered(self):
        """命令已注册"""
        from QuantNodes.cli.commands.alpha import AlphaPipelineCommand
        cmd = AlphaPipelineCommand()
        assert cmd.name == "alpha-pipeline"

    def test_command_help(self):
        """命令帮助"""
        from QuantNodes.cli.commands.alpha import AlphaPipelineCommand
        cmd = AlphaPipelineCommand()
        assert cmd.name == "alpha-pipeline"
        assert "端到端" in cmd.description

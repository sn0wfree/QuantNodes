# coding=utf-8
"""
test_logic_driven_pipeline_coverage.py - 补 logic_driven_pipeline.py 覆盖 (Phase D.2)

目标: logic_driven_pipeline.py 50% → 80%+
- 覆盖 run() 主方法 (lines 185-255)
- 覆盖 _run_alphalogics (lines 257-278)
- 覆盖 _enhance_with_mcts (lines 280-326)
- 覆盖 _run_standard_pipeline (lines 328-359)
- 覆盖 _persist_to_wiki (lines 361-394)
- 覆盖 _build_summary (line 396-)
"""
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.logic_driven_pipeline import (
    LogicDrivenPipeline,
    LogicDrivenPipelineConfig,
    LogicDrivenPipelineResult,
)
from QuantNodes.research.quant_alpha.pipeline import PipelineConfig


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    np.random.seed(42)
    rows = []
    for d in range(30):
        for s in ["A", "B", "C"]:
            rows.append({
                "date": f"2024-01-{d + 1:02d}",
                "code": s,
                "close": 100.0 + d * 0.5 + np.random.randn() * 2,
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "vol": 1000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def default_config() -> LogicDrivenPipelineConfig:
    return LogicDrivenPipelineConfig(
        objective="test",
        logic_driven=True,
    )


@pytest.fixture
def disabled_config() -> LogicDrivenPipelineConfig:
    return LogicDrivenPipelineConfig(
        objective="test",
        logic_driven=False,
    )


# ==============================================================================
# Test Class 1: __init__
# ==============================================================================


class TestInit:
    def test_init_stores_config(self, default_config):
        pipeline = LogicDrivenPipeline(config=default_config)
        assert pipeline.config is default_config
        assert pipeline.llm_client is None

    def test_init_with_llm_client(self, default_config):
        mock_llm = MagicMock()
        pipeline = LogicDrivenPipeline(config=default_config, llm_client=mock_llm)
        assert pipeline.llm_client is mock_llm


# ==============================================================================
# Test Class 2: _run_alphalogics
# ==============================================================================


class TestRunAlphalogics:
    """_run_alphalogics() 转换 LogicDrivenPipelineConfig → AlphaLogicsConfig 并跑"""

    def test_creates_alphalogics_workflow_and_runs(self, default_config, sample_data):
        """应创建 AlphaLogicsWorkflow 并 run()"""
        mock_alphalogics_result = MagicMock()
        mock_alphalogics_result.best_logic = MagicMock()
        mock_alphalogics_result.best_evidence = MagicMock()
        mock_alphalogics_result.inner_results = []

        # patch 模块 namespace 绑定, 不是源
        with patch("QuantNodes.research.quant_alpha.logic_driven_pipeline.AlphaLogicsWorkflow") as mock_wf_class:
            mock_wf = MagicMock()
            mock_wf.run = MagicMock(return_value=mock_alphalogics_result)
            mock_wf_class.return_value = mock_wf

            pipeline = LogicDrivenPipeline(config=default_config)
            result = pipeline._run_alphalogics(sample_data)

        assert result is mock_alphalogics_result
        # 验证配置转换: LogicDrivenPipelineConfig 字段应正确传到 AlphaLogicsConfig
        call_args = mock_wf_class.call_args
        config = call_args.kwargs["config"]
        assert config.max_outer_rounds == default_config.alphalogics_max_outer_rounds
        assert config.min_ir_threshold == default_config.min_ir_threshold

    def test_passes_llm_client(self, default_config, sample_data):
        """llm_client 应传给 AlphaLogicsWorkflow"""
        mock_llm = MagicMock()
        with patch("QuantNodes.research.quant_alpha.logic_driven_pipeline.AlphaLogicsWorkflow") as mock_wf_class:
            mock_wf_class.return_value = MagicMock(run=MagicMock(return_value=MagicMock()))

            pipeline = LogicDrivenPipeline(config=default_config, llm_client=mock_llm)
            pipeline._run_alphalogics(sample_data)

        assert mock_wf_class.call_args.kwargs["llm_client"] is mock_llm


# ==============================================================================
# Test Class 3: run() 主方法
# ==============================================================================


class TestRunMain:
    """run() 主方法 (lines 176-255)"""

    def test_logic_driven_false_runs_standard(self, disabled_config, sample_data):
        """logic_driven=False 应走 standard pipeline"""
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.final_pool = []
        mock_pipeline_result.summary = {}

        with patch.object(LogicDrivenPipeline, "_run_standard_pipeline") as mock_std:
            mock_std.return_value = LogicDrivenPipelineResult()
            pipeline = LogicDrivenPipeline(config=disabled_config)
            result = pipeline.run(sample_data)

        mock_std.assert_called_once()
        assert result is not None

    def test_no_best_logic_returns_error(self, default_config, sample_data):
        """alphalogics 返回无 best_logic 应返回 error"""
        mock_alphalogics_result = MagicMock()
        mock_alphalogics_result.best_logic = None
        mock_alphalogics_result.best_evidence = None
        mock_alphalogics_result.inner_results = []

        with patch.object(LogicDrivenPipeline, "_run_alphalogics",
                          return_value=mock_alphalogics_result):
            pipeline = LogicDrivenPipeline(config=default_config)
            result = pipeline.run(sample_data)

        assert result.best_logic_name is None
        assert result.summary.get("error") == "no_best_logic"
        assert result.elapsed_seconds >= 0

    def test_best_logic_no_structured_uses_inner_factors(self, default_config, sample_data):
        """best_logic.structured=None 时, 直接用内层 factors"""
        mock_logic = MagicMock()
        mock_logic.name = "test_logic"
        mock_logic.structured = None

        mock_evidence = MagicMock()
        mock_evidence.best_ir = 0.1

        # inner_results with one final_pool entry
        mock_factor = MagicMock()
        mock_factor.formula_id = "F1"
        mock_factor.ic_mean = 0.05
        mock_factor.ir = 0.1

        mock_alphagpt_result = MagicMock()
        mock_alphagpt_result.final_pool = [mock_factor]

        mock_inner = MagicMock()
        mock_inner.alphagpt_result = mock_alphagpt_result

        mock_alphalogics_result = MagicMock()
        mock_alphalogics_result.best_logic = mock_logic
        mock_alphalogics_result.best_evidence = mock_evidence
        mock_alphalogics_result.inner_results = [mock_inner]

        with patch.object(LogicDrivenPipeline, "_run_alphalogics",
                          return_value=mock_alphalogics_result), \
             patch.object(LogicDrivenPipeline, "_persist_to_wiki"):
            pipeline = LogicDrivenPipeline(config=default_config)
            result = pipeline.run(sample_data)

        assert result.best_logic_name == "test_logic"
        # 应添加 1 个 logic_driven_factor
        assert len(result.logic_driven_factors) == 1
        assert result.logic_driven_factors[0].formula_id == "F1"

    def test_full_flow_with_structured(self, default_config, sample_data):
        """完整流程: best_logic.structured 存在, 跑 _enhance_with_mcts"""
        mock_structured = MagicMock()

        mock_logic = MagicMock()
        mock_logic.name = "test_logic"
        mock_logic.structured = mock_structured

        mock_evidence = MagicMock()
        mock_evidence.best_ir = 0.15

        mock_alphalogics_result = MagicMock()
        mock_alphalogics_result.best_logic = mock_logic
        mock_alphalogics_result.best_evidence = mock_evidence
        mock_alphalogics_result.inner_results = []

        with patch.object(LogicDrivenPipeline, "_run_alphalogics",
                          return_value=mock_alphalogics_result), \
             patch.object(LogicDrivenPipeline, "_enhance_with_mcts") as mock_enhance, \
             patch.object(LogicDrivenPipeline, "_persist_to_wiki") as mock_persist:
            pipeline = LogicDrivenPipeline(config=default_config)
            result = pipeline.run(sample_data)

        mock_enhance.assert_called_once()
        # persist 不应调用 (final_pool 为空)
        mock_persist.assert_not_called()
        assert result.best_gamma is not None
        assert result.summary["best_ir"] == 0.15

    def test_persist_to_wiki_when_final_pool_nonempty(self, default_config, sample_data):
        """final_pool 非空时应调 _persist_to_wiki"""
        mock_logic = MagicMock()
        mock_logic.name = "test_logic"
        mock_logic.structured = MagicMock()  # 走 MCTS 路径

        mock_alphalogics_result = MagicMock()
        mock_alphalogics_result.best_logic = mock_logic
        mock_alphalogics_result.best_evidence = MagicMock(best_ir=0.1)
        mock_alphalogics_result.inner_results = []

        with patch.object(LogicDrivenPipeline, "_run_alphalogics",
                          return_value=mock_alphalogics_result), \
             patch.object(LogicDrivenPipeline, "_enhance_with_mcts") as mock_enhance, \
             patch.object(LogicDrivenPipeline, "_persist_to_wiki") as mock_persist:
            # 模拟 _enhance_with_mcts 设置 final_pool
            def set_final_pool(*args, **kwargs):
                args[2].final_pool = [MagicMock()]
            mock_enhance.side_effect = set_final_pool

            pipeline = LogicDrivenPipeline(config=default_config)
            pipeline.run(sample_data)

        mock_persist.assert_called_once()


# ==============================================================================
# Test Class 4: _enhance_with_mcts
# ==============================================================================


class TestEnhanceWithMcts:
    """_enhance_with_mcts() (lines 280-326)"""

    def test_collects_inner_factors(self, default_config, sample_data):
        """从 inner_results 收集 FactorMetrics 到 result.logic_driven_factors"""
        mock_factor = MagicMock()
        mock_factor.formula_id = "F1"
        mock_factor.ic_mean = 0.05
        mock_factor.ir = 0.1

        mock_alphagpt_result = MagicMock()
        mock_alphagpt_result.final_pool = [mock_factor]

        mock_inner = MagicMock()
        mock_inner.alphagpt_result = mock_alphagpt_result

        mock_alphalogics = MagicMock()
        mock_alphalogics.inner_results = [mock_inner]

        result = LogicDrivenPipelineResult()
        result.alphalogics_result = mock_alphalogics

        with patch.object(LogicDrivenPipeline, "_run_alphalogics", return_value=mock_alphalogics), \
             patch("QuantNodes.research.quant_alpha.logic_driven_pipeline.AlphaPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.run = MagicMock(return_value=MagicMock(final_pool=[]))
            mock_pipeline_class.return_value = mock_pipeline

            pipeline = LogicDrivenPipeline(config=default_config)
            pipeline._enhance_with_mcts(sample_data, gamma=MagicMock(), result=result)

        assert len(result.logic_driven_factors) == 1
        assert result.logic_driven_factors[0].formula_id == "F1"
        assert result.mcts_enhanced is True

    def test_collects_multiple_inner_results(self, default_config, sample_data):
        """多个 inner_results 应累加所有 factors"""
        mock_factor_1 = MagicMock(formula_id="F1", ic_mean=0.05, ir=0.1)
        mock_factor_2 = MagicMock(formula_id="F2", ic_mean=0.06, ir=0.12)

        mock_inner_1 = MagicMock(alphagpt_result=MagicMock(final_pool=[mock_factor_1]))
        mock_inner_2 = MagicMock(alphagpt_result=MagicMock(final_pool=[mock_factor_2]))

        mock_alphalogics = MagicMock(inner_results=[mock_inner_1, mock_inner_2])
        result = LogicDrivenPipelineResult(alphalogics_result=mock_alphalogics)

        with patch("QuantNodes.research.quant_alpha.logic_driven_pipeline.AlphaPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.run = MagicMock(return_value=MagicMock(final_pool=[]))
            mock_pipeline_class.return_value = mock_pipeline

            pipeline = LogicDrivenPipeline(config=default_config)
            pipeline._enhance_with_mcts(sample_data, gamma=MagicMock(), result=result)

        assert len(result.logic_driven_factors) == 2
        ids = {f.formula_id for f in result.logic_driven_factors}
        assert ids == {"F1", "F2"}

    def test_skips_inners_without_alphagpt_result(self, default_config, sample_data):
        """inner.alphagpt_result=None 应跳过"""
        mock_inner_with = MagicMock(alphagpt_result=MagicMock(
            final_pool=[MagicMock(formula_id="F1", ic_mean=0.05, ir=0.1)]
        ))
        mock_inner_without = MagicMock(alphagpt_result=None)

        mock_alphalogics = MagicMock(inner_results=[mock_inner_with, mock_inner_without])
        result = LogicDrivenPipelineResult(alphalogics_result=mock_alphalogics)

        with patch("QuantNodes.research.quant_alpha.logic_driven_pipeline.AlphaPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.run = MagicMock(return_value=MagicMock(final_pool=[]))
            mock_pipeline_class.return_value = mock_pipeline

            pipeline = LogicDrivenPipeline(config=default_config)
            pipeline._enhance_with_mcts(sample_data, gamma=MagicMock(), result=result)

        # 只 1 个 factor (skip null)
        assert len(result.logic_driven_factors) == 1


# ==============================================================================
# Test Class 5: _run_standard_pipeline
# ==============================================================================


class TestRunStandardPipeline:
    """_run_standard_pipeline() fallback (lines 328-359)"""

    def test_runs_standard_pipeline(self, default_config, sample_data):
        """回退路径: 直接跑 AlphaPipeline"""
        mock_pool = [MagicMock(formula_id="F1"), MagicMock(formula_id="F2")]
        mock_pipeline_result = MagicMock(final_pool=mock_pool, summary={"ok": True})

        with patch("QuantNodes.research.quant_alpha.logic_driven_pipeline.AlphaPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.run = MagicMock(return_value=mock_pipeline_result)
            mock_pipeline_class.return_value = mock_pipeline

            pipeline = LogicDrivenPipeline(config=default_config)
            result = LogicDrivenPipelineResult()
            start_time = 100.0
            final = pipeline._run_standard_pipeline(sample_data, result, start_time)

        assert final is result
        assert result.final_pool == mock_pool
        assert result.summary == {"ok": True}
        assert result.elapsed_seconds >= 0


# ==============================================================================
# Test Class 6: _persist_to_wiki
# ==============================================================================


class TestPersistToWiki:
    """_persist_to_wiki() (lines 361-394)"""

    def test_persist_each_factor(self, default_config):
        """final_pool 每个 factor 应被存到 wiki"""
        mock_factors = []
        for i in range(3):
            f = MagicMock()
            f.formula_id = f"F{i}"
            f.formula = f"f_{i}"
            f.ir = 0.1
            f.ic_mean = 0.05
            f.ic_std = 0.01
            f.rank_ic_mean = 0.04
            mock_factors.append(f)

        result = LogicDrivenPipelineResult(final_pool=mock_factors, best_logic_name="test")

        with patch("QuantNodes.research.wiki.proxy.WikiFactorProxy") as mock_proxy_class:
            mock_proxy = MagicMock()
            mock_proxy.store_factor = MagicMock(side_effect=lambda f: f"page_{f.name}")
            mock_proxy_class.return_value = mock_proxy

            pipeline = LogicDrivenPipeline(config=default_config)
            pipeline._persist_to_wiki(result)

        assert len(result.wiki_pages) == 3

    def test_persist_exception_does_not_crash(self, default_config):
        """store_factor 抛异常时, 整体不应崩"""
        f = MagicMock(formula_id="F1", formula="f1", ir=0.1, ic_mean=0.05, ic_std=0.01, rank_ic_mean=0.04)
        result = LogicDrivenPipelineResult(final_pool=[f], best_logic_name="test")

        with patch("QuantNodes.research.wiki.proxy.WikiFactorProxy") as mock_proxy_class:
            mock_proxy = MagicMock()
            mock_proxy.store_factor = MagicMock(side_effect=RuntimeError("Wiki error"))
            mock_proxy_class.return_value = mock_proxy

            pipeline = LogicDrivenPipeline(config=default_config)
            # 不抛
            pipeline._persist_to_wiki(result)

        # 0 个 page (失败不计数)
        assert len(result.wiki_pages) == 0

    def test_wiki_proxy_creation_fails(self, default_config):
        """WikiFactorProxy 创建失败时, 整体 graceful"""
        result = LogicDrivenPipelineResult(final_pool=[MagicMock()], best_logic_name="test")

        with patch("QuantNodes.research.wiki.proxy.WikiFactorProxy",
                   side_effect=RuntimeError("Wiki init failed")):
            pipeline = LogicDrivenPipeline(config=default_config)
            # 不抛
            pipeline._persist_to_wiki(result)


# ==============================================================================
# Test Class 7: _build_summary
# ==============================================================================


class TestBuildSummary:
    """_build_summary() (lines 396-)"""

    def test_summary_includes_all_fields(self, default_config):
        """_build_summary 返回所有必要字段"""
        mock_evidence = MagicMock()
        mock_evidence.best_ir = 0.15
        mock_evidence.n_factors_explored = 5

        result = LogicDrivenPipelineResult(
            best_logic_name="test_logic",
            best_evidence=mock_evidence,
            logic_driven_factors=[MagicMock(), MagicMock()],
            mcts_enhanced=True,
        )

        pipeline = LogicDrivenPipeline(config=default_config)
        summary = pipeline._build_summary(result)

        assert summary["logic_driven"] is True
        assert summary["best_logic_name"] == "test_logic"
        assert summary["best_ir"] == 0.15
        assert summary["best_n_factors"] == 5
        assert summary["logic_driven_factors"] == 2
        assert summary["mcts_enhanced"] is True

    def test_summary_with_no_evidence(self, default_config):
        """无 best_evidence 时字段为 0"""
        result = LogicDrivenPipelineResult(
            best_logic_name=None,
            best_evidence=None,
            logic_driven_factors=[],
            mcts_enhanced=False,
        )

        pipeline = LogicDrivenPipeline(config=default_config)
        summary = pipeline._build_summary(result)

        assert summary["best_logic_name"] is None
        assert summary["best_ir"] == 0.0
        assert summary["best_n_factors"] == 0
        assert summary["logic_driven_factors"] == 0
        assert summary["mcts_enhanced"] is False


# ==============================================================================
# Phase D.5: logic_mining/pipelines.py _call_llm coverage (79→80%)
# ==============================================================================


class TestCallLlm:
    """Test _call_llm from logic_mining/pipelines.py (lines 62-73)."""

    def test_no_client_returns_default(self):
        """No client → return default."""
        from QuantNodes.research.quant_alpha.logic_mining.pipelines import _call_llm
        result = _call_llm(None, "agent", "prompt", "default")
        assert result == "default"

    def test_client_with_complete_method(self):
        """Client with complete() → call complete."""
        from QuantNodes.research.quant_alpha.logic_mining.pipelines import _call_llm
        mock_client = MagicMock()
        mock_client.complete.return_value = "llm response"
        result = _call_llm(mock_client, "agent", "test prompt", "default")
        assert result == "llm response"
        mock_client.complete.assert_called_once_with(agent_id="agent", prompt="test prompt")

    def test_client_callable(self):
        """Client is callable → call it directly."""
        from QuantNodes.research.quant_alpha.logic_mining.pipelines import _call_llm
        mock_client = MagicMock(spec=[])  # no 'complete' attribute
        mock_client.return_value = "callable response"
        result = _call_llm(mock_client, "agent", "test prompt", "default")
        assert result == "callable response"

    def test_client_exception_returns_default(self):
        """Client raises exception → fallback to default."""
        from QuantNodes.research.quant_alpha.logic_mining.pipelines import _call_llm
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("LLM failed")
        result = _call_llm(mock_client, "agent", "prompt", "default")
        assert result == "default"

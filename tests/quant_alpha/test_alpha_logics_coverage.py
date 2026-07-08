# coding=utf-8
"""
test_alpha_logics_coverage.py - 补 workflow/alpha_logics.py 覆盖 (Phase D.1)

目标: alpha_logics.py 44% → 80%+
- 覆盖 run() (lines 219-311)
- 覆盖 _build_initial_library() (lines 315-348)
- 覆盖 _run_inner_loop() (lines 356-397)
- 覆盖 _build_summary() (line 406)
- 覆盖 lazy import helpers (lines 76-77, 80-81)
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.workflow.alpha_gpt import AlphaGptResult
from QuantNodes.research.quant_alpha.workflow.state import FinalFormulaRecord
from QuantNodes.research.quant_alpha.workflow.alpha_logics import (
    AlphaLogicsConfig,
    AlphaLogicsResult,
    AlphaLogicsWorkflow,
    InnerLoopResult,
    _build_inner_evidence,
)


# ==============================================================================
# Helpers
# ==============================================================================


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """测试数据 (3 票 × 30 日)"""
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


# ==============================================================================
# Helpers
# ==============================================================================


def _make_final_record(rank: int, ir: float, formula_id: str = None) -> FinalFormulaRecord:
    """构造 FinalFormulaRecord (rank 是必填位置参数)"""
    return FinalFormulaRecord(
        rank=rank,
        formula_id=formula_id or f"F{rank}",
        formula=f"f_{rank}",
        ir=ir,
        ic_mean=ir * 0.1,
    )


def _make_alphagpt_result(irs: List[float] = None) -> AlphaGptResult:
    """构造 AlphaGptResult"""
    if irs is None:
        irs = [0.12, 0.08]
    pool = [_make_final_record(i + 1, ir, f"F{i + 1}") for i, ir in enumerate(irs)]
    return AlphaGptResult(
        objective="t",
        iterations_completed=1,
        total_formulas=len(irs),
        final_pool=pool,
        summary={"best_ir": max(irs) if irs else 0.0},
    )


def _make_logic_abstraction_result(structured, source_lib: str, source_formula: str):
    """构造 LogicAbstractionResult-like 对象"""
    res = MagicMock()
    res.structured_logic = structured
    res.source_lib = source_lib
    res.source_formula = source_formula
    return res


# ==============================================================================
# Test Class 1: 懒加载 helper
# ==============================================================================


class TestLazyImports:
    """测试 lazy import helpers (lines 75-81)"""

    def test_get_wiki_proxy(self):
        """_get_wiki_proxy() 返回 WikiFactorProxy 类"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _get_wiki_proxy
        from QuantNodes.research.wiki.proxy import WikiFactorProxy
        result = _get_wiki_proxy()
        assert result is WikiFactorProxy

    def test_get_wiki_logic(self):
        """_get_wiki_logic() 返回 WikiLogic 类"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _get_wiki_logic
        from QuantNodes.research.wiki.logic import WikiLogic
        result = _get_wiki_logic()
        assert result is WikiLogic

    def test_get_logic_source(self):
        """_get_logic_source() 返回 LogicSource 类"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _get_logic_source
        from QuantNodes.research.wiki.enums import LogicSource
        result = _get_logic_source()
        assert result is LogicSource


# ==============================================================================
# Test Class 2: AlphaLogicsWorkflow 初始化
# ==============================================================================


class TestWorkflowInit:
    """测试 __init__ 路径"""

    def test_init_creates_components(self, tmp_path: Path):
        """__init__ 应创建所有组件"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        # mock build_initial_logic_library 因为 init 不调用它, 但 mock Mining/Generator
        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator") as mock_gen_class, \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection") as mock_ref_class:
            mock_gen_class.return_value = MagicMock()
            mock_ref_class.return_value = MagicMock()

            workflow = AlphaLogicsWorkflow(config=config)

        assert workflow.config == config
        assert workflow.llm_client is None
        assert workflow.wiki is not None

    def test_init_with_llm_client(self, tmp_path: Path):
        """__init__ 接受 llm_client"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))
        mock_llm = MagicMock()

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator") as mock_gen_class, \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection") as mock_ref_class:
            mock_gen_class.return_value = MagicMock()
            mock_ref_class.return_value = MagicMock()

            workflow = AlphaLogicsWorkflow(config=config, llm_client=mock_llm)

        assert workflow.llm_client is mock_llm


# ==============================================================================
# Test Class 3: _build_initial_library
# ==============================================================================


class TestBuildInitialLibrary:
    """测试 _build_initial_library() (lines 313-348)"""

    def _stub_results(self, count: int):
        """构造 N 个 logic abstraction result"""
        results = []
        for i in range(count):
            structured = MagicMock()
            results.append(_make_logic_abstraction_result(
                structured=structured,
                source_lib="alpha101" if i < count // 2 else "alpha158",
                source_formula=f"formula_{i}",
            ))
        return results

    def test_builds_logics_from_results(self, tmp_path: Path):
        """_build_initial_library 应返回 WikiLogic 列表"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=self._stub_results(3)):
            workflow = AlphaLogicsWorkflow(config=config)
            logics = workflow._build_initial_library()

        assert len(logics) == 3
        for logic in logics:
            assert logic.name is not None

    def test_skips_results_with_no_structured_logic(self, tmp_path: Path):
        """_build_initial_library 应跳过 structured=None 的结果"""
        results = self._stub_results(2)
        results[0].structured_logic = None  # 第一个无 structured
        results[1].structured_logic = MagicMock()  # 第二个有

        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=results):
            workflow = AlphaLogicsWorkflow(config=config)
            logics = workflow._build_initial_library()

        assert len(logics) == 1

    def test_persist_best_logic_enabled(self, tmp_path: Path):
        """persist_best_logic=True 时应调用 wiki.store_logic"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"), persist_best_logic=True)

        mock_wiki = MagicMock()
        mock_wiki.store_logic = MagicMock()

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=self._stub_results(2)), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics._get_wiki_proxy",
                   return_value=lambda path: mock_wiki):
            workflow = AlphaLogicsWorkflow(config=config)
            workflow._build_initial_library()

        # 应调用 store_logic 2 次
        assert mock_wiki.store_logic.call_count == 2

    def test_persist_disabled(self, tmp_path: Path):
        """persist_best_logic=False 时不应调用 wiki.store_logic"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"), persist_best_logic=False)

        mock_wiki = MagicMock()
        mock_wiki.store_logic = MagicMock()

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=self._stub_results(2)), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics._get_wiki_proxy",
                   return_value=lambda path: mock_wiki):
            workflow = AlphaLogicsWorkflow(config=config)
            workflow._build_initial_library()

        assert mock_wiki.store_logic.call_count == 0

    def test_empty_results_returns_empty(self, tmp_path: Path):
        """空 results 应返回 []"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=[]):
            workflow = AlphaLogicsWorkflow(config=config)
            logics = workflow._build_initial_library()

        assert logics == []


# ==============================================================================
# Test Class 4: _run_inner_loop
# ==============================================================================


class TestRunInnerLoop:
    """测试 _run_inner_loop() (lines 350-402)"""

    def test_returns_evidence_with_mock_result(self, tmp_path: Path, sample_data):
        """_run_inner_loop 返回 InnerLoopResult 含 evidence"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"), data=sample_data)

        mock_logic = MagicMock()
        mock_logic.name = "test_logic"
        mock_logic.structured = MagicMock()
        mock_logic.content = "test content"

        mock_alphagpt_result = _make_alphagpt_result([0.12, 0.08])

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.AlphaGptWorkflow") as mock_workflow_class:
            mock_workflow = MagicMock()
            mock_workflow.run = MagicMock(return_value=mock_alphagpt_result)
            mock_workflow_class.return_value = mock_workflow

            workflow = AlphaLogicsWorkflow(config=config)
            result = workflow._run_inner_loop(mock_logic, 1)

        assert isinstance(result, InnerLoopResult)
        assert result.logic_name == "test_logic"
        assert result.evidence is not None
        assert result.alphagpt_result is mock_alphagpt_result
        assert result.evidence.best_ir == 0.12  # max of 0.12, 0.08

    def test_logic_without_structured_returns_empty_evidence(self, tmp_path: Path):
        """logic.structured=None 时返回空 evidence"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        mock_logic = MagicMock()
        mock_logic.name = "no_struct"
        mock_logic.structured = None

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"):
            workflow = AlphaLogicsWorkflow(config=config)
            result = workflow._run_inner_loop(mock_logic, 1)

        assert result.logic_name == "no_struct"
        assert result.evidence.best_ir == 0.0
        assert result.alphagpt_result is None

    def test_inner_loop_exception_handled(self, tmp_path: Path, sample_data):
        """AlphaGptWorkflow.run 抛异常时, 应 fallback"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"), data=sample_data)

        mock_logic = MagicMock()
        mock_logic.name = "test"
        mock_logic.structured = MagicMock()
        mock_logic.content = "test"

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.AlphaGptWorkflow") as mock_workflow_class:
            mock_workflow_class.side_effect = RuntimeError("LLM failed")
            workflow = AlphaLogicsWorkflow(config=config)
            result = workflow._run_inner_loop(mock_logic, 1)

        assert result.alphagpt_result is None
        assert result.evidence is not None


# ==============================================================================
# Test Class 5: run() 主方法
# ==============================================================================


class TestRunMainLoop:
    """测试 run() 主方法 (lines 213-311)"""

    def test_run_with_empty_library_returns_error(self, tmp_path: Path):
        """空初始库应返回 error 并终止"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"), max_outer_rounds=2)

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=[]):
            workflow = AlphaLogicsWorkflow(config=config)
            result = workflow.run()

        assert result.best_logic is None
        assert result.summary.get("error") == "initial_library_empty"

    def test_run_records_initial_library(self, tmp_path: Path):
        """run() 记录初始 library 到 result.library"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"), max_outer_rounds=1)

        # 1 个初始 logic
        initial_results = [_make_logic_abstraction_result(
            structured=MagicMock(), source_lib="alpha101", source_formula="f0")]

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"), \
             patch("QuantNodes.research.quant_alpha.workflow.alpha_logics.build_initial_logic_library",
                   return_value=initial_results):
            workflow = AlphaLogicsWorkflow(config=config)
            # mock _run_inner_loop 直接返回 evidence
            workflow._run_inner_loop = MagicMock(return_value=InnerLoopResult(
                logic_name="test",
                evidence=MagicMock(best_ir=0.5, n_factors_explored=3),
                alphagpt_result=MagicMock(),
            ))
            # mock generator
            workflow.generator.generate = MagicMock(return_value=_make_logic_abstraction_result(
                structured=MagicMock(), source_lib="alpha158", source_formula="f1"))
            workflow.refiner.refine = MagicMock(return_value={"diagnosis": "ok", "direction": "keep"})

            result = workflow.run()

        # 初始 library + 1 个新 logic
        assert len(result.library) == 2
        assert result.best_logic is not None


# ==============================================================================
# Test Class 6: _build_summary (单独测试)
# ==============================================================================


class TestBuildSummaryDirect:
    """测试 _build_summary() 直接调用"""

    def test_summary_with_no_evidence(self, tmp_path: Path):
        """无 best_evidence 时 summary 字段为 0"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"):
            workflow = AlphaLogicsWorkflow(config=config)
            result = AlphaLogicsResult()
            result.elapsed_seconds = 1.0
            summary = workflow._build_summary(result)

        assert "max_outer_rounds" in summary
        assert summary["best_logic"] is None
        assert summary["best_ir"] == 0.0
        assert summary["best_n_factors"] == 0
        assert summary["elapsed_seconds"] == 1.0

    def test_summary_with_evidence(self, tmp_path: Path):
        """有 best_evidence 时 summary 字段正确"""
        config = AlphaLogicsConfig(wiki_path=str(tmp_path / "wiki"))

        with patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicGenerator"), \
             patch("QuantNodes.research.quant_alpha.logic_mining.generator.MarketLogicRefinementDirection"):
            workflow = AlphaLogicsWorkflow(config=config)

            # 构造有 evidence 的 result
            mock_logic = MagicMock()
            mock_logic.name = "best_logic"
            mock_evidence = MagicMock()
            mock_evidence.best_ir = 0.15
            mock_evidence.n_factors_explored = 5

            result = AlphaLogicsResult(
                best_logic=mock_logic,
                best_evidence=mock_evidence,
                inner_results=[MagicMock(), MagicMock()],
            )
            result.library = [mock_logic, MagicMock(), MagicMock()]
            result.elapsed_seconds = 10.5
            summary = workflow._build_summary(result)

        assert summary["best_logic"] == "best_logic"
        assert summary["best_ir"] == 0.15
        assert summary["best_n_factors"] == 5
        assert summary["rounds_completed"] == 2
        assert summary["library_size"] == 3
        assert summary["elapsed_seconds"] == 10.5


# ==============================================================================
# Test Class 7: _build_inner_evidence 补完
# ==============================================================================


class TestBuildInnerEvidenceExtended:
    """_build_inner_evidence 补完 (lines 154-190)"""

    def test_with_alphagpt_no_final_pool(self):
        """alphagpt_result 存在但 final_pool 为空"""
        result = MagicMock()
        result.final_pool = []
        result.summary = {}
        ev = _build_inner_evidence("logic", result, 1)
        assert ev.n_factors_explored == 0
        assert ev.best_ir == 0.0
        assert ev.refinement_round == 1

    def test_with_only_negative_irs(self):
        """所有 IR 为负, best_ir 仍取 max(irs)"""
        result = _make_alphagpt_result([-0.05, -0.10])
        ev = _build_inner_evidence("logic", result, 2)
        assert ev.best_ir == -0.05
        assert ev.n_factors_explored == 2

    def test_best_factor_id_set(self):
        """best_factor_id 应从 best_idx 提取"""
        result = _make_alphagpt_result([0.05, 0.15, 0.08])
        ev = _build_inner_evidence("logic", result, 1)
        assert ev.best_factor_id == "F2"  # 0.15 → idx=1
        assert ev.best_ir == 0.15

    def test_with_none_result(self):
        """alphagpt_result=None"""
        ev = _build_inner_evidence("logic", None, 1)
        assert ev.n_factors_explored == 0
        assert ev.best_ir == 0.0
        assert ev.best_factor_id is None
        assert ev.refinement_round == 1

    def test_single_factor_evidence(self):
        """单个因子 evidence"""
        result = _make_alphagpt_result([0.10])
        ev = _build_inner_evidence("logic", result, 3)
        assert ev.n_factors_explored == 1
        assert ev.best_ir == 0.10
        assert ev.mean_ir == 0.10
        assert ev.refinement_round == 3
        assert ev.timestamp is not None

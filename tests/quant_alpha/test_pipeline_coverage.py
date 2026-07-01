# coding=utf-8
"""
test_pipeline_coverage.py - 补 pipeline.py 覆盖 (Phase D.3)

目标: pipeline.py 71% → 80%+
- _should_stop (lines 598-636)
- _check_timeout (lines 638-641)
- _precompute_forward_returns (lines 643+)
- _select_final_pool (lines 823-835)
- _get_formula_from_metrics (lines 837-856)
- _persist_to_wiki (lines 858-870)
"""
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
from QuantNodes.research.quant_alpha.pipeline import (
    AlphaPipeline,
    EarlyStopping,
    PipelineConfig,
    PipelineResult,
    RoundFeedback,
    TerminationConfig,
)


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
def pipeline(tmp_path: Path) -> AlphaPipeline:
    """最小化 AlphaPipeline 用于测试"""
    config = PipelineConfig(
        objective="test",
        termination=TerminationConfig(
            max_rounds=1,
            target_factors=5,
            early_stopping=True,
            patience=2,
            timeout_seconds=3600,
        ),
        output_dir=str(tmp_path / "out"),
    )
    return AlphaPipeline(config)


# ==============================================================================
# Test Class 1: _should_stop
# ==============================================================================


class TestShouldStop:
    """_should_stop() (lines 598-636)"""

    def test_no_feedback_returns_false(self, pipeline):
        """feedback=None → False"""
        es = EarlyStopping()
        assert pipeline._should_stop(None, [], es) is False

    def test_target_factors_reached(self, pipeline):
        """final_pool >= target_factors → True"""
        # target_factors=5, so len(pool)=5 → stop
        pool = [MagicMock() for _ in range(5)]
        es = EarlyStopping()
        feedback = RoundFeedback(round_num=1, best_ir=0.1, avg_ir=0.05, valid_count=3)
        assert pipeline._should_stop(feedback, pool, es) is True

    def test_early_stop_triggers(self, pipeline):
        """连续 N 轮无改善 → stop"""
        # patience=2, need 2 rounds of no improvement
        es = EarlyStopping(patience=2)
        # 第一轮
        pipeline._should_stop(
            RoundFeedback(round_num=1, best_ir=0.1, avg_ir=0.05, valid_count=3),
            [],
            es,
        )
        # 第二轮 (same ir, no improvement)
        pipeline._should_stop(
            RoundFeedback(round_num=2, best_ir=0.1, avg_ir=0.05, valid_count=3),
            [],
            es,
        )
        # 第三轮 (counter=2 >= patience=2 → stop)
        result = pipeline._should_stop(
            RoundFeedback(round_num=3, best_ir=0.1, avg_ir=0.05, valid_count=3),
            [],
            es,
        )
        assert result is True

    def test_timeout_triggers(self, pipeline):
        """超时 → stop"""
        # timeout_seconds=3600, set start time to now
        pipeline._start_time = time.time() - 3601  # 过期 1s
        es = EarlyStopping(patience=100)  # 很高 patience
        feedback = RoundFeedback(round_num=1, best_ir=0.5, avg_ir=0.3, valid_count=10)
        assert pipeline._should_stop(feedback, [], es) is True

    def test_no_stop_when_conditions_not_met(self, pipeline):
        """条件不满足 → False"""
        es = EarlyStopping(patience=100)
        pipeline._start_time = time.time()
        feedback = RoundFeedback(round_num=1, best_ir=0.1, avg_ir=0.05, valid_count=3)
        pool = [MagicMock() for _ in range(2)]  # < target=5
        assert pipeline._should_stop(feedback, pool, es) is False


# ==============================================================================
# Test Class 2: _check_timeout
# ==============================================================================


class TestCheckTimeout:
    """_check_timeout() (lines 638-641)"""

    def test_not_timed_out(self, pipeline):
        """未超时 → False"""
        pipeline._start_time = time.time()
        pipeline.config.termination.timeout_seconds = 3600
        assert pipeline._check_timeout() is False

    def test_timed_out(self, pipeline):
        """超时 → True"""
        pipeline._start_time = time.time() - 3601
        pipeline.config.termination.timeout_seconds = 3600
        assert pipeline._check_timeout() is True


# ==============================================================================
# Test Class 3: _precompute_forward_returns
# ==============================================================================


class TestPrecomputeForwardReturns:
    """_precompute_forward_returns() (lines 643+)"""

    def test_adds_fwd_columns(self, pipeline, sample_data):
        """应添加 _fwd_ret_{n}d 列"""
        result = pipeline._precompute_forward_returns(sample_data)
        # 默认 forward_returns=(1, 5, 20)
        assert "_fwd_ret_1d" in result.columns
        assert "_fwd_ret_5d" in result.columns
        assert "_fwd_ret_20d" in result.columns
        assert result.shape[0] == sample_data.shape[0]

    def test_fwd_values_computed(self, pipeline, sample_data):
        """fwd_return 列有值"""
        result = pipeline._precompute_forward_returns(sample_data)
        fwd_1 = result["_fwd_ret_1d"].drop_nulls()
        # 有非空值
        assert len(fwd_1) > 0

    def test_does_not_duplicate_columns(self, pipeline, sample_data):
        """重复调用不重复列"""
        r1 = pipeline._precompute_forward_returns(sample_data)
        r2 = pipeline._precompute_forward_returns(r1)
        # 列数不应翻倍
        fwd_cols = [c for c in r2.columns if c.startswith("_fwd_ret_")]
        assert len(fwd_cols) == 3  # 只 3 个 (1d, 5d, 20d)


# ==============================================================================
# Test Class 4: _select_final_pool
# ==============================================================================


class TestSelectFinalPool:
    """_select_final_pool() (lines 823-835)"""

    def test_empty_rounds_returns_empty(self, pipeline):
        """空轮次 → 空池"""
        assert pipeline._select_final_pool([]) == []

    def test_selects_top_k_by_abs_ir(self, pipeline):
        """按 |IR| 排序取 top_k"""
        # top_k=10 (默认), 构造 15 个 metrics
        from QuantNodes.research.quant_alpha.pipeline import RoundResult
        round_result = RoundResult(
            round_num=1,
            final_pool=[
                FactorMetrics(formula_id=f"F{i}", status="success", ir=float(i))
                for i in range(15)
            ],
        )
        result = pipeline._select_final_pool([round_result])
        # top_k=10
        assert len(result) == 10
        # 最高 |IR| 在前面
        assert result[0].formula_id == "F14"  # |14| > |13| > ...

    def test_merges_multi_rounds(self, pipeline):
        """多个轮次合并"""
        from QuantNodes.research.quant_alpha.pipeline import RoundResult
        r1 = RoundResult(
            round_num=1,
            final_pool=[
                FactorMetrics(formula_id="F1", status="success", ir=0.8),
                FactorMetrics(formula_id="F2", status="success", ir=0.3),
            ],
        )
        r2 = RoundResult(
            round_num=2,
            final_pool=[
                FactorMetrics(formula_id="F3", status="success", ir=0.5),
            ],
        )
        result = pipeline._select_final_pool([r1, r2])
        # top_k=10, 所有 3 个合并后取前 10
        assert len(result) == 3
        ids = [m.formula_id for m in result]
        assert ids[0] == "F1"  # ir=0.8
        assert ids[1] == "F3"  # ir=0.5
        assert ids[2] == "F2"  # ir=0.3


# ==============================================================================
# Test Class 5: _get_formula_from_metrics
# ==============================================================================


class TestGetFormulaFromMetrics:
    """_get_formula_from_metrics() (lines 837-856)"""

    def test_finds_in_alphagpt_result(self, pipeline):
        """从 alphagpt_result.final_pool 找到 formula"""
        mock_agpt = MagicMock()
        mock_agpt.final_pool = [
            MagicMock(formula_id="F1", formula="rank(close)"),
        ]
        metrics = FactorMetrics(formula_id="F1", status="success", ir=0.1)
        result = pipeline._get_formula_from_metrics(metrics, mock_agpt, None)
        assert result == "rank(close)"

    def test_finds_in_mcts_result(self, pipeline):
        """从 mcts_result.best_k_nodes 找到 formula"""
        mock_mcts = MagicMock()
        mock_mcts.best_k_nodes = [
            MagicMock(entry_id="F1", formula="ts_mean(close, 5)"),
        ]
        metrics = FactorMetrics(formula_id="F1", status="success", ir=0.1)
        result = pipeline._get_formula_from_metrics(metrics, None, mock_mcts)
        assert result == "ts_mean(close, 5)"

    def test_alphagpt_priority_over_mcts(self, pipeline):
        """alphagpt_result 优先于 mcts_result"""
        mock_agpt = MagicMock()
        mock_agpt.final_pool = [MagicMock(formula_id="F1", formula="from_agpt")]
        mock_mcts = MagicMock()
        mock_mcts.best_k_nodes = [MagicMock(entry_id="F1", formula="from_mcts")]
        metrics = FactorMetrics(formula_id="F1", status="success", ir=0.1)
        result = pipeline._get_formula_from_metrics(metrics, mock_agpt, mock_mcts)
        assert result == "from_agpt"

    def test_not_found_returns_none(self, pipeline):
        """找不到 → None"""
        mock_agpt = MagicMock()
        mock_agpt.final_pool = [MagicMock(formula_id="F1", formula="f1")]
        metrics = FactorMetrics(formula_id="F99", status="success", ir=0.1)
        result = pipeline._get_formula_from_metrics(metrics, mock_agpt, None)
        assert result is None

    def test_no_results_returns_none(self, pipeline):
        """两者都无 → None"""
        metrics = FactorMetrics(formula_id="F1", status="success", ir=0.1)
        result = pipeline._get_formula_from_metrics(metrics, None, None)
        assert result is None


# ==============================================================================
# Test Class 6: _persist_to_wiki
# ==============================================================================


class TestPersistToWiki:
    """_persist_to_wiki() (lines 858-870)"""

    def test_persist_each_factor(self, pipeline):
        """每个 factor 应被存"""
        factors = [
            FactorMetrics(formula_id="F1", status="success", ir=0.5, ic_mean=0.01),
            FactorMetrics(formula_id="F2", status="success", ir=0.3, ic_mean=0.02),
        ]
        pipeline.wiki = MagicMock()
        pipeline.wiki.store_factor = MagicMock(side_effect=lambda f: f"page_{f.name}")

        pages = pipeline._persist_to_wiki(factors)
        assert pages == ["page_F1", "page_F2"]

    def test_wiki_error_does_not_crash(self, pipeline):
        """store_factor 失败时 graceful"""
        factors = [FactorMetrics(formula_id="F1", status="success", ir=0.5)]
        pipeline.wiki = MagicMock()
        pipeline.wiki.store_factor = MagicMock(side_effect=RuntimeError("Wiki error"))

        pages = pipeline._persist_to_wiki(factors)
        # 失败不加入 pages
        assert pages == []

    def test_empty_factors(self, pipeline):
        """空 factors → 空 pages"""
        pipeline.wiki = MagicMock()
        assert pipeline._persist_to_wiki([]) == []

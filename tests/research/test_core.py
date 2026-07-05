"""Tests for QuantNodes.research.core — recipe/stage/pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest


# ── PaperRecipe ────────────────────────────────────────────────


class _StubSignalSource:
    """Minimal SignalSource stub for PaperRecipe tests."""

    def __init__(self, signals: list = None):
        self._signals = signals or []

    def iter_signals(self):
        from QuantNodes.research.signal_source.base import Signal
        for s in self._signals:
            if isinstance(s, Signal):
                yield s
            else:
                yield Signal(id=s["id"], name=s.get("name", s["id"]), formula_brief=s.get("f", ""))


class _StubDataSource:
    def get(self, symbol, start, end):
        return None


class _StubBacktest:
    def run(self, code, h5_path, signal):
        return {"ic_mean": 0.05}


class TestPaperRecipeValidation:
    """Tests for PaperRecipe validation in __post_init__."""

    def _recipe(self, **overrides):
        defaults = dict(
            paper_id="test_paper",
            signal_source=_StubSignalSource(),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        defaults.update(overrides)
        from QuantNodes.research.core.recipe import PaperRecipe
        return PaperRecipe(**defaults)

    def test_minimal_creation(self):
        r = self._recipe()
        assert r.paper_id == "test_paper"
        assert r.delay == 3.0
        assert r.workers == 1
        assert r.timeout == 180
        assert r.skip_existing is False
        assert r.max_failures == 999
        assert r.sinks == []
        assert r.metadata == {}

    def test_empty_paper_id_raises(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        with pytest.raises(ValueError, match="paper_id must be non-empty"):
            PaperRecipe(
                paper_id="",
                signal_source=_StubSignalSource(),
                data_source=_StubDataSource(),
                backtest_engine=_StubBacktest(),
            )

    def test_workers_below_one_raises(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        with pytest.raises(ValueError, match="workers must be >= 1"):
            PaperRecipe(
                paper_id="x",
                signal_source=_StubSignalSource(),
                data_source=_StubDataSource(),
                backtest_engine=_StubBacktest(),
                workers=0,
            )

    def test_workers_above_3_warns_and_caps(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        with pytest.warns(UserWarning, match="exceeds LLM rate-limit cap"):
            r = PaperRecipe(
                paper_id="x",
                signal_source=_StubSignalSource(),
                data_source=_StubDataSource(),
                backtest_engine=_StubBacktest(),
                workers=10,
            )
        assert r.workers == 3

    def test_workers_3_no_warning(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning = error
            r = PaperRecipe(
                paper_id="x",
                signal_source=_StubSignalSource(),
                data_source=_StubDataSource(),
                backtest_engine=_StubBacktest(),
                workers=3,
            )
        assert r.workers == 3

    def test_custom_delay(self):
        r = self._recipe(delay=0.5)
        assert r.delay == 0.5

    def test_custom_metadata(self):
        r = self._recipe(metadata={"k": "v"})
        assert r.metadata == {"k": "v"}

    def test_sinks_default_empty(self):
        r = self._recipe()
        assert r.sinks == []

    def test_paper_path_optional(self):
        r = self._recipe(paper_path=Path("/tmp/x.pdf"))
        assert r.paper_path == Path("/tmp/x.pdf")

    def test_paper_path_none_default(self):
        r = self._recipe()
        assert r.paper_path is None


# ── Stage / StageContext (legacy) ──────────────────────────────


class TestStageContext:
    """Tests for StageContext dataclass (legacy)."""

    def test_default_construction(self):
        from QuantNodes.research.core.stage import StageContext

        ctx = StageContext()
        assert ctx.workspace_path is None
        assert ctx.alpha_indices == []
        assert ctx.metadata == {}

    def test_with_workspace_path(self):
        from QuantNodes.research.core.stage import StageContext

        ctx = StageContext(workspace_path=Path("/tmp/workspace"))
        assert ctx.workspace_path == Path("/tmp/workspace")

    def test_with_alpha_indices(self):
        from QuantNodes.research.core.stage import StageContext

        ctx = StageContext(alpha_indices=["alpha-001", "alpha-002"])
        assert ctx.alpha_indices == ["alpha-001", "alpha-002"]

    def test_with_metadata(self):
        from QuantNodes.research.core.stage import StageContext

        ctx = StageContext(metadata={"k": "v"})
        assert ctx.metadata == {"k": "v"}


class TestStageAbstract:
    """Tests for Stage ABC."""

    def test_cannot_instantiate_abstract(self):
        from QuantNodes.research.core.stage import Stage

        with pytest.raises(TypeError, match="abstract"):
            Stage()  # type: ignore

    def test_subclass_must_implement_execute(self):
        from QuantNodes.research.core.stage import Stage, StageContext

        class _Bad(Stage):
            pass

        with pytest.raises(TypeError, match="abstract"):
            _Bad()

    def test_subclass_with_execute_works(self):
        from QuantNodes.research.core.stage import Stage, StageContext

        class _Good(Stage):
            name = "test_stage"

            def execute(self, ctx):
                ctx.metadata["ran"] = True
                return ctx

        s = _Good()
        assert s.name == "test_stage"
        ctx = StageContext()
        result = s.execute(ctx)
        assert result.metadata["ran"] is True

    def test_default_name(self):
        from QuantNodes.research.core.stage import Stage, StageContext

        class _Anonymous(Stage):
            def execute(self, ctx):
                return ctx

        assert _Anonymous.name == "unnamed"


# ── PaperPipeline ──────────────────────────────────────────────


class TestPaperPipelineSerial:
    """Tests for PaperPipeline serial mode."""

    def test_empty_signal_source(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        recipe = PaperRecipe(
            paper_id="empty",
            signal_source=_StubSignalSource([]),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run()
        assert results == []

    def test_serial_processes_all_signals(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [
            {"id": "alpha-001"},
            {"id": "alpha-002"},
            {"id": "alpha-003"},
        ]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run()
        assert len(results) == 3

    def test_results_contain_signal_id(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": "alpha-007"}, {"id": "alpha-008"}]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run()
        assert {r["signal_id"] for r in results} == {"alpha-007", "alpha-008"}

    def test_process_one_returns_placeholder(self):
        """PR1: returns placeholder dict with status='success'."""
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": "alpha-001"}]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run()
        assert results[0]["status"] == "success"
        assert results[0]["stage"] == "pr1_placeholder"


class TestPaperPipelineIndexFilter:
    """Tests for PaperPipeline with indices range filter."""

    def test_filter_by_indices(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": f"alpha-{i:03d}"} for i in range(1, 6)]  # 001-005
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run(indices=range(1, 4))  # 001, 002, 003
        ids = {r["signal_id"] for r in results}
        assert ids == {"alpha-001", "alpha-002", "alpha-003"}

    def test_filter_no_matches(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": f"alpha-{i:03d}"} for i in range(1, 4)]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run(indices=range(100, 200))
        assert results == []

    def test_filter_partial(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": f"alpha-{i:03d}"} for i in range(1, 11)]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run(indices=range(3, 5))  # 003, 004
        ids = sorted(r["signal_id"] for r in results)
        assert ids == ["alpha-003", "alpha-004"]


class TestPaperPipelineParallel:
    """Tests for PaperPipeline parallel mode (workers > 1)."""

    def test_parallel_processes_all_signals(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": f"alpha-{i:03d}"} for i in range(1, 6)]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
            workers=2,
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run()
        assert len(results) == 5

    def test_parallel_with_workers_3(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        signals = [{"id": f"alpha-{i:03d}"} for i in range(1, 4)]
        recipe = PaperRecipe(
            paper_id="p",
            signal_source=_StubSignalSource(signals),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
            workers=3,
        )
        pipeline = PaperPipeline(recipe)
        results = pipeline.run()
        assert len(results) == 3


class TestPaperPipelineState:
    """Tests for PaperPipeline state tracking."""

    def test_initial_state(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        recipe = PaperRecipe(
            paper_id="x",
            signal_source=_StubSignalSource([]),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        assert pipeline.results == []
        assert pipeline.failures == 0
        assert pipeline.t0 == 0.0
        assert pipeline.batch_t0 == 0.0

    def test_pipeline_has_recipe_reference(self):
        from QuantNodes.research.core.recipe import PaperRecipe
        from QuantNodes.research.core.pipeline import PaperPipeline

        recipe = PaperRecipe(
            paper_id="x",
            signal_source=_StubSignalSource([]),
            data_source=_StubDataSource(),
            backtest_engine=_StubBacktest(),
        )
        pipeline = PaperPipeline(recipe)
        assert pipeline.recipe is recipe
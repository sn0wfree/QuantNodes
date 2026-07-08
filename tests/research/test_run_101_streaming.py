"""Tests for run_101_alphas_v2 M4.6 NDJSON streaming + async driver (PR6.10).

Verifies:
  - `--stream-mode` flag activates BatchSummarySink.stream_write_async
  - `_write_summary_async` switches between write_batch_async (off) and
    stream_write_async (on) based on config.stream_mode
  - NDJSON output is one JSON object per line
  - Sync alias `_write_summary` still works for backward compat
  - RunConfig.stream_mode field exists with default False
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts/research to path so we can import the run_101_alphas_v2 module
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "research"
sys.path.insert(0, str(SCRIPT_DIR))

import run_101_alphas_v2 as r101  # noqa: E402
from QuantNodes.research.backtest.base import FactorResult  # noqa: E402
from QuantNodes.research.signal_source.base import Signal  # noqa: E402


# ── helpers ──────────────────────────────────────────────


def _make_signal(idx: int = 1) -> Signal:
    return Signal(
        id=f"alpha_{idx:03d}",
        name=f"Alpha#{idx}",
        formula_brief="rank(...) fake",
        metadata={"alpha_index": idx},
    )


def _make_result(idx: int = 1) -> FactorResult:
    return FactorResult(
        signal=_make_signal(idx),
        status="success",
        code="def compute_factor(df): return df['close']",
        code_chars=37,
        factor_series=None,
        long_df=None,
        h5_path=None,
        backtest={"ic_mean": 0.05, "icir": 0.3, "win_rate": 0.55},
        stage=None,
        error=None,
        elapsed_sec=10.0,
        metadata={},
    )


def _make_minimal_stage(config: Any) -> r101.FactorStage:
    """Build a FactorStage without running __init__'s heavy setup.

    Sets just the attributes needed for _write_summary_async:
      - config
      - results
      - batch_t0
      - _summary_sink
    """
    stage = r101.FactorStage.__new__(r101.FactorStage)
    stage.config = config
    stage.results = []
    stage.batch_t0 = 0.0
    stage._summary_sink = None
    return stage


# ── Test Class 1: RunConfig.stream_mode 字段 ─────────────


class TestRunConfigStreamMode:
    """M4.6 新 RunConfig.stream_mode 字段."""

    def test_stream_mode_field_exists(self) -> None:
        """RunConfig 必须有 stream_mode 字段 (default False)."""
        assert "stream_mode" in r101.RunConfig.__dataclass_fields__

    def test_stream_mode_default_false(self) -> None:
        """默认 False (向后兼容, 不启用 NDJSON streaming)."""
        config = r101.RunConfig()
        assert config.stream_mode is False

    def test_stream_mode_set_true(self) -> None:
        """CLI --stream-mode 传 True 走 NDJSON 路径."""
        config = r101.RunConfig(stream_mode=True)
        assert config.stream_mode is True


# ── Test Class 2: _write_summary_async 模式切换 ─────────


class TestWriteSummaryAsyncModeSwitch:
    """`--stream-mode` flag 切换 write_batch_async vs stream_write_async."""

    @pytest.mark.asyncio
    async def test_stream_mode_off_uses_write_batch_async(self, tmp_path: Path) -> None:
        """stream_mode=False → 调 BatchSummarySink.write_batch_async."""
        config = r101.RunConfig(stream_mode=False, output_dir=tmp_path)
        stage = _make_minimal_stage(config)

        # Mock _summary_sink
        stage._summary_sink = MagicMock()
        stage._summary_sink.write_batch_async = AsyncMock(return_value=[])
        stage._summary_sink.stream_write_async = AsyncMock(return_value=[])

        stage.results = [_make_result(idx=1)]

        await stage._write_summary_async()

        # write_batch_async 被调
        stage._summary_sink.write_batch_async.assert_awaited_once_with(stage.results)
        # stream_write_async 不被调
        stage._summary_sink.stream_write_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_mode_on_uses_stream_write_async(self, tmp_path: Path) -> None:
        """stream_mode=True → 调 BatchSummarySink.stream_write_async."""
        config = r101.RunConfig(stream_mode=True, output_dir=tmp_path)
        stage = _make_minimal_stage(config)

        # Mock _summary_sink
        stage._summary_sink = MagicMock()
        stage._summary_sink.write_batch_async = AsyncMock(return_value=[])
        stage._summary_sink.stream_write_async = AsyncMock(return_value=[])

        stage.results = [_make_result(idx=1), _make_result(idx=2)]

        await stage._write_summary_async()

        # stream_write_async 被调 (with AsyncIterator + filename)
        stage._summary_sink.stream_write_async.assert_awaited_once()
        # write_batch_async 不被调
        stage._summary_sink.write_batch_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_write_async_passes_ndjson_filename(self, tmp_path: Path) -> None:
        """stream_write_async 调时必须传 ndjson_filename 参数."""
        config = r101.RunConfig(stream_mode=True, output_dir=tmp_path)
        stage = _make_minimal_stage(config)

        stage._summary_sink = MagicMock()
        stage._summary_sink.stream_write_async = AsyncMock(return_value=[])

        stage.results = [_make_result(idx=1)]

        await stage._write_summary_async()

        # 验证 keyword 参数
        call_kwargs = stage._summary_sink.stream_write_async.call_args.kwargs
        assert "ndjson_filename" in call_kwargs
        assert call_kwargs["ndjson_filename"] == "multi_alpha_001_to_101.ndjson"


# ── Test Class 3: sync _write_summary backward compat ─────────


class TestWriteSummarySyncAlias:
    """sync `_write_summary` 仍 work (backward compat for tests)."""

    def test_write_summary_uses_write_batch(self, tmp_path: Path) -> None:
        """sync _write_summary 调 sync write_batch (与 pre-M4.6 一致)."""
        config = r101.RunConfig(output_dir=tmp_path)
        stage = _make_minimal_stage(config)

        stage._summary_sink = MagicMock()
        stage._summary_sink.write_batch = MagicMock(return_value=[])

        stage.results = [_make_result(idx=1)]

        stage._write_summary()

        stage._summary_sink.write_batch.assert_called_once_with(stage.results)


# ── Test Class 4: _iter_results_async 生成器 ─────────


class TestIterResultsAsync:
    """`_iter_results_async()` async generator — 喂 stream_write_async."""

    @pytest.mark.asyncio
    async def test_iter_results_yields_all_in_order(self) -> None:
        """按顺序 yield self.results 全部元素."""
        config = r101.RunConfig()
        stage = _make_minimal_stage(config)

        results = [_make_result(idx=i) for i in range(1, 6)]
        stage.results = results

        out = []
        async for r in stage._iter_results_async():
            out.append(r)

        assert out == results

    @pytest.mark.asyncio
    async def test_iter_results_empty(self) -> None:
        """空 self.results → 0 个 yield."""
        config = r101.RunConfig()
        stage = _make_minimal_stage(config)
        stage.results = []

        out = []
        async for r in stage._iter_results_async():
            out.append(r)

        assert out == []


# ── Test Class 5: 真实 sink 集成测试 (with NDJSON file) ─────────


class TestNDJSONFileIntegration:
    """端到端: 真实 BatchSummarySink + stream_write_async → NDJSON 文件."""

    @pytest.mark.asyncio
    async def test_stream_write_appends_ndjson_file(self, tmp_path: Path) -> None:
        """3 个 result → NDJSON 文件 3 行, 每行 valid JSON 含 signal.id."""
        from QuantNodes.research.sink import BatchSummarySink

        sink = BatchSummarySink(
            output_dir=tmp_path,
            paper_id="stream_test",
            json_filename="multi_test.json",  # not used
            md_filename="multi_test.md",      # not used
        )

        results = [_make_result(idx=i) for i in range(1, 4)]

        async def gen() -> Any:
            for r in results:
                yield r

        await sink.stream_write_async(gen(), ndjson_filename="out.ndjson")

        ndjson_path = tmp_path / "out.ndjson"
        assert ndjson_path.exists()

        lines = ndjson_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        # 每行 valid JSON, 含 alpha_index (从 to_dict() 字段 — flattened)
        for i, line in enumerate(lines, 1):
            data = json.loads(line)
            # to_dict() flattens: data["signal_id"] = "alpha_NNN", data["alpha_index"] = N
            assert "signal_id" in data, f"missing 'signal_id' in {data}"
            assert data["signal_id"] == f"alpha_{i:03d}"
            assert data["alpha_index"] == i

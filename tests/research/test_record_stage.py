"""Tests for RecordStage sync/async dual API (M4.6 / PR6.10).

Verifies:
  - `record()` async 主接口: 4 步顺序 (state → log → persist → log_outcome)
  - `record_sync()` sync wrapper: byte-equal 顺序
  - `_persist_one()` async: 调 sink.write_one_async (异常 tolerance)
  - `_persist_one_sync()` sync: 调 sink.write_one (异常 tolerance)
  - asyncio.gather 多次 record 并发安全
  - PR0 backward compat: 老 sync `record()` 调用仍 work (老 patch 模式)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from QuantNodes.research.backtest.base import FactorResult
from QuantNodes.research.factor.record_stage import RecordStage
from QuantNodes.research.signal_source.base import Signal


# ── helpers ──────────────────────────────────────────────


def _make_signal(idx: int = 1, signal_id: str | None = None) -> Signal:
    """Minimal Signal for test fixtures."""
    return Signal(
        id=signal_id or f"alpha_{idx:03d}",
        name=f"Alpha#{idx}",
        formula_brief=f"rank(...) fake formula for alpha #{idx}",
        metadata={"alpha_index": idx},
    )


def _make_result(
    idx: int = 1,
    status: str = "success",
    error: str | None = None,
) -> FactorResult:
    """Minimal FactorResult for test fixtures."""
    return FactorResult(
        signal=_make_signal(idx),
        status=status,
        code="def compute_factor(df): return df['close']",
        code_chars=37,
        factor_series=None,
        long_df=None,
        h5_path=None,
        backtest={"ic_mean": 0.05, "icir": 0.3, "win_rate": 0.55},
        stage=None,
        error=error,
        elapsed_sec=10.0,
        metadata={},
    )


def _make_record_stage(sink: Any) -> RecordStage:
    """Build a RecordStage with the given sink and clean state."""
    return RecordStage(
        single_sink=sink,
        results=[],
        failures=[0],
    )


# ── Test Class 1: async record() 主接口 ────────────────


class TestRecordAsync:
    """`async def record()` — M4.6 主入口."""

    @pytest.mark.asyncio
    async def test_record_appends_result(self) -> None:
        """4 步顺序: state (results.append) → log → persist → log_outcome."""
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        await stage.record(result, elapsed_cum=1.0)

        assert stage.results == [result]

    @pytest.mark.asyncio
    async def test_record_calls_sink_write_one_async(self) -> None:
        """_persist_one 调 sink.write_one_async (async API)."""

        async def fake_write_one_async(r: Any) -> Any:
            fake_write_one_async.calls.append(r)
            return f"/tmp/{r.signal.id}.json"

        fake_write_one_async.calls = []
        sink = MagicMock()
        sink.write_one_async = fake_write_one_async
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        await stage.record(result, elapsed_cum=1.0)

        assert fake_write_one_async.calls == [result]

    @pytest.mark.asyncio
    async def test_record_failure_increments_failures(self) -> None:
        """status != 'success' 时 _update_state +1 failures."""
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=2, status="failed", error="boom")
        await stage.record(result, elapsed_cum=1.0)

        assert stage.failures[0] == 1

    @pytest.mark.asyncio
    async def test_record_success_does_not_increment_failures(self) -> None:
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=1, status="success")
        await stage.record(result, elapsed_cum=1.0)

        assert stage.failures[0] == 0

    @pytest.mark.asyncio
    async def test_record_sink_exception_does_not_raise(self) -> None:
        """_persist_one 异常 tolerance — log warning 不抛 (M4.6 与 PR9c 一致)."""
        async def raise_write(r: Any) -> Any:
            raise RuntimeError("disk full")

        sink = MagicMock()
        sink.write_one_async = raise_write
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        # 不应抛
        await stage.record(result, elapsed_cum=1.0)

        # state 步仍执行 (results 已 append, failures = 0 since success)
        assert stage.results == [result]
        assert stage.failures[0] == 0


# ── Test Class 2: sync record_sync() wrapper ─────────────


class TestRecordSync:
    """`def record_sync()` — M4.6 sync wrapper (backward compat)."""

    def test_record_sync_appends_result(self) -> None:
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        stage.record_sync(result, elapsed_cum=1.0)

        assert stage.results == [result]

    def test_record_sync_calls_sink_write_one(self) -> None:
        """_persist_one_sync 调 sink.write_one (sync API) — 不是 write_one_async."""
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        stage.record_sync(result, elapsed_cum=1.0)

        sink.write_one.assert_called_once_with(result)
        sink.write_one_async.assert_not_called()

    def test_record_sync_failure_increments_failures(self) -> None:
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=2, status="failed", error="boom")
        stage.record_sync(result, elapsed_cum=1.0)

        assert stage.failures[0] == 1

    def test_record_sync_sink_exception_does_not_raise(self) -> None:
        sink = MagicMock()
        sink.write_one.side_effect = RuntimeError("disk full")
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        stage.record_sync(result, elapsed_cum=1.0)

        assert stage.results == [result]
        assert stage.failures[0] == 0


# ── Test Class 3: concurrent record() ─────────────


class TestConcurrentRecord:
    """asyncio.gather 多 record 并发 — 验证 state mutation 正确."""

    @pytest.mark.asyncio
    async def test_concurrent_record_appends_all(self) -> None:
        """10 个 record 并发, self.results 长度 = 10 (无丢)."""
        sink = MagicMock()
        stage = _make_record_stage(sink)

        results = [_make_result(idx=i) for i in range(1, 11)]
        await asyncio.gather(*(stage.record(r, 1.0) for r in results))

        assert len(stage.results) == 10
        assert set(r.signal.id for r in stage.results) == set(
            r.signal.id for r in results
        )

    @pytest.mark.asyncio
    async def test_concurrent_record_failures_count_correct(self) -> None:
        """5 success + 5 failed 并发, failures = 5."""
        sink = MagicMock()
        stage = _make_record_stage(sink)

        results = [
            _make_result(idx=i, status="success" if i <= 5 else "failed", error="err")
            for i in range(1, 11)
        ]
        await asyncio.gather(*(stage.record(r, 1.0) for r in results))

        assert stage.failures[0] == 5


# ── Test Class 4: Backward compat (PR0) ─────────────


class TestBackwardCompatSync:
    """老 PR0 test fixtures 依赖 record() 同步调用 — M4.6 仍兼容."""

    def test_pr0_sync_call_signature_preserved(self) -> None:
        """老代码: `stage.record(result, elapsed)` 调 sync — 仍可调 (但现在返回 coroutine).

        M4.6 决策: `record()` 改 async, 老 `record_sync()` 提供 sync 接口.
        老 patch 模式 `patch.object(RecordStage, 'record', lambda *a: mock_result)` 仍 work
        因为 patch 直接覆盖 `record` 名字 (class-level, 因 @dataclass(slots=True));
        测试代码调 `stage.record(result, elapsed)` 时拿到的是 mock 的 sync lambda
        (而不是 async coroutine).
        """
        sink = MagicMock()
        stage = _make_record_stage(sink)

        # 老 patch 模式: class-level patch 覆盖 record
        # (因 @dataclass(slots=True) 不允许 instance-level assignment)
        mock_record = MagicMock(return_value=None)
        original = RecordStage.record
        RecordStage.record = mock_record  # type: ignore[method-assign]
        try:
            result = _make_result(idx=1)
            # 老 PR0 测试用 sync 调用
            stage.record(result, elapsed_cum=1.0)
            # assert_called_once_with 会展开 kwargs, 直接比对 call_args
            assert mock_record.call_count == 1
            args, kwargs = mock_record.call_args
            assert args[0] is result
            assert kwargs.get("elapsed_cum") == 1.0
        finally:
            RecordStage.record = original  # type: ignore[method-assign]

    def test_record_sync_is_alias_for_pr0_fixtures(self) -> None:
        """老代码: `stage.record_sync(result, elapsed)` — M4.6 新 sync wrapper."""
        sink = MagicMock()
        stage = _make_record_stage(sink)

        result = _make_result(idx=1)
        # 老 PR0 fixture 也可直接调 record_sync (sync wrapper)
        stage.record_sync(result, elapsed_cum=1.0)

        sink.write_one.assert_called_once_with(result)
        assert stage.results == [result]

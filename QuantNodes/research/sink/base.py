"""Sink Protocol — output destination for FactorResult.

A Sink writes one FactorResult to some destination (file, DB, webhook, MQ, etc.).
Provides sync + async dual API. Async default implementation wraps sync via
``asyncio.to_thread`` — event loop is not blocked by I/O.

Sync API:
  - write_one(result) → Path: per-signal write (called in pipeline loop)
  - write_batch(results) → list[Path]: end-of-batch aggregation
  - flush() → None: close/cleanup

Async API (M4.4 / PR6.8):
  - write_one_async(result) → Path: defaults to ``await to_thread(write_one)``
  - write_batch_async(results) → list[Path]: defaults to ``await to_thread(write_batch)``
  - flush_async() → None: defaults to ``await to_thread(flush)``

Why a Sink Protocol (not ABC)?
  - Different sinks have wildly different interfaces (file vs webhook)
  - Structural typing lets duck-typed classes satisfy it without inheritance
  - PR4 provides 3 file-based sinks; future PRs can add webhook/MQ sinks

Why dual sync/async API?
  - Existing callers (record_stage, run_101_alphas_v2) use sync and must keep working
  - Async callers (future: l5_orchestrator, codegen_pipeline) need non-blocking I/O
  - Default ``to_thread`` implementation gives async benefit without rewriting sinks
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..backtest.base import FactorResult


class Sink(Protocol):
    """Output destination for FactorResult (file, DB, webhook, ...)."""

    # === Sync API (existing, 不动) ===

    def write_one(self, result: FactorResult) -> Path:
        """Write one FactorResult to the destination.

        Returns:
            Path to the written artifact (file path, message id, etc.).
            On error, may return a sentinel like Path("/dev/null") or raise.
        """
        ...

    def write_batch(self, results: list[FactorResult]) -> list[Path]:
        """End-of-batch aggregation (e.g. summary JSON/MD).

        Called once after all signals processed. Default returns [].
        """
        return []

    def flush(self) -> None:
        """Cleanup resources (close files, flush buffers). No-op default."""
        return None

    # === Async API (new in M4.4 / PR6.8) ===

    async def write_one_async(self, result: FactorResult) -> Path:
        """Async write one result.

        Default implementation offloads sync I/O to a thread pool via
        ``asyncio.to_thread``, so the event loop is not blocked.

        Subclasses can override for true async I/O (e.g. aiofiles, async DB).
        For PR6.8 we keep the default — sync I/O is fast enough (~50ms per
        factor) and our 3 file-based sinks work well with ``to_thread``.
        """
        return await asyncio.to_thread(self.write_one, result)

    async def write_batch_async(self, results: list[FactorResult]) -> list[Path]:
        """Async batch write. Default delegates to ``to_thread(write_batch)``."""
        return await asyncio.to_thread(self.write_batch, results)

    async def flush_async(self) -> None:
        """Async flush. Default delegates to ``to_thread(flush)``."""
        await asyncio.to_thread(self.flush)
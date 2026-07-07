"""BatchSummarySink — write multi_alpha_<paper_id>.json + .md at batch end.

Replaces v2's `FactorStage._write_summary`:
    BatchSerializer.write_json(self.results, self.config.output_dir / "multi_alpha_001_to_101.json")
    BatchSerializer.write_markdown(self.results, self.config.output_dir / "multi_alpha_summary.md")
    BatchReporter.log_summary(self.results)

PR5: delegates to `reporting.BatchSerializer` + `reporting.BatchReporter`
(refactored from PR4's inline aggregation). FactorResult list is converted
to dicts via `reporting.adapters.factor_results_to_dicts` so the same
serializer logic works for both v2 dict results and new FactorResult objects.

Output structure:
    output_dir/multi_alpha_<paper_id>.json    # aggregated metrics + per-alpha summary
    output_dir/multi_alpha_<paper_id>.md      # human-readable markdown table

Async API (M4.4 / PR6.8):
  - write_batch_async: same as sync, offloaded to thread
  - stream_write_async: NDJSON streaming, one result per line (no aiofiles)
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from ..reporting import (
    BatchReporter,
    BatchSerializer,
    factor_results_to_dicts,
)

if TYPE_CHECKING:
    from ..backtest.base import FactorResult

logger = logging.getLogger(__name__)


class BatchSummarySink:
    """Writes aggregated batch summary at end of pipeline.

    Args:
        output_dir: Directory to write summary files into.
        paper_id: Used in default filename (e.g. "101_alphas_minimal" → "multi_alpha_101_alphas_minimal.json").
        json_filename: Override JSON filename (e.g. "multi_alpha_001_to_101.json" for v2 compat).
        md_filename: Override MD filename (e.g. "multi_alpha_summary.md" for v2 compat).
        log_summary: If True, also call BatchReporter.log_summary() (default True).
    """

    def __init__(
        self,
        output_dir: Path,
        paper_id: str = "batch",
        json_filename: str | None = None,
        md_filename: str | None = None,
        log_summary: bool = True,
    ) -> None:
        self._dir = Path(output_dir)
        self._paper_id = paper_id
        self._json_filename = json_filename or f"multi_alpha_{paper_id}.json"
        self._md_filename = md_filename or f"multi_alpha_{paper_id}.md"
        self._log_summary = log_summary

    @property
    def output_dir(self) -> Path:
        return self._dir

    @property
    def paper_id(self) -> str:
        return self._paper_id

    @property
    def json_path(self) -> Path:
        return self._dir / self._json_filename

    @property
    def md_path(self) -> Path:
        return self._dir / self._md_filename

    def write_one(self, result: FactorResult) -> Path:
        """No-op: batch summary is only written at end.

        Returns Path("/dev/null") as sentinel.
        """
        return Path("/dev/null")

    def write_batch(self, results: list[FactorResult]) -> list[Path]:
        """Write aggregated JSON + Markdown summaries.

        Returns:
            List of paths written (typically 2: JSON + Markdown).
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        dicts = factor_results_to_dicts(results)
        json_path = self.json_path
        md_path = self.md_path
        try:
            BatchSerializer.write_json(dicts, json_path)
            paths.append(json_path)
        except Exception as exc:
            logger.warning("[sink] batch JSON failed: %s: %s", type(exc).__name__, exc)
        try:
            BatchSerializer.write_markdown(dicts, md_path)
            paths.append(md_path)
        except Exception as exc:
            logger.warning("[sink] batch MD failed: %s: %s", type(exc).__name__, exc)
        if self._log_summary:
            try:
                BatchReporter.log_summary(dicts)
            except Exception as exc:
                logger.warning("[sink] log summary failed: %s: %s", type(exc).__name__, exc)
        return paths

    def flush(self) -> None:
        """No-op."""
        return None

    # === Async API (M4.4 / PR6.8) ===

    async def write_batch_async(self, results: list[FactorResult]) -> list[Path]:
        """Async batch write — defaults to ``to_thread(write_batch)``.

        The aggregated JSON+MD are written together; offloading to a worker
        thread prevents blocking the event loop.
        """
        return await asyncio.to_thread(self.write_batch, results)

    async def flush_async(self) -> None:
        """Async flush — no-op for BatchSummarySink."""
        return None

    async def stream_write_async(
        self,
        results: AsyncIterator[FactorResult],
        ndjson_filename: str | None = None,
    ) -> list[Path]:
        """Stream-write results to NDJSON (newline-delimited JSON).

        Each result is appended as one JSON object per line. Uses
        ``loop.run_in_executor`` for the actual file write — no aiofiles
        dependency.

        Args:
            results: AsyncIterator yielding FactorResult one at a time.
            ndjson_filename: Override output filename. Default: ``{paper_id}_stream.ndjson``.

        Returns:
            List with one path (the NDJSON file) if any results were streamed;
            empty list otherwise.
        """
        ndjson_filename = ndjson_filename or f"{self._paper_id}_stream.ndjson"
        ndjson_path = self._dir / ndjson_filename
        loop = asyncio.get_event_loop()

        def _append(payload: str) -> None:
            with open(ndjson_path, "a", encoding="utf-8") as f:
                f.write(payload + "\n")

        self._dir.mkdir(parents=True, exist_ok=True)
        count = 0
        async for result in results:
            payload = json.dumps(
                factor_results_to_dicts([result])[0],
                ensure_ascii=False,
                default=str,
            )
            await loop.run_in_executor(None, _append, payload)
            count += 1

        return [ndjson_path] if count > 0 else []

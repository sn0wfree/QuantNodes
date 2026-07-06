"""QuantNodesBacktest — QuantNodes PipelineRunner adapter.

Replaces v2's `_run_pipeline_backtest` method. Wraps:
  1. `build_qn_config(factor_name, h5_path, code, config=run_config)`
  2. `PipelineRunner.from_dict(qn_config).run()` → ctx
  3. `extract_full_backtest_from_ctx(ctx)` → metrics dict

Configuration:
  - `config` (optional): RunConfig or similar — used by `build_qn_config` to
    fill in date ranges, groups, hedge, adj_mode, etc. If None, defaults
    are used (single-stock 5-group IC analyzer).
  - `factor_name_resolver` (optional): callable(signal) → str. Default
    sanitizes `signal.name` with the same regex the H5 writer uses
    (matching `safe_factor_name` in run_101_alphas_v2.py:683).

Why a separate `factor_name_resolver`?
  - `signal.name` may be Chinese (招商/浙商 broker reports) or hyphenated
    (`alpha-005`); H5 keys must be filesystem-safe.
  - The H5 writer (pipeline/data_loader.py:32) stores under
    `re.sub(r"[^A-Za-z0-9_]", "_", factor_name)` where factor_name is
    derived from `signal.name` (e.g. "alpha-005" → "alpha_005"). The
    reader MUST use the same key for lookup.
  - Using `signal.id` alone (just the numeric part "005") does NOT match
    the writer's convention — fixed in Phase B (M3 H5 bug).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..signal_source.base import Signal

logger = logging.getLogger(__name__)

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_]")


class QuantNodesBacktest:
    """QuantNodes PipelineRunner adapter for backtest execution.

    Args:
        config: Optional RunConfig-like object. If None, defaults are used.
        factor_name_resolver: callable(Signal) → str. Default sanitizes
            `signal.name` to match the H5 writer's key convention.

    Example:
        engine = QuantNodesBacktest(config=run_config)
        metrics = engine.run(
            code="def compute_factor(df): return df['close'].rank()",
            h5_path=Path("/data/alpha_001.h5"),
            signal=signal,
        )
        # metrics = {"ic_mean": 0.01, "icir": 0.1, "win_rate": 0.51, ...}
    """

    def __init__(
        self,
        config: Any = None,
        factor_name_resolver: Callable[[Signal], str] | None = None,
    ) -> None:
        self._config = config
        self._resolve = factor_name_resolver or self._default_resolver

    @staticmethod
    def _default_resolver(signal: Signal) -> str:
        """Default: sanitize `signal.name` with the same regex as the H5 writer.

        Phase B fix (M3 H5 key bug):
          - Writer (scripts/research/run_101_alphas_v2.py:683 + pipeline/
            data_loader.py:32) stores the factor wide DataFrame under the
            sanitized name `re.sub(r"[^A-Za-z0-9_]", "_", factor_name)`
            where `factor_name = signal.name = "alpha-{idx:03d}"`. The
            sanitized key becomes `alpha_{idx:03d}` (e.g. `alpha_005`).
          - This resolver previously returned `signal.id` (just `005`),
            causing `LoadDataNode` to raise `KeyError: /005 not found` at
            H5 lookup. Now it applies the SAME sanitization to
            `signal.name` so writer and reader agree on the key.
        """
        return _SAFE_KEY_RE.sub("_", signal.name)

    def run(
        self,
        code: str,
        h5_path: Path,
        signal: Signal,
    ) -> dict[str, Any]:
        """Run QuantNodes 12-node pipeline and extract metrics.

        Args:
            code: Generated Python function source.
            h5_path: Path to factor H5 file.
            signal: Source Signal (used for factor_name).

        Returns:
            Metrics dict from `extract_full_backtest_from_ctx`. On failure,
            returns `{"error": "..."}` (does NOT raise — caller decides).

        Raises:
            FileNotFoundError: If QuantNodes dependencies are missing.
        """
        from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner

        from QuantNodes.research.pipeline.backtest_config import build_qn_config

        from QuantNodes.research.pipeline.backtest_extract import extract_full_backtest_from_ctx


        factor_name: str = self._resolve(signal)
        logger.info("[backtest] factor=%s h5=%s", factor_name, h5_path.name)
        try:
            qn_config = build_qn_config(
                factor_name=factor_name,
                h5_path=h5_path,
                expression=code,
                config=self._config,
            )
            runner = PipelineRunner.from_dict(qn_config)
            ctx = runner.run()
            return extract_full_backtest_from_ctx(ctx)
        except Exception as exc:
            logger.warning(
                "[backtest] %s failed: %s: %s",
                factor_name, type(exc).__name__, str(exc)[:100],
            )
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "ic_mean": None,
                "icir": None,
                "win_rate": None,
            }

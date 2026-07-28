"""Block bootstrap stability test.

Implements Sensitivity Phase 3 (v7.6): block_size=63 ~ 1 quarter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BootstrapResult:
    """Result of bootstrap stability test."""

    cv: float
    ci_lower: float
    ci_upper: float
    status: str
    n_bootstrap: int
    block_size: int
    mean_metric: float
    std_metric: float

    def to_dict(self) -> dict:
        return {
            "cv": self.cv,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "status": self.status,
            "n_bootstrap": self.n_bootstrap,
            "block_size": self.block_size,
            "mean_metric": self.mean_metric,
            "std_metric": self.std_metric,
        }


class BootstrapStability:
    """Block bootstrap stability test.

    The backtest_fn should accept data indices and return a metric (e.g., Sharpe).
    """

    def __init__(
        self,
        n_bootstrap: int = 30,
        block_size: int = 63,
        threshold_pass: float = 0.25,
        threshold_deprecate: float = 0.50,
        seed: int | None = None,
    ):
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.threshold_pass = threshold_pass
        self.threshold_deprecate = threshold_deprecate
        self.seed = seed

    def run(
        self,
        backtest_fn: Callable[..., float],
        data_length: int | None = None,
    ) -> BootstrapResult:
        """Run bootstrap stability test.

        Args:
            backtest_fn: Callable that takes resampled indices and returns metric
            data_length: Total length of data (if backtest_fn needs it)

        Returns:
            BootstrapResult
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        rng = np.random.default_rng(self.seed)
        n = data_length or 252

        metrics: list[float] = []
        for _ in range(self.n_bootstrap):
            # Sample blocks of indices
            n_blocks = max(1, n // self.block_size)
            blocks = [
                rng.integers(0, n - self.block_size)
                for _ in range(n_blocks)
            ]
            indices = np.concatenate(
                [np.arange(b, b + self.block_size) for b in blocks]
            )
            indices = indices[indices < n]

            try:
                metric = backtest_fn(indices)
                if metric is not None:
                    metrics.append(float(metric))
            except Exception:
                continue

        if not metrics:
            return BootstrapResult(
                cv=float("inf"),
                ci_lower=0.0,
                ci_upper=0.0,
                status="INSUFFICIENT_DATA",
                n_bootstrap=self.n_bootstrap,
                block_size=self.block_size,
                mean_metric=0.0,
                std_metric=0.0,
            )

        mean = float(np.mean(metrics))
        std = float(np.std(metrics))
        cv = std / abs(mean) if abs(mean) > 1e-9 else float("inf")
        ci_lower = float(np.percentile(metrics, 2.5))
        ci_upper = float(np.percentile(metrics, 97.5))

        if cv < self.threshold_pass:
            status = "PASS"
        elif cv < self.threshold_deprecate:
            status = "PROMISING"
        else:
            status = "DEPRECATED"

        return BootstrapResult(
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            status=status,
            n_bootstrap=self.n_bootstrap,
            block_size=self.block_size,
            mean_metric=mean,
            std_metric=std,
        )

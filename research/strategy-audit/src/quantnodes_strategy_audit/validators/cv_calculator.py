"""CV% calculator: start-date dependency test.

Implements L-203: start-date CV% < 25% PASS / 25-50% PROMISING / > 50% DEPRECATED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CVTestResult:
    """Result of a CV% test."""

    cv: float
    mean: float
    std: float
    status: str
    n_starts: int
    calmars: list[float]

    def to_dict(self) -> dict:
        return {
            "cv": self.cv,
            "mean": self.mean,
            "std": self.std,
            "status": self.status,
            "n_starts": self.n_starts,
            "calmars": self.calmars,
        }


class CVCalculator:
    """Compute start-date dependency CV% (L-203).

    Each start_date should be passed as a string 'YYYY-MM-DD'.
    The backtest_fn should accept start_date and return OOS Calmar.
    """

    def __init__(
        self,
        threshold_pass: float = 0.25,
        threshold_deprecate: float = 0.50,
    ):
        self.threshold_pass = threshold_pass
        self.threshold_deprecate = threshold_deprecate

    def run(
        self,
        backtest_fn: Callable[..., float],
        start_dates: list[str],
    ) -> CVTestResult:
        """Run CV% test across start dates.

        Args:
            backtest_fn: Callable that takes start_date and returns OOS Calmar
            start_dates: List of 'YYYY-MM-DD' strings

        Returns:
            CVTestResult with cv, mean, std, status, calmars
        """
        calmars: list[float] = []
        for start_date in start_dates:
            try:
                calmar = backtest_fn(start_date)
                if calmar is not None:
                    calmars.append(float(calmar))
            except Exception:
                continue

        if len(calmars) < 2:
            return CVTestResult(
                cv=float("inf"),
                mean=float(np.mean(calmars)) if calmars else 0.0,
                std=0.0,
                status="INSUFFICIENT_DATA",
                n_starts=len(calmars),
                calmars=calmars,
            )

        mean = float(np.mean(calmars))
        std = float(np.std(calmars))
        cv = std / abs(mean) if abs(mean) > 1e-9 else float("inf")

        if cv < self.threshold_pass:
            status = "PASS"
        elif cv < self.threshold_deprecate:
            status = "PROMISING"
        else:
            status = "DEPRECATED"

        return CVTestResult(
            cv=cv,
            mean=mean,
            std=std,
            status=status,
            n_starts=len(calmars),
            calmars=calmars,
        )

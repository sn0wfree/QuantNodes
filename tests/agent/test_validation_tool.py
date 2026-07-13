"""Tests for ValidationTool (nanobot entry point)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from QuantNodes.agent.tools.validation import ValidationTool


def _make_etf_records(n_days: int = 1200, n_codes: int = 10, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    codes = [f"E{i:03d}" for i in range(n_codes)]
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, (n_days, n_codes))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, index=idx, columns=codes)
    return [
        {"date": d.strftime("%Y-%m-%d"), "code": c, "close": float(df.loc[d, c])}
        for d in df.index for c in codes
    ]


class TestValidationTool:
    def test_metadata(self) -> None:
        tool = ValidationTool()
        assert tool.name == "quant_validation"
        assert "起点依赖" in tool.description
        assert "etf_nav" in tool.parameters["properties"]

    def test_run_full(self) -> None:
        tool = ValidationTool()
        records = _make_etf_records()
        result = asyncio.run(tool.execute(
            etf_nav=records,
            lookback=120,
            top_n=5,
            actions=["all"],
            start_points=["2018-01-01", "2020-01-01"],
        ))
        assert result.success
        assert "report_markdown" in result.content
        assert "起点依赖" in result.content["report_markdown"]
        assert "passed" in result.content

    def test_run_single_action(self) -> None:
        tool = ValidationTool()
        records = _make_etf_records()
        result = asyncio.run(tool.execute(
            etf_nav=records,
            lookback=120,
            top_n=5,
            actions=["start"],
            start_points=["2018-01-01", "2020-01-01"],
        ))
        assert result.success
        assert "results" in result.content
        assert len(result.content["results"]) == 1

    def test_empty_etf(self) -> None:
        tool = ValidationTool()
        result = asyncio.run(tool.execute(etf_nav=[]))
        assert not result.success
        assert "etf_nav 为空" in result.error

    def test_missing_date_column(self) -> None:
        tool = ValidationTool()
        result = asyncio.run(tool.execute(etf_nav=[{"foo": 1}]))
        assert not result.success

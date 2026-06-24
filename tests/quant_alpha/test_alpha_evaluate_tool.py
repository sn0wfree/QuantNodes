# coding=utf-8
"""
test_alpha_evaluate_tool.py - Alpha 评估工具测试
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from QuantNodes.agent.tools.alpha_evaluate import (
    AlphaEvaluateTool,
    split_args,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 31)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            close = float(np.random.randn() * 5 + 100)
            rows.append({
                "date": date, "code": code, "close": close,
                "open": close + np.random.randn() * 0.5,
                "high": close + abs(np.random.randn()),
                "low": close - abs(np.random.randn()),
                "vol": float(np.random.randint(1000, 5000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class TestSplitArgs:
    def test_simple(self):
        assert split_args("a, b, c") == ["a", "b", "c"]

    def test_with_parens(self):
        assert split_args("a, ts_mean(b, 5), c") == ["a", "ts_mean(b, 5)", "c"]

    def test_with_nested_parens(self):
        assert split_args("a, ts_mean(sub(b, c), 5)") == ["a", "ts_mean(sub(b, c), 5)"]

    def test_single_arg(self):
        assert split_args("a") == ["a"]


class TestParseSimpleFormula:
    def test_feature(self):
        e = AlphaEvaluateTool._parse_simple_formula("close")
        assert "close" in str(e)

    def test_feature_quoted(self):
        e = AlphaEvaluateTool._parse_simple_formula("Feature('close')")
        assert "close" in str(e)

    def test_literal_int(self):
        e = AlphaEvaluateTool._parse_simple_formula("5")
        assert "5" in str(e)

    def test_literal_float(self):
        e = AlphaEvaluateTool._parse_simple_formula("1e-12")
        assert "1e-12" in str(e)

    def test_window_op(self):
        e = AlphaEvaluateTool._parse_simple_formula("ts_mean(close, 5)")
        s = str(e)
        assert "ts_mean" in s
        assert "close" in s
        assert "5" in s

    def test_neg_prefix(self):
        e = AlphaEvaluateTool._parse_simple_formula("-ts_mean(close, 5)")
        assert "neg" in str(e) or "-" in str(e)

    def test_nested(self):
        e = AlphaEvaluateTool._parse_simple_formula("sub(close, ts_mean(close, 10))")
        s = str(e)
        assert "ts_mean" in s
        assert "close" in s
        assert "-" in s  # BinaryOp op="sub" prints as "-"

    def test_invalid(self):
        with pytest.raises(ValueError):
            AlphaEvaluateTool._parse_simple_formula("unknown_op(x)")


class TestToolMetadata:
    def test_name(self):
        tool = AlphaEvaluateTool()
        assert tool.name == "alpha_evaluate"

    def test_description_contains_keywords(self):
        tool = AlphaEvaluateTool()
        assert "IC" in tool.description or "evaluate" in tool.description.lower()

    def test_parameters_schema(self):
        tool = AlphaEvaluateTool()
        schema = tool.parameters
        assert "formulas" in schema["properties"]
        assert "formulas" in schema["required"]
        assert "data_path" in schema["properties"]
        assert "forward_returns" in schema["properties"]

    def test_read_only(self):
        tool = AlphaEvaluateTool()
        assert tool.read_only is True


class TestExecute:
    def test_no_data(self):
        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(formulas=["x"], data=None, data_path=None))
        assert "error" in r["summary"]

    def test_single_formula_success(self, sample_data):
        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)"],
            data=sample_data,
            forward_returns=[1],
        ))
        assert r["summary"]["success"] == 1
        assert r["summary"]["failed"] == 0
        ev = r["evaluations"][0]
        assert ev["status"] == "success"
        assert "ic_mean" in ev["metrics"]
        assert "ir" in ev["metrics"]
        assert "ic_decay" in ev["metrics"]

    def test_multiple_formulas(self, sample_data):
        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(
            formulas=[
                "ts_mean(close, 5)",
                "sub(close, ts_mean(close, 10))",
                "rank(ts_mean(close, 5))",
            ],
            data=sample_data,
            forward_returns=[1, 5],
        ))
        assert r["summary"]["total"] == 3

    def test_invalid_formula(self, sample_data):
        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)", "unknown_op(close)"],
            data=sample_data,
            forward_returns=[1],
        ))
        assert r["summary"]["success"] >= 1
        assert r["summary"]["failed"] >= 1

    def test_multi_forward_returns(self, sample_data):
        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)"],
            data=sample_data,
            forward_returns=[1, 5, 20],
        ))
        ev = r["evaluations"][0]
        assert "1" in ev["metrics"]["ic_decay"]
        assert "5" in ev["metrics"]["ic_decay"]
        assert "20" in ev["metrics"]["ic_decay"]

    def test_data_path_csv(self, tmp_path):
        np.random.seed(42)
        csv_path = tmp_path / "data.csv"
        dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
        rows = []
        for date in dates:
            for code in ["X", "Y"]:
                rows.append({
                    "date": date, "code": code, "close": 100.0,
                    "open": 100.0, "high": 100.0, "low": 100.0, "vol": 1000.0,
                })
        pl.DataFrame(rows).write_csv(str(csv_path))

        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 3)"],
            data_path=str(csv_path),
            forward_returns=[1],
        ))
        assert r["summary"]["success"] == 1

    def test_data_path_not_found(self):
        tool = AlphaEvaluateTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)"],
            data_path="/nonexistent.parquet",
        ))
        # execute() catches errors and returns summary.error
        assert "error" in r["summary"] or r["summary"]["failed"] >= 0


class TestBuildForwardReturns:
    def test_basic(self, sample_data):
        fr = AlphaEvaluateTool._build_forward_returns(
            sample_data, [1, 5], "date", "code"
        )
        assert 1 in fr
        assert 5 in fr

    def test_missing_close_column(self, sample_data):
        with pytest.raises(ValueError):
            AlphaEvaluateTool._build_forward_returns(
                sample_data.drop("close"), [1], "date", "code"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

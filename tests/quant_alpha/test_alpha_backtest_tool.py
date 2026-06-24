# coding=utf-8
"""
test_alpha_backtest_tool.py - Alpha Trading 回测工具测试
"""

from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import pytest

from QuantNodes.agent.tools.alpha_backtest import AlphaBacktestTool


@pytest.fixture
def sample_data() -> pl.DataFrame:
    import datetime as dt
    np.random.seed(42)
    base = dt.date(2024, 1, 1)
    dates = [(base + dt.timedelta(days=i)).isoformat() for i in range(60)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E", "F", "G", "H"]:
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


class TestToolMetadata:
    def test_name(self):
        assert AlphaBacktestTool().name == "alpha_backtest"

    def test_description(self):
        d = AlphaBacktestTool().description
        assert "Trading" in d or "backtest" in d.lower()

    def test_parameters(self):
        params = AlphaBacktestTool().parameters
        assert "formulas" in params["properties"]
        assert "top_k" in params["properties"]
        assert "initial_cash" in params["properties"]

    def test_read_only(self):
        assert AlphaBacktestTool().read_only is True


class TestExecute:
    def test_no_data(self):
        tool = AlphaBacktestTool()
        r = run_async(tool.execute(formulas=["x"], data=None, data_path=None))
        assert "error" in r["summary"]

    def test_single_formula_success(self, sample_data):
        tool = AlphaBacktestTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)"],
            data=sample_data,
            top_k=2,
            rebalance_freq=10,
            initial_cash=1_000_000.0,
        ))
        bt = r["backtests"][0]
        assert bt["status"] == "success"
        for k in ("annual_return", "sharpe", "max_drawdown", "win_rate"):
            assert k in bt["backtest"]

    def test_long_short(self, sample_data):
        tool = AlphaBacktestTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)"],
            data=sample_data,
            top_k=2, bottom_k=2,
            rebalance_freq=10,
        ))
        bt = r["backtests"][0]
        assert bt["status"] == "success"

    def test_invalid_formula(self, sample_data):
        tool = AlphaBacktestTool()
        r = run_async(tool.execute(
            formulas=["unknown_op(close, 5)"],
            data=sample_data,
            top_k=2,
        ))
        assert r["summary"]["failed"] >= 1

    def test_summary_fields(self, sample_data):
        tool = AlphaBacktestTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)", "sub(close, ts_mean(close, 10))"],
            data=sample_data,
            top_k=2,
        ))
        for k in ("total", "success", "failed", "avg_sharpe", "best_sharpe"):
            assert k in r["summary"]

    def test_max_drawdown_negative(self, sample_data):
        """max_drawdown 应该是 ≤ 0"""
        tool = AlphaBacktestTool()
        r = run_async(tool.execute(
            formulas=["ts_mean(close, 5)"],
            data=sample_data,
            top_k=2,
        ))
        bt = r["backtests"][0]["backtest"]
        assert bt["max_drawdown"] <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

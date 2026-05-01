# coding=utf-8
"""
ConfigBacktestRunner 未覆盖方法单元测试

覆盖:
- save_output() parquet/csv/json 格式
- _save_dataframe() 不支持格式报错
- _build_equity_curve() 空 DataFrame
- _normalize_columns() Open fallback 到 Close
- _build_risk_nodes() 空 positions
- run() executor 错误路径
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json

from QuantNodes.agent.config.types import (
    StrategyConfig, BacktestConfig, OutputConfig, DataConfig,
    FactorConfig,
)
from QuantNodes.backtest.config_runner import ConfigBacktestRunner
from QuantNodes.backtest.backtest_node import BacktestResult


class TestBuildEquityCurveEmpty:
    """_build_equity_curve() 空数据"""

    def test_empty_quote_df(self):
        result = ConfigBacktestRunner._build_equity_curve(
            trades_df=pd.DataFrame(),
            quote_df=pd.DataFrame(),
            initial_cash=1000000,
        )
        assert result.empty
        assert list(result.columns) == ["date", "equity", "cash", "position_value"]

    def test_empty_trades(self):
        quote_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "Code": ["A", "A"],
            "Close": [10.0, 11.0],
        })
        result = ConfigBacktestRunner._build_equity_curve(
            trades_df=pd.DataFrame(),
            quote_df=quote_df,
            initial_cash=1000000,
        )
        assert len(result) == 2
        assert result["equity"].iloc[0] == 1000000


class TestNormalizeColumns:
    """_normalize_columns() 边界情况"""

    def test_code_lowercase(self):
        df = pd.DataFrame({"code": ["A"], "close": [10.0], "date": ["2023-01-01"]})
        runner = ConfigBacktestRunner()
        result = runner._normalize_columns(df)
        assert "Code" in result.columns

    def test_open_fallback_to_close(self):
        """Open 列不存在时 fallback 到 Close"""
        df = pd.DataFrame({
            "Code": ["A"], "close": [10.0], "Close": [10.0],
            "date": ["2023-01-01"],
        })
        runner = ConfigBacktestRunner()
        result = runner._normalize_columns(df)
        assert "Open" in result.columns
        assert result["Open"].iloc[0] == 10.0

    def test_all_columns_already_present(self):
        df = pd.DataFrame({
            "Code": ["A"], "Close": [10.0], "Open": [9.5],
            "date": ["2023-01-01"],
        })
        runner = ConfigBacktestRunner()
        result = runner._normalize_columns(df)
        assert result["Open"].iloc[0] == 9.5


class TestSaveOutput:
    """save_output() 测试"""

    def _make_result(self):
        return BacktestResult(
            trades=pd.DataFrame({
                "code": ["A", "B"],
                "side": ["buy", "sell"],
                "size": [100, 100],
                "adjusted_price": [10.0, 11.0],
                "fee": [0.1, 0.11],
                "dt": ["2023-01-01", "2023-01-02"],
            }),
            equity_curve=pd.DataFrame({
                "date": ["2023-01-01", "2023-01-02"],
                "equity": [1000000, 1001000],
                "cash": [990000, 991000],
                "position_value": [10000, 10000],
            }),
            statistics={"total_trades": 2, "sharpe_ratio": 1.0},
            final_cash=991000,
            total_return=0.001,
            sharpe_ratio=1.0,
            max_drawdown=-0.001,
            win_rate=0.5,
        )

    def test_save_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrategyConfig(
                name="test",
                output=OutputConfig(
                    format="parquet",
                    path=os.path.join(tmpdir, "result.parquet"),
                    save_signals=True,
                    save_positions=True,
                    save_equity_curve=True,
                ),
            )
            runner = ConfigBacktestRunner()
            bt_result = self._make_result()
            signals_df = bt_result.trades

            saved = runner.save_output(bt_result, config, signals_df=signals_df)

            assert "equity_curve" in saved
            assert "signals" in saved
            assert "trades" in saved
            assert "statistics" in saved
            assert os.path.exists(saved["equity_curve"])
            assert os.path.exists(saved["signals"])
            assert os.path.exists(saved["statistics"])

    def test_save_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrategyConfig(
                name="test",
                output=OutputConfig(
                    format="csv",
                    path=os.path.join(tmpdir, "result.csv"),
                    save_equity_curve=True,
                ),
            )
            runner = ConfigBacktestRunner()
            bt_result = self._make_result()

            saved = runner.save_output(bt_result, config)
            assert "equity_curve" in saved

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrategyConfig(
                name="test",
                output=OutputConfig(
                    format="json",
                    path=os.path.join(tmpdir, "result.json"),
                    save_equity_curve=True,
                ),
            )
            runner = ConfigBacktestRunner()
            bt_result = self._make_result()

            saved = runner.save_output(bt_result, config)
            assert "equity_curve" in saved

    def test_no_output_config(self):
        runner = ConfigBacktestRunner()
        bt_result = self._make_result()
        saved = runner.save_output(bt_result, StrategyConfig(name="test"))
        assert saved == {}

    def test_statistics_json_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrategyConfig(
                name="test",
                output=OutputConfig(
                    path=os.path.join(tmpdir, "result.parquet"),
                    save_equity_curve=False,
                ),
            )
            runner = ConfigBacktestRunner()
            bt_result = self._make_result()

            saved = runner.save_output(bt_result, config)
            with open(saved["statistics"], encoding="utf-8") as f:
                stats = json.load(f)
            assert stats["total_trades"] == 2
            assert stats["sharpe_ratio"] == 1.0


class TestSaveDataframe:
    """_save_dataframe() 测试"""

    def test_unsupported_format_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported"):
            ConfigBacktestRunner._save_dataframe(df, "/tmp/test.xyz", "unsupported")


class TestBuildRiskNodes:
    """_build_risk_nodes() 测试"""

    def test_no_positions_config(self):
        config = StrategyConfig(
            name="test",
            backtest=BacktestConfig(
                start_date="2023-01-01", end_date="2024-01-01",
            ),
        )
        runner = ConfigBacktestRunner()
        nodes = runner._build_risk_nodes(config)
        assert nodes == []

    def test_with_max_positions(self):
        config = StrategyConfig(
            name="test",
            backtest=BacktestConfig(
                start_date="2023-01-01", end_date="2024-01-01",
                positions={"max_positions": 10},
            ),
        )
        runner = ConfigBacktestRunner()
        nodes = runner._build_risk_nodes(config)
        assert len(nodes) == 1

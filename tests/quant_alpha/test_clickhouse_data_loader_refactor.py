# coding=utf-8
"""
test_clickhouse_data_loader_refactor.py

P2.12c.3/c.4 重构后的 ClickHouseDataLoader 单元测试。

重构:
- 原 `_query_clickhouse()` 用 raw http.client, 现委托 ClickHouseNode.query()
- 原 `load_summary()` 同上

测试目标:
- mock ClickHouseNode.query() 验证两个方法都正确委托
- pandas → polars 转换正确
- 错误路径 (query 失败 / 返回空) 优雅处理
- ClickHouseNode 实例化时使用正确的 host/port/user/password
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation.clickhouse_data_loader import (
    ClickHouseDataLoader,
)


@pytest.fixture
def loader():
    """标准 ClickHouseDataLoader fixture。"""
    return ClickHouseDataLoader(
        host="localhost",
        port=8123,
        user="default",
        password="",
        table="quote.stock_quote",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )


@pytest.fixture
def mock_clickhouse_node():
    """mock ClickHouseNode 类 (在 clickhouse_data_loader 内部 patch)。"""
    with patch(
        "QuantNodes.database_node.clickhouse_node.ClickHouseNode"
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


# ==============================================================================
# _query_clickhouse
# ==============================================================================


class TestQueryClickhouse:
    def test_uses_clickhouse_node_query(self, loader, mock_clickhouse_node):
        """_query_clickhouse 委托 ClickHouseNode.query() 而非 raw HTTP。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame({
            "code": ["A", "B"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "vol": [1000.0, 2000.0],
            "amount": [10000.0, 20000.0],
            "date": ["2020-01-01", "2020-01-02"],
        })

        df = loader._query_clickhouse()

        # ClickHouseNode.query() 被调用 1 次
        mock_clickhouse_node.query.assert_called_once()
        # 返回 polars DataFrame
        assert isinstance(df, pl.DataFrame)
        assert df.height == 2

    def test_passes_correct_connection_params(self, loader):
        """ClickHouseNode 实例化时使用 loader 的 host/port/user/password。"""
        loader = ClickHouseDataLoader(
            host="myhost",
            port=9999,
            user="alice",
            password="secret",
        )

        with patch(
            "QuantNodes.database_node.clickhouse_node.ClickHouseNode"
        ) as mock_cls:
            mock_cls.return_value.query.return_value = pd.DataFrame({"x": [1]})
            loader._query_clickhouse()

            # ClickHouseNode 用正确的参数实例化
            mock_cls.assert_called_once_with(
                host="myhost",
                port=9999,
                user="alice",
                passwd="secret",
                database="default",
            )

    def test_pandas_to_polars_conversion(self, loader, mock_clickhouse_node):
        """pandas DataFrame 正确转换为 polars。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame({
            "code": ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        })

        df = loader._query_clickhouse()

        assert isinstance(df, pl.DataFrame)
        assert "code" in df.columns
        assert "value" in df.columns
        assert df.height == 3
        assert df["value"].to_list() == [1.0, 2.0, 3.0]

    def test_query_failure_raises_runtime_error(self, loader, mock_clickhouse_node):
        """ClickHouseNode.query() 抛异常时, _query_clickhouse 抛 RuntimeError。"""
        mock_clickhouse_node.query.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="ClickHouse query failed"):
            loader._query_clickhouse()

    def test_empty_result_raises_runtime_error(self, loader, mock_clickhouse_node):
        """ClickHouseNode 返回空 DataFrame 时, _query_clickhouse 抛 RuntimeError。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame()

        with pytest.raises(RuntimeError, match="ClickHouse returned empty result"):
            loader._query_clickhouse()

    def test_none_result_raises_runtime_error(self, loader, mock_clickhouse_node):
        """ClickHouseNode 返回 None 时, _query_clickhouse 抛 RuntimeError。"""
        mock_clickhouse_node.query.return_value = None

        with pytest.raises(RuntimeError, match="ClickHouse returned empty result"):
            loader._query_clickhouse()

    def test_sql_contains_field_mapping(self, loader, mock_clickhouse_node):
        """生成的 SQL 包含 FIELD_MAP 的所有列别名。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame({"x": [1]})

        loader._query_clickhouse()
        sql = mock_clickhouse_node.query.call_args[0][0]

        # FIELD_MAP 的所有目标列名应出现在 SQL 中
        for pl_name in ClickHouseDataLoader.FIELD_MAP.values():
            assert pl_name in sql, f"{pl_name} not in SQL: {sql[:200]}"

        # WHERE 子句
        assert "WHERE trade_date >=" in sql
        assert "2020-01-01" in sql
        assert "2020-12-31" in sql


# ==============================================================================
# load_summary
# ==============================================================================


class TestLoadSummary:
    def test_returns_dict_from_first_row(self, loader, mock_clickhouse_node):
        """load_summary 返回 pandas 第一行 dict。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame([{
            "min_date": "2020-01-01",
            "max_date": "2020-12-31",
            "total_rows": 1000,
            "n_stocks": 50,
        }])

        summary = loader.load_summary()

        assert summary["total_rows"] == 1000
        assert summary["n_stocks"] == 50
        assert summary["min_date"] == "2020-01-01"
        assert summary["max_date"] == "2020-12-31"

    def test_returns_error_on_query_exception(self, loader, mock_clickhouse_node):
        """query() 抛异常时, 返回 {'error': ...}。"""
        mock_clickhouse_node.query.side_effect = RuntimeError("connection timeout")

        summary = loader.load_summary()

        assert "error" in summary

    def test_returns_error_on_empty_result(self, loader, mock_clickhouse_node):
        """返回空 DataFrame 时, 返回 {'error': ...}。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame()

        summary = loader.load_summary()

        assert "error" in summary

    def test_returns_error_on_none_result(self, loader, mock_clickhouse_node):
        """返回 None 时, 返回 {'error': ...}。"""
        mock_clickhouse_node.query.return_value = None

        summary = loader.load_summary()

        assert "error" in summary

    def test_handles_timestamp_serialization(self, loader, mock_clickhouse_node):
        """pandas Timestamp 字段自动转为 ISO string (避免 JSON 失败)。"""
        mock_clickhouse_node.query.return_value = pd.DataFrame([{
            "min_date": pd.Timestamp("2020-01-01"),
            "max_date": pd.Timestamp("2020-12-31"),
            "total_rows": 1000,
            "n_stocks": 50,
        }])

        summary = loader.load_summary()

        # Timestamp → ISO string
        assert isinstance(summary["min_date"], str)
        assert "2020-01-01" in summary["min_date"]


# ==============================================================================
# ClickHouseNode 实例化参数验证
# ==============================================================================


class TestClickHouseNodeInstantiation:
    """验证 ClickHouseDataLoader 委托时使用了正确的 ClickHouseNode 参数。"""

    def test_query_clickhouse_uses_loader_config(self, loader):
        """_query_clickhouse 实例化的 ClickHouseNode 使用 loader 的所有连接参数。"""
        with patch(
            "QuantNodes.database_node.clickhouse_node.ClickHouseNode"
        ) as mock_cls:
            mock_cls.return_value.query.return_value = pd.DataFrame({"x": [1]})
            loader._query_clickhouse()
            # ClickHouseNode 实例化 1 次, 参数来自 loader
            assert mock_cls.call_count == 1
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 8123
            assert call_kwargs["user"] == "default"
            assert call_kwargs["passwd"] == ""

    def test_load_summary_uses_loader_config(self, loader):
        """load_summary 实例化的 ClickHouseNode 使用 loader 的所有连接参数。"""
        with patch(
            "QuantNodes.database_node.clickhouse_node.ClickHouseNode"
        ) as mock_cls:
            mock_cls.return_value.query.return_value = pd.DataFrame([{"x": 1}])
            loader.load_summary()
            assert mock_cls.call_count == 1
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 8123
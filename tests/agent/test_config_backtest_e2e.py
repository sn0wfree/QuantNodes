# coding=utf-8
"""
核心功能1 端到端集成测试

测试完整链路: YAML配置 → ConfigLoader → ConfigExecutor → ConfigBacktestRunner → 结果
"""

import asyncio
import os
import tempfile
import pytest
import polars as pl

from QuantNodes.agent.config.loader import ConfigLoader
from QuantNodes.agent.config.types import (
    StrategyConfig,
    DataConfig,
    FactorConfig,
    OperationConfig,
    CompositeConfig,
    BacktestConfig,
)


def _make_sample_csv(path: str):
    """创建示例 CSV 数据文件"""
    df = pl.DataFrame({
        "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
        "code": ["A", "B", "C", "D"] * 3,
        "open": [100.0, 200.0, 50.0, 300.0,
                 101.0, 201.0, 51.0, 301.0,
                 102.0, 202.0, 52.0, 302.0],
        "high": [105.0, 205.0, 55.0, 305.0,
                 106.0, 206.0, 56.0, 306.0,
                 107.0, 207.0, 57.0, 307.0],
        "low": [95.0, 195.0, 45.0, 295.0,
                96.0, 196.0, 46.0, 296.0,
                97.0, 197.0, 47.0, 297.0],
        "close": [102.0, 202.0, 52.0, 302.0,
                  103.0, 203.0, 53.0, 303.0,
                  104.0, 204.0, 54.0, 304.0],
        "volume": [1000, 2000, 500, 3000,
                   1100, 2100, 600, 3100,
                   1200, 2200, 700, 3200],
    })
    df.write_csv(path)


def _make_sample_csv_custom_columns(path: str):
    """创建带非标准列名的 CSV 数据文件"""
    df = pl.DataFrame({
        "trade_date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
        "ts_code": ["A", "B", "C", "D"] * 2,
        "close": [100.0, 200.0, 50.0, 300.0, 101.0, 201.0, 51.0, 301.0],
        "open": [99.0, 199.0, 49.0, 299.0, 100.0, 200.0, 50.0, 300.0],
        "vol": [1000, 2000, 500, 3000, 1100, 2100, 600, 3100],
    })
    df.write_csv(path)


class TestE2E_CSV_FullPipeline:
    """CSV 文件端到端: YAML → 回测 → 统计结果"""

    def test_momentum_strategy_csv(self):
        """动量因子策略: CSV数据 → 完整回测"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "stock_data.csv")
            _make_sample_csv(csv_path)

            yaml_str = f"""
name: "momentum_e2e"
description: "端到端动量策略测试"

data:
  source: csv
  path: "{csv_path}"
  columns: [date, code, open, high, low, close, volume]
  date_column: date
  code_column: code

factors:
  - name: ret
    expr: "close / open - 1"
    description: "日内收益率"

operations:
  - type: time_series
    name: ret_ma
    category: ts_mean
    inputs: [ret]
    params:
      window: 2

  - type: section
    name: ret_rank
    category: rank
    inputs: [ret_ma]

composite:
  - name: alpha
    formula: "ret_rank"

backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-03"
  initial_cash: 1000000
  commission: 0.001
  slippage: 0.001
  signals:
    buy_threshold: 0.6
    sell_threshold: 0.4
"""
            tool = ConfigBacktestTool()
            result = asyncio.run(tool.execute(config_yaml=yaml_str))

            assert result["status"] == "success"
            assert result["summary"]["final_cash"] > 0
            assert result["summary"]["trading_days"] > 0
            assert result["config_info"]["name"] == "momentum_e2e"
            assert result["config_info"]["factors"] == 1
            assert result["config_info"]["operations"] == 2

    def test_simple_factor_only(self):
        """仅因子计算: 无 operations/composite"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            _make_sample_csv(csv_path)

            yaml_str = f"""
name: "simple_factor"
data:
  source: csv
  path: "{csv_path}"
factors:
  - name: ret
    expr: "close / open - 1"
backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-03"
  initial_cash: 1000000
  signals:
    buy_threshold: 0.01
    sell_threshold: -0.01
"""
            tool = ConfigBacktestTool()
            result = asyncio.run(tool.execute(config_yaml=yaml_str))

            assert result["status"] == "success"
            assert result["summary"]["final_cash"] > 0

    def test_override_params(self):
        """参数覆盖: start_date, end_date, initial_cash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            _make_sample_csv(csv_path)

            yaml_str = f"""
name: "override_test"
data:
  source: csv
  path: "{csv_path}"
factors:
  - name: ret
    expr: "close / open - 1"
backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-03"
  initial_cash: 1000000
  signals:
    buy_threshold: 0.01
    sell_threshold: -0.01
"""
            tool = ConfigBacktestTool()
            result = asyncio.run(tool.execute(
                config_yaml=yaml_str,
                start_date="2024-01-02",
                initial_cash=500000,
            ))

            assert result["status"] == "success"
            assert result["summary"]["final_cash"] > 0


class TestE2E_CustomColumnMapping:
    """列名映射端到端测试"""

    def test_column_mapping_csv(self):
        """非标准列名 → 标准列名"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "custom_cols.csv")
            _make_sample_csv_custom_columns(csv_path)

            yaml_str = f"""
name: "column_mapping_test"
data:
  source: csv
  path: "{csv_path}"
  date_column: date
  code_column: code
  column_mapping:
    trade_date: date
    ts_code: code
    vol: volume
factors:
  - name: ret
    expr: "close / open - 1"
backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  initial_cash: 1000000
  signals:
    buy_threshold: 0.01
    sell_threshold: -0.01
"""
            tool = ConfigBacktestTool()
            result = asyncio.run(tool.execute(config_yaml=yaml_str))

            assert result["status"] == "success"
            assert result["summary"]["final_cash"] > 0


class TestE2E_DuckDB:
    """DuckDB 端到端: 内存/文件模式"""

    def test_duckdb_memory(self):
        """DuckDB 内存模式: 插入数据 → 查询 → 回测"""
        from QuantNodes.database_node import DuckDBNode

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.duckdb")

            # 创建测试数据并写入 DuckDB
            df = pl.DataFrame({
                "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
                "code": ["A", "B", "C", "D"] * 2,
                "open": [100.0, 200.0, 50.0, 300.0, 101.0, 201.0, 51.0, 301.0],
                "high": [105.0, 205.0, 55.0, 305.0, 106.0, 206.0, 56.0, 306.0],
                "low": [95.0, 195.0, 45.0, 295.0, 96.0, 196.0, 46.0, 296.0],
                "close": [102.0, 202.0, 52.0, 302.0, 103.0, 203.0, 53.0, 303.0],
                "volume": [1000, 2000, 500, 3000, 1100, 2100, 600, 3100],
            })
            node = DuckDBNode(database=db_path)
            node.connect()
            node.insert_df(df.to_pandas(), "stock_data", if_exists="replace")
            node.disconnect()

            yaml_str = f"""
name: "duckdb_e2e"
data:
  source: duckdb
  path: "{db_path}"
  table: stock_data
  columns: [date, code, open, high, low, close, volume]
  date_column: date
  code_column: code
factors:
  - name: ret
    expr: "close / open - 1"
backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  initial_cash: 1000000
  signals:
    buy_threshold: 0.01
    sell_threshold: -0.01
"""
            tool = ConfigBacktestTool()
            result = asyncio.run(tool.execute(config_yaml=yaml_str))

            assert result["status"] == "success"
            assert result["summary"]["final_cash"] > 0

    def test_duckdb_query_filter(self):
        """DuckDB + query_filter 过滤"""
        from QuantNodes.database_node import DuckDBNode

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "filtered.duckdb")

            df = pl.DataFrame({
                "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
                "code": ["A", "B", "C", "D"] * 2,
                "open": [100.0, 200.0, 50.0, 300.0, 101.0, 201.0, 51.0, 301.0],
                "high": [105.0, 205.0, 55.0, 305.0, 106.0, 206.0, 56.0, 306.0],
                "low": [95.0, 195.0, 45.0, 295.0, 96.0, 196.0, 46.0, 296.0],
                "close": [102.0, 202.0, 52.0, 302.0, 103.0, 203.0, 53.0, 303.0],
                "volume": [1000, 2000, 500, 3000, 1100, 2100, 600, 3100],
            })
            node = DuckDBNode(database=db_path)
            node.connect()
            node.insert_df(df.to_pandas(), "stock_data", if_exists="replace")
            node.disconnect()

            yaml_str = f"""
name: "duckdb_filter"
data:
  source: duckdb
  path: "{db_path}"
  table: stock_data
  columns: [date, code, open, high, low, close, volume]
  date_column: date
  code_column: code
  query_filter: "WHERE code IN ('A', 'B')"
factors:
  - name: ret
    expr: "close / open - 1"
backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-02"
  initial_cash: 1000000
  signals:
    buy_threshold: 0.01
    sell_threshold: -0.01
"""
            tool = ConfigBacktestTool()
            result = asyncio.run(tool.execute(config_yaml=yaml_str))

            assert result["status"] == "success"
            assert result["summary"]["final_cash"] > 0


class TestE2E_LoaderRoundTrip:
    """ConfigLoader 往返测试: to_yaml → load → 验证字段"""

    def test_roundtrip_preserves_data_fields(self):
        """to_yaml → load 后 DataConfig 字段完整保留"""
        config = StrategyConfig(
            name="roundtrip_test",
            data=DataConfig(
                source="clickhouse",
                table="quote.cn_stock",
                conn_ini="conn.ini",
                conn_section="ClickHouse",
                columns=["ts_code", "trade_date", "close"],
                date_column="trade_date",
                code_column="ts_code",
                column_mapping={"ts_code": "code", "trade_date": "date"},
                query_filter="WHERE trade_date >= '2024-01-01'",
            ),
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-12-31",
                initial_cash=1000000,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "roundtrip.yaml")
            loader = ConfigLoader()
            loader.to_yaml(config, yaml_path)
            loaded = loader.load(yaml_path)

            assert loaded.data.source == "clickhouse"
            assert loaded.data.table == "quote.cn_stock"
            assert loaded.data.conn_ini == "conn.ini"
            assert loaded.data.conn_section == "ClickHouse"
            assert loaded.data.columns == ["ts_code", "trade_date", "close"]
            assert loaded.data.date_column == "trade_date"
            assert loaded.data.code_column == "ts_code"
            assert loaded.data.column_mapping == {"ts_code": "code", "trade_date": "date"}
            assert loaded.data.query_filter == "WHERE trade_date >= '2024-01-01'"


class TestE2E_ErrorPaths:
    """错误路径测试"""

    def test_no_data_source(self):
        """缺少数据源 → ValueError"""
        async def _test():
            tool = ConfigBacktestTool()
            yaml_str = """
name: "no_data"
factors:
  - name: ret
    expr: "close / open - 1"
"""
            result = await tool.execute(config_yaml=yaml_str)
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_unsupported_file_format(self):
        """不支持的文件格式 → ValueError"""
        async def _test():
            tool = ConfigBacktestTool()
            yaml_str = """
name: "bad_format"
data:
  source: csv
  path: "data.xlsx"
factors:
  - name: ret
    expr: "close / open - 1"
"""
            result = await tool.execute(config_yaml=yaml_str)
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_missing_conn_ini(self):
        """clickhouse 源缺少 conn.ini → FileNotFoundError"""
        async def _test():
            tool = ConfigBacktestTool()
            yaml_str = """
name: "missing_ini"
data:
  source: clickhouse
  conn_ini: "/nonexistent/conn.ini"
  conn_section: ClickHouse
  table: test
factors:
  - name: ret
    expr: "close / open - 1"
"""
            result = await tool.execute(config_yaml=yaml_str)
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_invalid_yaml(self):
        """无效 YAML → error"""
        async def _test():
            tool = ConfigBacktestTool()
            result = await tool.execute(config_yaml="{{invalid yaml}}")
            assert result["status"] == "error"

        asyncio.run(_test())


class TestE2E_ConfigLoaderParse:
    """ConfigLoader._parse() 新字段解析测试"""

    def test_parse_clickhouse_source(self):
        """解析 clickhouse 源配置"""
        loader = ConfigLoader()
        config = loader._parse({
            "name": "ch_test",
            "data": {
                "source": "clickhouse",
                "table": "quote.cn_stock",
                "conn_ini": "/etc/conn.ini",
                "conn_section": "ClickHouse",
                "date_column": "trade_date",
                "code_column": "ts_code",
                "column_mapping": {"ts_code": "code"},
                "query_filter": "WHERE code > 0",
            },
        })
        assert config.data.source == "clickhouse"
        assert config.data.table == "quote.cn_stock"
        assert config.data.conn_ini == "/etc/conn.ini"
        assert config.data.column_mapping == {"ts_code": "code"}
        assert config.data.query_filter == "WHERE code > 0"

    def test_parse_csv_defaults(self):
        """CSV 源默认值"""
        loader = ConfigLoader()
        config = loader._parse({
            "data": {"source": "csv", "path": "data.csv"},
        })
        assert config.data.table == ""
        assert config.data.conn_ini == "conn.ini"
        assert config.data.column_mapping == {}
        assert config.data.query_filter == ""


class TestE2E_BuildQuery:
    """_build_query() SQL 构建测试"""

    def test_basic_query(self):
        """基础 SQL 构建"""
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
        from QuantNodes.agent.config.types import DataConfig

        tool = ConfigBacktestTool()
        data_cfg = DataConfig(
            columns=["date", "code", "close"],
            table="stock_daily",
            date_column="date",
            code_column="code",
        )
        sql = tool._build_query(data_cfg)
        assert "SELECT date, code, close FROM stock_daily" in sql
        assert "ORDER BY code, date" in sql

    def test_query_with_filter(self):
        """带 WHERE 条件的 SQL"""
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
        from QuantNodes.agent.config.types import DataConfig

        tool = ConfigBacktestTool()
        data_cfg = DataConfig(
            table="stock_daily",
            date_column="date",
            code_column="code",
            query_filter="WHERE trade_date >= '2024-01-01'",
        )
        sql = tool._build_query(data_cfg)
        assert "WHERE" in sql
        assert "trade_date >= '2024-01-01'" in sql

    def test_query_no_table_raises(self):
        """缺少 table → ValueError"""
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
        from QuantNodes.agent.config.types import DataConfig

        tool = ConfigBacktestTool()
        data_cfg = DataConfig(columns=["date", "code"])
        with pytest.raises(ValueError, match="table"):
            tool._build_query(data_cfg)


# 导入被测试的工具
from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool

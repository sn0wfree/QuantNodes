# coding=utf-8
"""
DataConfig 扩展字段 + 两层列名映射 单元测试

覆盖:
- DataConfig 新增字段 (db_date_column, db_code_column, standard_columns)
- 两层列名设计的解析和序列化
- ConfigLoader._parse() 和 to_yaml() 对新字段的支持
"""

import yaml
import tempfile
import os

from QuantNodes.agent.config.types import DataConfig, StrategyConfig, FactorConfig
from QuantNodes.agent.config.loader import ConfigLoader


class TestDataConfigExtendedFields:
    """DataConfig 扩展字段测试"""

    def test_default_values(self):
        dc = DataConfig()
        assert dc.db_date_column == ""
        assert dc.db_code_column == ""
        assert dc.standard_columns == {
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        }

    def test_custom_values(self):
        dc = DataConfig(
            db_date_column="trade_date",
            db_code_column="ts_code",
            standard_columns={"open": "Open", "close": "Close"},
        )
        assert dc.db_date_column == "trade_date"
        assert dc.db_code_column == "ts_code"
        assert dc.standard_columns["open"] == "Open"

    def test_column_mapping(self):
        dc = DataConfig(
            column_mapping={"ts_code": "code", "trade_date": "date", "vol": "volume"},
        )
        assert dc.column_mapping["ts_code"] == "code"
        assert len(dc.column_mapping) == 3


class TestTwoLayerColumnName:
    """两层列名设计测试"""

    def test_clickhouse_two_layer_config(self):
        data = {
            "data": {
                "source": "clickhouse",
                "table": "quote.cn_stock",
                "columns": ["ts_code", "trade_date", "close", "vol"],
                "db_date_column": "trade_date",
                "db_code_column": "ts_code",
                "date_column": "date",
                "code_column": "code",
                "column_mapping": {
                    "ts_code": "code",
                    "trade_date": "date",
                    "vol": "volume",
                },
                "query_filter": "WHERE trade_date >= '2023-01-01'",
            },
            "factors": [{"name": "ma5", "expr": "ts_mean(close, 5)"}],
            "operations": [],
            "composite": [],
        }
        loader = ConfigLoader()
        config = loader._parse(data)

        assert config.data.source == "clickhouse"
        assert config.data.db_date_column == "trade_date"
        assert config.data.db_code_column == "ts_code"
        assert config.data.date_column == "date"
        assert config.data.code_column == "code"
        assert config.data.column_mapping["ts_code"] == "code"

    def test_csv_no_db_columns(self):
        data = {
            "data": {
                "source": "csv",
                "path": "data/test.csv",
                "date_column": "date",
                "code_column": "code",
            },
            "factors": [{"name": "ma5", "expr": "ts_mean(close, 5)"}],
            "operations": [],
            "composite": [],
        }
        loader = ConfigLoader()
        config = loader._parse(data)

        assert config.data.source == "csv"
        assert config.data.db_date_column == ""
        assert config.data.db_code_column == ""
        assert config.data.column_mapping == {}

    def test_to_yaml_roundtrip_with_db_columns(self):
        """to_yaml 保留 db_* 字段"""
        config = StrategyConfig(
            name="test_roundtrip",
            data=DataConfig(
                source="clickhouse",
                table="quote.cn_stock",
                db_date_column="trade_date",
                db_code_column="ts_code",
                date_column="date",
                code_column="code",
                column_mapping={"ts_code": "code", "trade_date": "date"},
                query_filter="WHERE trade_date >= '2023-01-01'",
                conn_ini="conn.ini",
                conn_section="ClickHouse",
            ),
            factors=[FactorConfig(name="ma5", expr="ts_mean(close, 5)")],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            loader = ConfigLoader()
            loader.to_yaml(config, tmp_path)

            with open(tmp_path, encoding="utf-8") as f:
                exported = yaml.safe_load(f)

            assert exported["data"]["db_date_column"] == "trade_date"
            assert exported["data"]["db_code_column"] == "ts_code"
            assert exported["data"]["column_mapping"]["ts_code"] == "code"
            assert exported["data"]["query_filter"] == "WHERE trade_date >= '2023-01-01'"

            # 再加载回来
            config2 = loader.load(tmp_path)
            assert config2.data.db_date_column == "trade_date"
            assert config2.data.db_code_column == "ts_code"
            assert config2.data.column_mapping == {"ts_code": "code", "trade_date": "date"}
        finally:
            os.unlink(tmp_path)

    def test_to_yaml_omits_empty_db_columns(self):
        """db_* 为空时不序列化"""
        config = StrategyConfig(
            name="test_omit",
            data=DataConfig(source="csv", path="data.csv"),
            factors=[FactorConfig(name="ma5", expr="ts_mean(close, 5)")],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            loader = ConfigLoader()
            loader.to_yaml(config, tmp_path)

            with open(tmp_path, encoding="utf-8") as f:
                exported = yaml.safe_load(f)

            assert "db_date_column" not in exported["data"]
            assert "db_code_column" not in exported["data"]
        finally:
            os.unlink(tmp_path)

    def test_to_yaml_standard_columns(self):
        """standard_columns 序列化"""
        config = StrategyConfig(
            name="test_std_cols",
            data=DataConfig(
                source="clickhouse",
                standard_columns={"open": "Open", "close": "Close"},
            ),
            factors=[FactorConfig(name="ma5", expr="ts_mean(close, 5)")],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            loader = ConfigLoader()
            loader.to_yaml(config, tmp_path)

            with open(tmp_path, encoding="utf-8") as f:
                exported = yaml.safe_load(f)

            assert exported["data"]["standard_columns"]["open"] == "Open"
        finally:
            os.unlink(tmp_path)

    def test_to_yaml_universe(self):
        """universe 非默认值时序列化"""
        config = StrategyConfig(
            name="test_universe",
            data=DataConfig(source="csv"),
            backtest=BacktestConfig(
                start_date="2023-01-01",
                end_date="2024-01-01",
                universe="000001.SZ,600000.SH",
            ),
            factors=[FactorConfig(name="ma5", expr="ts_mean(close, 5)")],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            loader = ConfigLoader()
            loader.to_yaml(config, tmp_path)

            with open(tmp_path, encoding="utf-8") as f:
                exported = yaml.safe_load(f)

            assert exported["backtest"]["universe"] == "000001.SZ,600000.SH"
        finally:
            os.unlink(tmp_path)


# Import needed for test_to_yaml_universe
from QuantNodes.agent.config.types import BacktestConfig

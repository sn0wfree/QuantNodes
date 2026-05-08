# -*- coding: utf-8 -*-
"""QuantNodes 全局测试 fixtures"""
import pytest
import pandas as pd
import polars as pl


@pytest.fixture
def sample_df():
    """测试用 DataFrame"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'score': [85.5, 90.0, 78.5, 92.0, 88.0]
    })


@pytest.fixture
def sample_df_with_null():
    """带空值的测试 DataFrame"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', None, 'Charlie', 'David', 'Eve'],
        'age': [25, None, 35, 40, 45],
        'score': [85.5, 90.0, None, 92.0, 88.0]
    })


@pytest.fixture
def temp_csv_file(tmp_path, sample_df):
    """CSV 临时文件"""
    filepath = tmp_path / "test_data.csv"
    sample_df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def temp_parquet_file(tmp_path, sample_df):
    """Parquet 临时文件"""
    filepath = tmp_path / "test_data.parquet"
    sample_df.to_parquet(filepath, index=False)
    return filepath


@pytest.fixture
def temp_sqlite_db(tmp_path):
    """SQLite 临时数据库文件"""
    return tmp_path / "test_sqlite.db"


@pytest.fixture
def temp_duckdb_db(tmp_path):
    """DuckDB 临时数据库文件"""
    return tmp_path / "test_duckdb.duckdb"


@pytest.fixture
def temp_csv_file_with_null(tmp_path, sample_df_with_null):
    """带空值的 CSV 临时文件"""
    filepath = tmp_path / "test_data_null.csv"
    sample_df_with_null.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def polars_df():
    """基础 Polars DataFrame fixture"""
    return pl.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def market_data_df():
    """行情数据 DataFrame，用于因子计算测试

    包含 date, code, close, volume, forward_return 列
    """
    return pl.DataFrame({
        'date': ['2024-01-01', '2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03', '2024-01-03'],
        'code': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'close': [100.0, 200.0, 300.0, 101.0, 202.0, 303.0, 99.0, 198.0, 297.0],
        'volume': [1000, 2000, 3000, 1100, 2100, 3100, 900, 1900, 2900],
        'forward_return': [0.01, 0.02, 0.015, -0.01, 0.01, 0.005, 0.02, -0.01, 0.01],
    })


@pytest.fixture
def market_data_pdf():
    """行情数据 pandas DataFrame"""
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03', '2024-01-03'],
        'code': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
        'close': [100.0, 200.0, 300.0, 101.0, 202.0, 303.0, 99.0, 198.0, 297.0],
        'volume': [1000, 2000, 3000, 1100, 2100, 3100, 900, 1900, 2900],
        'forward_return': [0.01, 0.02, 0.015, -0.01, 0.01, 0.005, 0.02, -0.01, 0.01],
    })


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端，用于不需要真实 API 调用的测试

    返回预设的响应内容，不发起真实 HTTP 请求。
    """
    from unittest.mock import MagicMock
    from QuantNodes.ai.llm.base import ChatCompletion, MessageRole

    client = MagicMock()
    client.model = "gpt-4-mock"

    def mock_chat(messages, **kwargs):
        content = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                content_str = msg.content
            else:
                content_str = str(msg)
            if "提取" in content_str or "extract" in content_str.lower():
                content = '{"logics": [{"logic_type": "factor", "title": "测试因子", "description": "测试描述", "formula": "close / volume", "evidence": "原文依据", "confidence": 0.9}]}'
            elif "挖掘" in content_str or "mining" in content_str.lower():
                content = '{"factor_name": "test_factor", "formula": "ts_mean(close, 20)"}'
            else:
                content = '{"result": "ok"}'
        return ChatCompletion(
            content=content,
            role=MessageRole.ASSISTANT,
            finish_reason="stop",
        )

    client.chat = mock_chat
    return client


@pytest.fixture
def temp_yaml_config(tmp_path):
    """临时 YAML 策略配置文件"""
    config_content = """
name: test_strategy
description: 测试策略

data:
  source: csv
  path: ./data.csv

operators:
  - type: select
    columns: [date, code, close, volume]

factors:
  - name: ma5
    formula: ts_mean(close, 5)
  - name: ma10
    formula: ts_mean(close, 10)

backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
  commission: 0.001
  slippage: 0.001
"""
    filepath = tmp_path / "test_strategy.yaml"
    filepath.write_text(config_content, encoding="utf-8")
    return filepath


@pytest.fixture
def monitor_db(tmp_path):
    """Monitor 临时数据库，带 schema 初始化

    返回 (db_path, DatabaseManager) 元组
    """
    from QuantNodes.monitor.storage.repository import DatabaseManager

    db_path = str(tmp_path / "monitor_test.db")
    db = DatabaseManager(db_path)
    db.connect()
    yield db_path, db
    db.close()

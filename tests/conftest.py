# -*- coding: utf-8 -*-
"""database_node 测试 fixtures"""
import os
import pytest
import pandas as pd


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

# -*- coding: utf-8 -*-
"""Integration tests fixtures"""
import os
import pytest
import polars as pl


@pytest.fixture
def wiki_path(tmp_path):
    """Wiki 临时目录路径"""
    wiki_dir = tmp_path / "test_wiki"
    wiki_dir.mkdir(exist_ok=True)
    return str(wiki_dir)


@pytest.fixture
def monitor_db_path(tmp_path):
    """Monitor 临时数据库路径"""
    return str(tmp_path / "monitor_test.db")


@pytest.fixture
def monitor_db(monitor_db_path):
    """Monitor 临时数据库（已连接并初始化 schema）

    返回 DatabaseManager 实例
    """
    from QuantNodes.monitor.storage.repository import DatabaseManager

    db = DatabaseManager(monitor_db_path)
    db.connect()
    yield db
    db.close()


@pytest.fixture
def eval_data():
    """因子评估用的样本数据（Polars DataFrame）"""
    return pl.DataFrame({
        'date': ['2024-01-01', '2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03', '2024-01-03',
                 '2024-01-04', '2024-01-04', '2024-01-04'],
        'code': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C',
                 'A', 'B', 'C'],
        'close': [100.0, 200.0, 300.0, 101.0, 202.0, 303.0, 99.0, 198.0, 297.0,
                  102.0, 204.0, 306.0],
        'volume': [1000, 2000, 3000, 1100, 2100, 3100, 900, 1900, 2900,
                   1200, 2200, 3200],
        'forward_return': [0.01, 0.02, 0.015, -0.01, 0.01, 0.005, 0.02, -0.01, 0.01,
                          0.015, 0.02, 0.01],
    })


@pytest.fixture
def market_data_df():
    """行情数据 DataFrame"""
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
def sample_pdf_text():
    """样本研报文本"""
    return """
    量化研究报告：基于均线系统的选股策略

    本报告提出一种基于移动平均线的量化选股策略。

    因子构建：
    1. 短期均线：MA5 = TS_MEAN(close, 5)
    2. 长期均线：MA20 = TS_MEAN(close, 20)
    3. 均线差值：MA_diff = MA5 - MA20

    交易规则：
    - 当 MA5 上穿 MA20 时买入
    - 当 MA5 下穿 MA20 时卖出

    回测结果：
    - 年化收益率：15.3%
    - Sharpe比率：1.52
    - 最大回撤：-8.2%
    """

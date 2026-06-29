# coding=utf-8
"""
conftest.py for tests/quant_alpha/

Pytest 配置:
- @pytest.mark.unit (默认, < 1s)
- @pytest.mark.integration (< 5s, 端到端 mock)
- @pytest.mark.smoke (< 30s, 真实数据子集)
- @pytest.mark.slow (慢测试, 默认跳过)

运行:
  pytest tests/quant_alpha/                     # 默认跑 unit
  pytest tests/quant_alpha/ -m unit             # 只 unit
  pytest tests/quant_alpha/ -m integration      # 只 integration
  pytest tests/quant_alpha/ -m "not slow"       # 跳慢
"""
import pytest


def pytest_configure(config):
    """注册自定义 marker"""
    config.addinivalue_line(
        "markers", "unit: fast unit tests (default, < 1s)"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests with mock LLM (< 5s)"
    )
    config.addinivalue_line(
        "markers", "smoke: smoke tests with real data (< 30s)"
    )
    config.addinivalue_line(
        "markers", "slow: slow tests (skipped by default)"
    )


def pytest_collection_modifyitems(config, items):
    """默认按 marker 分类

    - 没有任何 marker 的测试: 视为 unit
    - 慢测试默认跳过 (用 -m slow 启用)
    """
    skip_slow = pytest.mark.skip(reason="slow test, use -m slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

# -*- coding: utf-8 -*-
"""Agent 子系统测试 fixtures"""
import pytest


@pytest.fixture
def mock_tool_registry():
    """Mock 工具注册表"""
    from unittest.mock import MagicMock
    from QuantNodes.agent.tools.registry import ToolRegistry

    registry = MagicMock(spec=ToolRegistry)
    registry.list_tools.return_value = []
    registry.get_tool.return_value = None
    return registry


@pytest.fixture
def mock_agent_context():
    """Mock Agent 上下文"""
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.get_messages.return_value = []
    ctx.add_message.return_value = None
    ctx.get_state.return_value = {}
    return ctx


@pytest.fixture
def temp_strategy_yaml(tmp_path):
    """临时策略 YAML 文件（用于配置回测测试）"""
    yaml_content = """
name: dual_ma_strategy
description: 双均线策略

data:
  source: csv
  path: ./test/data.csv

operators:
  - type: select
    columns: [date, code, close, volume]

factors:
  - name: ma5
    formula: ts_mean(close, 5)
  - name: ma20
    formula: ts_mean(close, 20)

backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
  commission: 0.001
  slippage: 0.001
"""
    filepath = tmp_path / "dual_ma.yaml"
    filepath.write_text(yaml_content, encoding="utf-8")
    return filepath


@pytest.fixture
def sample_backtest_result():
    """样本回测结果"""
    return {
        "status": "success",
        "sharpe_ratio": 1.52,
        "sortino_ratio": 2.0,
        "max_drawdown": -0.082,
        "annualized_return": 0.153,
        "win_rate": 0.55,
        "total_trades": 100,
        "equity_curve": [1000000, 1010000, 1020000],
    }

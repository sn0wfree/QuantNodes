# -*- coding: utf-8 -*-
"""QuantNodes.monitor.agent_tools.monitor_tool 单元测试"""
import pytest
from unittest.mock import MagicMock, patch

from QuantNodes.monitor.agent_tools.monitor_tool import MonitorTool


class TestMonitorTool:
    def test_init(self):
        tool = MonitorTool(db_path="~/.quantnodes/test_monitor.db")
        assert tool._db_path == "~/.quantnodes/test_monitor.db"
        assert tool._dashboard is None

    def test_name_property(self):
        tool = MonitorTool()
        assert tool.name == "strategy_monitor"

    def test_description_property(self):
        tool = MonitorTool()
        assert "监控" in tool.description
        assert "绩效" in tool.description

    def test_parameters_property(self):
        tool = MonitorTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert params["properties"]["action"]["type"] == "string"
        assert "status" in params["properties"]["action"]["enum"]

    def test_read_only_true(self):
        tool = MonitorTool()
        assert tool.read_only is True

    def test_to_openai_schema(self):
        tool = MonitorTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "strategy_monitor"

    @pytest.mark.asyncio
    async def test_execute_status_without_strategy_name(self):
        tool = MonitorTool()
        result = await tool.execute(action="status")
        assert "error" in result
        assert "strategy_name required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_history_without_strategy_name(self):
        tool = MonitorTool()
        result = await tool.execute(action="history")
        assert "error" in result
        assert "strategy_name required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_alerts_without_strategy_name(self):
        tool = MonitorTool()
        result = await tool.execute(action="alerts")
        assert "error" in result
        assert "strategy_name required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_compare_without_strategy_names(self):
        tool = MonitorTool()
        result = await tool.execute(action="compare")
        assert "error" in result
        assert "strategy_names required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        tool = MonitorTool()
        result = await tool.execute(action="unknown")
        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_status_with_mock_dashboard(self):
        tool = MonitorTool()

        mock_dashboard = MagicMock()
        mock_dashboard.get_strategy_summary.return_value = {
            "strategy_name": "test_strategy",
            "latest_run": {"status": "success"},
            "performance": {"sharpe_ratio": 1.5},
            "pending_alerts": 0,
        }

        with patch.object(tool, "_get_dashboard", return_value=mock_dashboard):
            result = await tool.execute(action="status", strategy_name="test_strategy")

        assert "strategy_name" in result
        assert result["strategy_name"] == "test_strategy"

    @pytest.mark.asyncio
    async def test_execute_history_with_mock_dashboard(self):
        tool = MonitorTool()

        mock_dashboard = MagicMock()
        mock_dashboard.get_performance_history.return_value = [
            {"date": "2024-01-01", "sharpe_ratio": 1.5}
        ]

        with patch.object(tool, "_get_dashboard", return_value=mock_dashboard):
            result = await tool.execute(action="history", strategy_name="test_strategy", days=30)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_execute_alerts_with_mock_dashboard(self):
        tool = MonitorTool()

        mock_dashboard = MagicMock()
        mock_dashboard.get_alert_history.return_value = [
            {"type": "ks_test", "severity": "warning"}
        ]

        with patch.object(tool, "_get_dashboard", return_value=mock_dashboard):
            result = await tool.execute(action="alerts", strategy_name="test_strategy", days=30)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_execute_compare_with_mock_dashboard(self):
        tool = MonitorTool()

        mock_dashboard = MagicMock()
        mock_dashboard.get_comparison.return_value = {
            "strategy_a": {"sharpe_ratio": 1.5},
            "strategy_b": {"sharpe_ratio": 1.2},
        }

        with patch.object(tool, "_get_dashboard", return_value=mock_dashboard):
            result = await tool.execute(action="compare", strategy_names=["strategy_a", "strategy_b"])

        assert "strategy_a" in result
        assert "strategy_b" in result

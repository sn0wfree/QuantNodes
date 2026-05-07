# -*- coding: utf-8 -*-
"""QuantNodes.monitor.agent_tools.schedule_tool 单元测试"""
import pytest
from unittest.mock import MagicMock, patch

from QuantNodes.monitor.agent_tools.schedule_tool import ScheduleTool


class TestScheduleTool:
    def test_init(self):
        tool = ScheduleTool(db_path="~/.quantnodes/test_schedule.db")
        assert tool._db_path == "~/.quantnodes/test_schedule.db"
        assert tool._scheduler is None

    def test_name_property(self):
        tool = ScheduleTool()
        assert tool.name == "strategy_schedule"

    def test_description_property(self):
        tool = ScheduleTool()
        assert "调度" in tool.description
        assert "定时" in tool.description

    def test_parameters_property(self):
        tool = ScheduleTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert params["properties"]["action"]["type"] == "string"
        assert "add" in params["properties"]["action"]["enum"]

    def test_read_only_false(self):
        tool = ScheduleTool()
        assert tool.read_only is False

    def test_to_openai_schema(self):
        tool = ScheduleTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "strategy_schedule"

    @pytest.mark.asyncio
    async def test_execute_remove_not_found(self):
        tool = ScheduleTool()

        mock_scheduler = MagicMock()
        mock_scheduler.remove_job.return_value = False

        with patch.object(tool, "_get_scheduler", return_value=mock_scheduler):
            result = await tool.execute(action="remove", strategy_name="nonexistent")

        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_execute_list(self):
        tool = ScheduleTool()

        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = [
            {"name": "job1", "strategy": "strategy_a"},
            {"name": "job2", "strategy": "strategy_b"},
        ]

        with patch.object(tool, "_get_scheduler", return_value=mock_scheduler):
            result = await tool.execute(action="list", strategy_name="any_strategy")

        assert "jobs" in result
        assert len(result["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_execute_pause(self):
        tool = ScheduleTool()

        mock_scheduler = MagicMock()
        mock_scheduler.pause_job.return_value = True

        with patch.object(tool, "_get_scheduler", return_value=mock_scheduler):
            result = await tool.execute(action="pause", strategy_name="test_strategy")

        assert result["status"] == "paused"
        mock_scheduler.pause_job.assert_called_once_with("test_strategy")

    @pytest.mark.asyncio
    async def test_execute_resume(self):
        tool = ScheduleTool()

        mock_scheduler = MagicMock()
        mock_scheduler.resume_job.return_value = True

        with patch.object(tool, "_get_scheduler", return_value=mock_scheduler):
            result = await tool.execute(action="resume", strategy_name="test_strategy")

        assert result["status"] == "resumed"
        mock_scheduler.resume_job.assert_called_once_with("test_strategy")

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        tool = ScheduleTool()
        result = await tool.execute(action="unknown", strategy_name="test_strategy")
        assert "error" in result
        assert "Unknown action" in result["error"]

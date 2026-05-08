# coding=utf-8
"""
测试CLI接口
"""

import asyncio
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestGetDefaultProvider:
    def test_get_default_provider_success(self):
        from QuantNodes.agent.cli.main import _get_default_provider

        mock_client = Mock()
        mock_provider = Mock()

        with patch("QuantNodes.ai.llm.openai.OpenAIClient", return_value=mock_client):
            with patch("QuantNodes.agent.providers.quantnodes.QuantNodesLLMProvider", return_value=mock_provider):
                result = _get_default_provider()
                assert result == mock_provider

    def test_get_default_provider_failure(self):
        from QuantNodes.agent.cli.main import _get_default_provider

        with patch("QuantNodes.ai.llm.openai.OpenAIClient", side_effect=Exception("Init failed")):
            with patch("sys.stderr"):
                result = _get_default_provider()
                assert result is None


class TestRunSingle:
    def test_run_single(self):
        from QuantNodes.agent.cli.main import run_single

        mock_loop = Mock()
        mock_loop.chat = AsyncMock(return_value="Test response")

        with patch("builtins.print") as mock_print:
            asyncio.run(run_single(mock_loop, "Hello"))

        mock_loop.chat.assert_called_once_with("Hello", session_id="cli_single")
        mock_print.assert_called_once_with("Test response")


class TestMain:
    def test_main_no_provider(self):
        from QuantNodes.agent.cli.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("QuantNodes.agent.cli.main._get_default_provider", return_value=None):
                with patch("sys.argv", ["agent"]):
                    with patch("sys.stderr"):
                        with pytest.raises(SystemExit) as excinfo:
                            main()
                        assert excinfo.value.code == 1

    def test_main_single_message(self):
        from QuantNodes.agent.cli.main import main

        mock_provider = Mock()
        mock_loop_instance = Mock()
        mock_loop_instance.chat = AsyncMock(return_value="Response")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("QuantNodes.agent.cli.main._get_default_provider", return_value=mock_provider):
                with patch("QuantNodes.agent.cli.main.AgentLoop", return_value=mock_loop_instance):
                    with patch("sys.argv", ["agent", "Hello", "World"]):
                        with patch("builtins.print"):
                            main()

        mock_loop_instance.chat.assert_called_once_with("Hello World", session_id="cli_single")

    def test_main_no_echo(self):
        from QuantNodes.agent.cli.main import main

        mock_provider = Mock()
        mock_loop_instance = Mock()
        mock_loop_instance.chat = AsyncMock(return_value="Response")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("QuantNodes.agent.cli.main._get_default_provider", return_value=mock_provider):
                with patch("QuantNodes.agent.cli.main.AgentLoop", return_value=mock_loop_instance):
                    with patch("sys.argv", ["agent", "--no-echo", "test"]):
                        with patch("builtins.print"):
                            main()

        mock_loop_instance.register_tool.assert_not_called()

    def test_main_with_echo(self):
        from QuantNodes.agent.cli.main import main

        mock_provider = Mock()
        mock_loop_instance = Mock()
        mock_loop_instance.chat = AsyncMock(return_value="Response")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("QuantNodes.agent.cli.main._get_default_provider", return_value=mock_provider):
                with patch("QuantNodes.agent.cli.main.AgentLoop", return_value=mock_loop_instance):
                    with patch("sys.argv", ["agent", "test"]):
                        with patch("builtins.print"):
                            main()

        mock_loop_instance.register_tool.assert_called_once()

    def test_main_with_custom_workspace(self):
        from QuantNodes.agent.cli.main import main

        mock_provider = Mock()
        mock_loop_instance = Mock()
        mock_loop_instance.chat = AsyncMock(return_value="Response")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("QuantNodes.agent.cli.main._get_default_provider", return_value=mock_provider):
                with patch("QuantNodes.agent.cli.main.AgentLoop", return_value=mock_loop_instance):
                    with patch("sys.argv", ["agent", "--workspace", tmpdir, "test"]):
                        with patch("builtins.print"):
                            main()

    def test_main_with_custom_model(self):
        from QuantNodes.agent.cli.main import main

        mock_provider = Mock()
        mock_loop_instance = Mock()
        mock_loop_instance.chat = AsyncMock(return_value="Response")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("QuantNodes.agent.cli.main._get_default_provider", return_value=mock_provider):
                with patch("QuantNodes.agent.cli.main.AgentLoop", return_value=mock_loop_instance):
                    with patch("sys.argv", ["agent", "--model", "gpt-4", "test"]):
                        with patch("builtins.print"):
                            main()


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

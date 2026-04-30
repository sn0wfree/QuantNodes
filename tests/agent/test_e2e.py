# coding=utf-8
"""
端到端测试

测试策略生成 → 验证 → 回测 完整闭环。
"""

import asyncio

from QuantNodes.agent.tools import (
    SandboxTool,
    PipelineTool,
    StrategyTool,
    BacktestTool,
    FactorTool,
)
from QuantNodes.agent.bus.events import InboundMessage, OutboundMessage
from QuantNodes.agent.session.manager import SessionManager
from QuantNodes.agent.core.loop import AgentLoop


class TestSandboxTool:
    """沙箱工具测试"""

    def test_validate_safe_code(self):
        async def _test():
            tool = SandboxTool()
            result = await tool.execute(code="import pandas\ndf = pd.DataFrame()")
            assert result["is_safe"] is True
            assert len(result["errors"]) == 0

        asyncio.run(_test())

    def test_validate_dangerous_code(self):
        async def _test():
            tool = SandboxTool()
            result = await tool.execute(code="import os\nos.system('ls')")
            assert result["is_safe"] is False
            assert len(result["errors"]) > 0

        asyncio.run(_test())

    def test_validate_empty_code(self):
        async def _test():
            tool = SandboxTool()
            result = await tool.execute(code="")
            assert result["is_safe"] is False

        asyncio.run(_test())

    def test_validate_long_code_exceeds_max_length(self):
        async def _test():
            tool = SandboxTool(max_code_length=100)
            long_code = "x = 1\n" * 50
            result = await tool.execute(code=long_code)
            assert result["is_safe"] is False
            assert any("max length" in e.lower() for e in result["errors"])

        asyncio.run(_test())

    def test_validate_with_warnings_only(self):
        async def _test():
            tool = SandboxTool(allow_warnings=False)
            result = await tool.execute(code="import os\nx = 1")
            assert result["is_safe"] is False
            assert len(result["warnings"]) > 0

        asyncio.run(_test())

    def test_validate_safe_nested_imports(self):
        async def _test():
            tool = SandboxTool()
            code = """import pandas as pd
import numpy as np
from datetime import datetime

df = pd.DataFrame()
"""
            result = await tool.execute(code=code)
            assert result["is_safe"] is True

        asyncio.run(_test())

    def test_warnings_returned_only(self):
        async def _test():
            tool = SandboxTool()
            result = await tool.execute(code="import os\nprint('test')")
            assert "warnings_only" in result

        asyncio.run(_test())

    def test_max_code_length_boundary(self):
        async def _test():
            tool = SandboxTool(max_code_length=10)
            result = await tool.execute(code="x = 12345678901")
            assert result["is_safe"] is False

        asyncio.run(_test())


class TestPipelineTool:
    """Pipeline工具测试"""

    def test_validate_valid_pipeline(self):
        async def _test():
            tool = PipelineTool()
            code = """import pandas as pd
from QuantNodes.factor_node import PointFactorNode

factor = PointFactorNode(expression="close / open - 1")
"""
            result = await tool.execute(code=code)
            assert result["is_valid"] is True
            assert "PointFactorNode" in result["nodes"]

        asyncio.run(_test())

    def test_validate_invalid_syntax(self):
        async def _test():
            tool = PipelineTool()
            code = """import pandas as pd
df = pd.DataFrame(
"""
            result = await tool.execute(code=code)
            assert result["is_valid"] is False
            assert len(result["errors"]) > 0

        asyncio.run(_test())

    def test_extract_code_from_markdown(self):
        async def _test():
            tool = PipelineTool()
            code = """```python
import pandas as pd
df = pd.DataFrame()
```"""
            result = await tool.execute(code=code)
            assert result["is_valid"] is True

        asyncio.run(_test())

    def test_validate_empty_code(self):
        async def _test():
            tool = PipelineTool()
            code = ""
            result = await tool.execute(code=code)
            assert result["is_valid"] is False

        asyncio.run(_test())

    def test_multiple_nodes_extraction(self):
        async def _test():
            tool = PipelineTool()
            code = """from QuantNodes.factor_node import PointFactorNode, RollingFactorNode
import pandas as pd

f1 = PointFactorNode(expression="close")
f2 = RollingFactorNode(expression="close")
"""
            result = await tool.execute(code=code)
            assert result["is_valid"] is True
            assert "PointFactorNode" in result["nodes"]
            assert "RollingFactorNode" in result["nodes"]

        asyncio.run(_test())

    def test_no_nodes_found(self):
        async def _test():
            tool = PipelineTool()
            code = "x = 1 + 2"
            result = await tool.execute(code=code)
            assert result["is_valid"] is True
            assert len(result["nodes"]) == 0

        asyncio.run(_test())

    def test_markdown_without_python(self):
        async def _test():
            tool = PipelineTool()
            code = """```
import pandas as pd
df = pd.DataFrame()
```"""
            result = await tool.execute(code=code)
            assert result["is_valid"] is True

        asyncio.run(_test())


class TestStrategyTool:
    """策略生成工具测试"""

    def test_tool_properties(self):
        tool = StrategyTool()
        assert tool.name == "strategy"
        assert tool.description != ""
        assert "description" in tool.parameters["required"]

    def test_execute_returns_placeholder(self):
        async def _test():
            tool = StrategyTool()
            result = await tool.execute(description="生成一个动量策略")
            assert "message" in result
            assert result["is_valid"] is False

        asyncio.run(_test())

    def test_parameters_with_defaults(self):
        tool = StrategyTool()
        params = tool.parameters
        assert "description" in params["properties"]
        assert params["properties"]["description"]["type"] == "string"
        assert "validate" in params["properties"]
        assert params["properties"]["validate"]["default"] is True

    def test_validate_false_option(self):
        async def _test():
            tool = StrategyTool()
            result = await tool.execute(description="生成策略", validate=False)
            assert "message" in result

        asyncio.run(_test())

    def test_extra_kwargs_passed(self):
        async def _test():
            tool = StrategyTool()
            result = await tool.execute(
                description="生成策略",
                extra_param="ignored"
            )
            assert "message" in result

        asyncio.run(_test())


class TestBacktestTool:
    """回测工具测试"""

    def test_tool_properties(self):
        tool = BacktestTool()
        assert tool.name == "backtest"
        assert tool.description != ""
        assert tool.concurrency_safe is False

    def test_parameters_complete(self):
        tool = BacktestTool()
        params = tool.parameters
        assert "pipeline_code" in params["required"]
        assert "start_date" not in params["required"]
        assert "end_date" not in params["required"]

    def test_execute_returns_result(self):
        async def _test():
            tool = BacktestTool()
            result = await tool.execute(
                pipeline_code="x = 1",
                start_date="2020-01-01",
                end_date="2023-12-31"
            )
            assert result["status"] == "error"
            assert "errors" in result

        asyncio.run(_test())

    def test_with_custom_initial_cash(self):
        async def _test():
            tool = BacktestTool()
            result = await tool.execute(
                pipeline_code="x = 1",
                start_date="2020-01-01",
                end_date="2023-12-31",
                initial_cash=500000
            )
            assert result["config"]["initial_cash"] == 500000

        asyncio.run(_test())

    def test_all_required_params(self):
        async def _test():
            tool = BacktestTool()
            result = await tool.execute(
                pipeline_code="factor = PointFactorNode(expression='close')",
                start_date="2021-01-01",
                end_date="2022-12-31",
                initial_cash=100000
            )
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_read_only_false(self):
        tool = BacktestTool()
        assert tool.read_only is False

    def test_extra_params_ignored(self):
        async def _test():
            tool = BacktestTool()
            result = await tool.execute(
                pipeline_code="x = 1",
                start_date="2020-01-01",
                end_date="2023-12-31",
                unknown_param="ignored"
            )
            assert result["status"] == "error"

        asyncio.run(_test())


class TestFactorTool:
    """因子分析工具测试"""

    def test_tool_properties(self):
        tool = FactorTool()
        assert tool.name == "factor"
        assert tool.read_only is True

    def test_parameters_enum(self):
        tool = FactorTool()
        params = tool.parameters
        assert params["properties"]["analysis_type"]["enum"] == ["ic", "correlation", "both"]

    def test_execute_returns_result(self):
        async def _test():
            tool = FactorTool()
            result = await tool.execute(
                factor_code="x = 1",
                analysis_type="ic"
            )
            assert result["status"] == "error"
            assert "errors" in result

        asyncio.run(_test())

    def test_with_date_range(self):
        async def _test():
            tool = FactorTool()
            result = await tool.execute(
                factor_code="x = 1",
                analysis_type="both",
                start_date="2020-01-01",
                end_date="2023-12-31"
            )
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_analysis_type_ic(self):
        async def _test():
            tool = FactorTool()
            result = await tool.execute(
                factor_code="x = 1",
                analysis_type="ic"
            )
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_analysis_type_correlation(self):
        async def _test():
            tool = FactorTool()
            result = await tool.execute(
                factor_code="x = 1",
                analysis_type="correlation"
            )
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_analysis_type_both(self):
        async def _test():
            tool = FactorTool()
            result = await tool.execute(
                factor_code="x = 1",
                analysis_type="both"
            )
            assert result["status"] == "error"

        asyncio.run(_test())

    def test_missing_dates(self):
        async def _test():
            tool = FactorTool()
            result = await tool.execute(
                factor_code="x = 1",
                analysis_type="ic"
            )
            assert result["status"] == "error"

        asyncio.run(_test())


class TestE2EWorkflow:
    """端到端工作流测试"""

    def test_sandbox_pipeline_workflow(self):
        async def _test():
            sandbox = SandboxTool()
            pipeline = PipelineTool()

            code = "import pandas\ndf = pd.DataFrame()"

            sandbox_result = await sandbox.execute(code=code)
            assert sandbox_result["is_safe"] is True

            pipeline_result = await pipeline.execute(code=code)
            assert pipeline_result["is_valid"] is True

        asyncio.run(_test())

    def test_tool_registry_integration(self):
        from QuantNodes.agent.tools import ToolRegistry

        async def _test():
            registry = ToolRegistry()
            registry.register(SandboxTool())
            registry.register(PipelineTool())
            registry.register(StrategyTool())
            registry.register(BacktestTool())
            registry.register(FactorTool())

            assert len(registry.list_tools()) >= 5

            sandbox_result = await registry.execute_tool(
                "sandbox",
                code="import pandas"
            )
            assert sandbox_result.success is True

        asyncio.run(_test())

    def test_full_workflow_sandbox_pipeline(self):
        async def _test():
            sandbox_tool = SandboxTool()
            pipeline_tool = PipelineTool()

            code = """from QuantNodes.factor_node import PointFactorNode
factor = PointFactorNode(expression="close / open - 1")
"""
            sandbox_result = await sandbox_tool.execute(code=code)
            pipeline_result = await pipeline_tool.execute(code=code)

            assert sandbox_result["is_safe"] is True
            assert pipeline_result["is_valid"] is True
            assert "PointFactorNode" in pipeline_result["nodes"]

        asyncio.run(_test())

    def test_dangerous_code_blocked_in_pipeline(self):
        async def _test():
            sandbox_tool = SandboxTool()
            pipeline_tool = PipelineTool()

            dangerous_code = """import os
os.system('rm -rf /')
"""
            sandbox_result = await sandbox_tool.execute(code=dangerous_code)
            assert sandbox_result["is_safe"] is False
            assert len(sandbox_result["errors"]) > 0

        asyncio.run(_test())

    def test_tool_registry_sequential_execution(self):
        from QuantNodes.agent.tools import ToolRegistry

        async def _test():
            registry = ToolRegistry()
            registry.register(SandboxTool())
            registry.register(PipelineTool())

            code = "import pandas"

            r1 = await registry.execute_tool("sandbox", code=code)
            assert r1.success is True
            assert r1.content["is_safe"] is True

            r2 = await registry.execute_tool("pipeline", code=code)
            assert r2.success is True

        asyncio.run(_test())
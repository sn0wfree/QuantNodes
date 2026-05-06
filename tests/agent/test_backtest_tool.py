# coding=utf-8
"""QuantNodes.agent.tools.backtest 单元测试"""
import pytest

from QuantNodes.agent.tools.backtest import BacktestTool


@pytest.fixture
def tool():
    return BacktestTool()


class TestBacktestTool:
    def test_creation(self):
        tool = BacktestTool()
        assert tool is not None

    def test_name(self):
        tool = BacktestTool()
        assert tool.name == "backtest"

    def test_description(self):
        tool = BacktestTool()
        assert "回测" in tool.description

    def test_parameters(self):
        tool = BacktestTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "pipeline_code" in params["properties"]
        assert "start_date" in params["properties"]
        assert "end_date" in params["properties"]
        assert "initial_cash" in params["properties"]
        assert "commission" in params["properties"]
        assert params["required"] == ["pipeline_code"]

    def test_read_only(self):
        tool = BacktestTool()
        assert tool.read_only is False

    def test_concurrency_safe(self):
        tool = BacktestTool()
        assert tool.concurrency_safe is False

    def test_to_openai_schema(self):
        tool = BacktestTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "backtest"

    @pytest.mark.asyncio
    async def test_execute_empty_code(self, tool):
        result = await tool.execute(pipeline_code="")
        assert result["status"] == "error"
        assert any("No valid code found" in err for err in result["errors"])

    @pytest.mark.asyncio
    async def test_execute_no_strategy(self, tool):
        code = """
import pandas as pd
quote_data = pd.DataFrame({"a": [1, 2, 3]})
"""
        result = await tool.execute(pipeline_code=code)
        assert result["status"] == "error"
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_execute_no_quote_data(self, tool):
        code = """
from QuantNodes.backtest.strategy_node import MAStrategyNode
strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
"""
        result = await tool.execute(pipeline_code=code)
        assert result["status"] == "error"
        assert "quote_data" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_execute_with_code_block(self, tool):
        code = """
```python
import pandas as pd
from QuantNodes.backtest.strategy_node import MAStrategyNode

strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
quote_data = pd.DataFrame({"date": ["2024-01-01"], "close": [100]})
```
"""
        result = await tool.execute(pipeline_code=code)
        assert result["status"] == "error"
        assert "StrategyNode" in result["errors"][0] or result["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_config_defaults(self, tool):
        code = """
import pandas as pd
quote_data = pd.DataFrame({"a": [1]})
"""
        result = await tool.execute(pipeline_code=code)
        assert result["config"]["initial_cash"] == 100000
        assert result["config"]["commission"] == 0.001

    def test_extract_code_with_code_block(self, tool):
        code = """
```python
print("hello")
```
"""
        extracted = tool._extract_code(code)
        assert extracted == 'print("hello")'

    def test_extract_code_without_code_block(self, tool):
        code = 'print("hello")'
        extracted = tool._extract_code(code)
        assert extracted == code

    def test_extract_code_with_markdown(self, tool):
        code = """
Some text before
```python
x = 1
```
Some text after
"""
        extracted = tool._extract_code(code)
        assert extracted == "x = 1"

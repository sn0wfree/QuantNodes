# coding=utf-8
"""QuantNodes.agent.tools.pipeline 单元测试"""
import pytest

from QuantNodes.agent.tools.pipeline import PipelineTool


@pytest.fixture
def tool():
    return PipelineTool()


class TestPipelineTool:
    def test_creation(self):
        tool = PipelineTool()
        assert tool is not None

    def test_name(self):
        tool = PipelineTool()
        assert tool.name == "pipeline"

    def test_description(self):
        tool = PipelineTool()
        assert "Pipeline" in tool.description
        assert "验证" in tool.description

    def test_parameters(self):
        tool = PipelineTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "code" in params["properties"]
        assert "code" in params["required"]

    def test_read_only(self):
        tool = PipelineTool()
        assert tool.read_only is True

    def test_to_openai_schema(self):
        tool = PipelineTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "pipeline"

    @pytest.mark.asyncio
    async def test_execute_empty_code(self, tool):
        result = await tool.execute(code="")
        assert result["is_valid"] is False
        assert "No valid code found" in result["errors"]

    @pytest.mark.asyncio
    async def test_execute_simple_valid_code(self, tool):
        code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3]})
"""
        result = await tool.execute(code=code)
        assert result["is_valid"] is True
        assert result["security_status"] in ("safe", "skipped", "unknown")

    @pytest.mark.asyncio
    async def test_execute_with_code_block(self, tool):
        code = """
```python
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3]})
```
"""
        result = await tool.execute(code=code)
        assert result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_execute_with_node_patterns(self, tool):
        code = """
from QuantNodes.core.node import BaseNode
from QuantNodes.operator_node.transform import TransformNode

node = TransformNode()
"""
        result = await tool.execute(code=code)
        assert result["is_valid"] is True
        assert "TransformNode" in result["nodes"]

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self, tool):
        code = """
def broken_function(
    return None
"""
        result = await tool.execute(code=code)
        assert result["is_valid"] is False
        assert any("Syntax error" in err for err in result["errors"])

    def test_extract_code_with_code_block(self, tool):
        code = """```python
print("hello")
```"""
        extracted = tool._extract_code(code)
        assert extracted == 'print("hello")'

    def test_extract_code_without_code_block(self, tool):
        code = 'print("hello")'
        extracted = tool._extract_code(code)
        assert extracted == 'print("hello")'

    def test_extract_nodes(self, tool):
        code = """
from QuantNodes.core.node import BaseNode
from QuantNodes.operator_node.transform import TransformNode

node = TransformNode()
"""
        nodes = tool._extract_nodes(code)
        assert "TransformNode" in nodes

    def test_extract_nodes_no_duplicates(self, tool):
        code = """
TransformNode()
TransformNode()
"""
        nodes = tool._extract_nodes(code)
        assert len(nodes) == 1

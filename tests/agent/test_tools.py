# coding=utf-8
"""
测试工具系统
"""

import asyncio
from QuantNodes.agent.tools import ToolRegistry, EchoTool
from QuantNodes.agent.tools.base import Tool


class TestEchoTool:
    def test_name(self):
        tool = EchoTool()
        assert tool.name == "echo"

    def test_parameters(self):
        tool = EchoTool()
        assert "message" in tool.parameters["properties"]

    def test_execute(self):
        async def _test():
            tool = EchoTool()
            result = await tool.execute(message="hello")
            assert result == "hello"

        asyncio.run(_test())


class TestToolRegistry:
    def test_register(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        assert len(registry.list_tools()) == 1

    def test_get(self):
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        assert registry.get("echo") == tool

    def test_execute_tool(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())
            result = await registry.execute_tool("echo", message="test")
            assert result.success is True
            assert result.content == "test"

        asyncio.run(_test())

    def test_execute_missing_tool(self):
        async def _test():
            registry = ToolRegistry()
            result = await registry.execute_tool("nonexistent", message="test")
            assert result.success is False
            assert "not found" in result.error

        asyncio.run(_test())

    def test_validation_missing_param(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())
            result = await registry.execute_tool("echo")
            assert result.success is False
            assert "Missing required parameter" in result.error

        asyncio.run(_test())


class TestToolBaseClass:
    def test_echo_tool_read_only(self):
        tool = EchoTool()
        assert tool.read_only is True

    def test_echo_tool_concurrency_safe_default(self):
        tool = EchoTool()
        assert tool.concurrency_safe is True

    def test_echo_tool_to_openai_schema(self):
        tool = EchoTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_echo_tool_cast_params(self):
        tool = EchoTool()
        params = {"message": "test", "extra": "value"}
        result = tool.cast_params(params)
        assert result == params

    def test_validate_params_no_schema_type(self):
        class NoSchemaTypeTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test desc"

            @property
            def parameters(self):
                return {"properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        tool = NoSchemaTypeTool()
        errors = tool.validate_params({"any": "value"})
        assert errors == []

    def test_validate_params_all_required_provided(self):
        class SimpleTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test desc"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = SimpleTool()
        errors = tool.validate_params({"a": "hello", "b": 42})
        assert errors == []

    def test_validate_params_multiple_missing(self):
        class SimpleTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test desc"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = SimpleTool()
        errors = tool.validate_params({})
        assert len(errors) == 2
        assert "a" in errors[0]
        assert "b" in errors[1]


class TestToolExecutionResult:
    def test_tool_execution_result_success(self):
        from QuantNodes.agent.tools.base import ToolExecutionResult

        result = ToolExecutionResult(
            tool_name="echo",
            success=True,
            content="hello",
        )
        assert result.tool_name == "echo"
        assert result.success is True
        assert result.content == "hello"
        assert result.error is None

    def test_tool_execution_result_error(self):
        from QuantNodes.agent.tools.base import ToolExecutionResult

        result = ToolExecutionResult(
            tool_name="echo",
            success=False,
            content=None,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestToolRegistryAdvanced:
    def test_unregister_tool(self):
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)

        assert registry.get("echo") is not None

        registry.unregister("echo")

        assert registry.get("echo") is None

    def test_unregister_nonexistent_tool(self):
        registry = ToolRegistry()
        registry.unregister("nonexistent")

    def test_get_tool_schemas_caching(self):
        registry = ToolRegistry()
        registry.register(EchoTool())

        schemas1 = registry.get_tool_schemas()
        schemas2 = registry.get_tool_schemas()

        assert schemas1 is schemas2

    def test_get_tool_schemas_cache_invalidated_on_register(self):
        registry = ToolRegistry()
        registry.register(EchoTool())

        schemas1 = registry.get_tool_schemas()

        class AnotherTool(Tool):
            @property
            def name(self):
                return "another"

            @property
            def description(self):
                return "another tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        registry.register(AnotherTool())
        schemas2 = registry.get_tool_schemas()

        assert schemas1 is not schemas2
        assert len(schemas2) == 2

    def test_execute_tool_with_exception(self):
        class FailingTool(Tool):
            @property
            def name(self):
                return "failing"

            @property
            def description(self):
                return "fails"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                raise RuntimeError("Something went wrong!")

        async def _test():
            registry = ToolRegistry()
            registry.register(FailingTool())
            result = await registry.execute_tool("failing")
            return result

        result = asyncio.run(_test())
        assert result.success is False
        assert "Something went wrong!" in result.error

    def test_execute_tool_schema_caching(self):
        class CustomTool(Tool):
            @property
            def name(self):
                return "custom"

            @property
            def description(self):
                return "custom tool"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        registry = ToolRegistry()
        registry.register(CustomTool())

        schemas = registry.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "custom"


class TestToolRegistryParallelExecutionErrors:
    def test_execute_parallel_with_missing_tool(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())

            calls = [
                {"name": "echo", "arguments": {"message": "hello"}},
                {"name": "nonexistent", "arguments": {}},
            ]
            results = await registry.execute_tools_parallel(calls)
            return results

        results = asyncio.run(_test())
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False

    def test_execute_parallel_with_exception_in_tool(self):
        class FailingReadOnlyTool(Tool):
            @property
            def name(self):
                return "failing_ro"

            @property
            def description(self):
                return "fails"

            @property
            def read_only(self):
                return True

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                raise RuntimeError("Oops!")

        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())
            registry.register(FailingReadOnlyTool())

            calls = [
                {"name": "echo", "arguments": {"message": "hello"}},
                {"name": "failing_ro", "arguments": {}},
            ]
            results = await registry.execute_tools_parallel(calls)
            return results

        results = asyncio.run(_test())
        assert len(results) == 2
        assert results[0].success is True or results[1].success is True
        assert any(r.success is False and "Oops" in (r.error or "") for r in results)

    def test_execute_parallel_empty_list(self):
        async def _test():
            registry = ToolRegistry()
            results = await registry.execute_tools_parallel([])
            return results

        results = asyncio.run(_test())
        assert results == []

    def test_execute_parallel_missing_name_field(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())

            calls = [
                {"arguments": {"message": "hello"}},
            ]
            results = await registry.execute_tools_parallel(calls)
            return results

        results = asyncio.run(_test())
        assert len(results) == 1
        assert results[0].success is False


class TestSchemaValidationBoundary:
    def test_validate_params_string_type(self):
        class StringTool(Tool):
            @property
            def name(self):
                return "string_test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = StringTool()
        errors = tool.validate_params({"value": "hello"})
        assert errors == []

    def test_validate_params_number_type(self):
        class NumberTool(Tool):
            @property
            def name(self):
                return "number_test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = NumberTool()
        errors = tool.validate_params({"value": 42.5})
        assert errors == []

    def test_validate_params_integer_type(self):
        class IntTool(Tool):
            @property
            def name(self):
                return "int_test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = IntTool()
        errors = tool.validate_params({"value": 42})
        assert errors == []

    def test_validate_params_boolean_type(self):
        class BoolTool(Tool):
            @property
            def name(self):
                return "bool_test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"value": {"type": "boolean"}},
                    "required": ["value"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = BoolTool()
        errors = tool.validate_params({"value": True})
        assert errors == []

    def test_validate_params_array_type(self):
        class ArrayTool(Tool):
            @property
            def name(self):
                return "array_test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"value": {"type": "array"}},
                    "required": ["value"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = ArrayTool()
        errors = tool.validate_params({"value": [1, 2, 3]})
        assert errors == []

    def test_validate_params_object_type(self):
        class ObjectTool(Tool):
            @property
            def name(self):
                return "object_test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {"inner": {"type": "string"}},
                        }
                    },
                    "required": ["nested"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = ObjectTool()
        errors = tool.validate_params({"nested": {"inner": "hello"}})
        assert errors == []

    def test_validate_params_nested_required(self):
        class NestedRequiredTool(Tool):
            @property
            def name(self):
                return "nested_required"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "outer": {
                            "type": "object",
                            "properties": {"inner": {"type": "string"}},
                            "required": ["inner"],
                        }
                    },
                    "required": ["outer"],
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = NestedRequiredTool()
        errors = tool.validate_params({"outer": {"inner": "hello"}})
        assert errors == []


class TestToolConcurrencyMarkers:
    def test_default_read_only_is_false(self):
        class DefaultReadOnlyTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        tool = DefaultReadOnlyTool()
        assert tool.read_only is False

    def test_overridden_read_only(self):
        class ReadOnlyTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test"

            @property
            def read_only(self):
                return True

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        tool = ReadOnlyTool()
        assert tool.read_only is True

    def test_default_concurrency_safe_is_true(self):
        class DefaultConcurrencyTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        tool = DefaultConcurrencyTool()
        assert tool.concurrency_safe is True

    def test_overridden_concurrency_safe(self):
        class NotConcurrencySafeTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test"

            @property
            def concurrency_safe(self):
                return False

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        tool = NotConcurrencySafeTool()
        assert tool.concurrency_safe is False

    def test_echo_tool_concurrency_markers(self):
        tool = EchoTool()
        assert tool.read_only is True
        assert tool.concurrency_safe is True


class TestToolSchemaCaching:
    def test_to_openai_schema_returns_correct_structure(self):
        class SimpleTool(Tool):
            @property
            def name(self):
                return "simple"

            @property
            def description(self):
                return "A simple tool"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                }

            async def execute(self, **kwargs):
                return "ok"

        tool = SimpleTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "simple"
        assert schema["function"]["description"] == "A simple tool"
        assert "parameters" in schema["function"]

    def test_to_openai_schema_contains_all_properties(self):
        tool = EchoTool()
        schema = tool.to_openai_schema()

        assert "type" in schema
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert "properties" in schema["function"]["parameters"]


class TestToolExecutionResultDataclass:
    def test_tool_execution_result_all_fields(self):
        from QuantNodes.agent.tools.base import ToolExecutionResult

        result = ToolExecutionResult(
            tool_name="test",
            success=True,
            content={"data": "value"},
            error=None,
        )
        assert result.tool_name == "test"
        assert result.success is True
        assert result.content == {"data": "value"}
        assert result.error is None

    def test_tool_execution_result_mutable(self):
        from QuantNodes.agent.tools.base import ToolExecutionResult

        result = ToolExecutionResult(
            tool_name="test",
            success=False,
            content=None,
            error="initial error",
        )
        result.error = "updated error"
        result.content = "recovered content"

        assert result.error == "updated error"
        assert result.content == "recovered content"


class TestToolParamsCasting:
    def test_cast_params_returns_same_values_by_default(self):
        class SimpleTool(Tool):
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return "test"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        tool = SimpleTool()
        input_params = {"a": 1, "b": "hello", "c": True}
        result = tool.cast_params(input_params)

        assert result == input_params

    def test_cast_params_with_extra_fields(self):
        tool = EchoTool()
        params = {"message": "test", "unknown_field": "value", "another": 42}
        result = tool.cast_params(params)

        assert result == params

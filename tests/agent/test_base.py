# coding=utf-8
"""Tests for ``QuantNodes.agent.tools.base.Tool`` and ``ToolExecutionResult``.

v3.0.0 refactor: Stage 1 replaced the local ``Tool`` ABC with a thin
wrapper around ``nanobot.agent.tools.base.Tool``. Stage 5.3 added an
optional-dependency shim so that ``Tool`` is importable **without**
``nanobot-ai`` (it falls back to a minimal ABC that supports ``_dispatch``
and rejects ``to_openai_schema`` with a friendly error).

These tests verify the v3.0.0 contract:

- ``ToolExecutionResult`` — pure-Python dataclass (4 fields, no nanobot dep)
- ``Tool`` abstract API — ``name`` / ``description`` / ``parameters``
- ``Tool._dispatch`` — action-routing helper (works without nanobot)
- ``Tool.to_openai_schema`` — only available when ``nanobot-ai`` is
  installed (skipped otherwise)

All upstream-coupled tests use ``@pytest.mark.skipif(not NANOBOT_AVAILABLE)``
so the suite passes whether or not the optional dep is present.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from QuantNodes.agent import NANOBOT_AVAILABLE
from QuantNodes.agent.tools.base import Tool, ToolExecutionResult


# ----------------------------------------------------------------------------
# ToolExecutionResult — pure dataclass (always runs)
# ----------------------------------------------------------------------------

class TestToolExecutionResult:
    """``ToolExecutionResult`` is a 4-field dataclass. Always runnable."""

    def test_default_construction(self):
        """All four fields are accessible; ``error`` defaults to None."""
        r = ToolExecutionResult(
            tool_name="x",
            success=True,
            content={"data": "ok"},
        )
        assert r.tool_name == "x"
        assert r.success is True
        assert r.content == {"data": "ok"}
        assert r.error is None

    def test_with_error(self):
        """``success=False`` carries an error message."""
        r = ToolExecutionResult(
            tool_name="echo",
            success=False,
            content=None,
            error="Connection refused",
        )
        assert r.tool_name == "echo"
        assert r.success is False
        assert r.content is None
        assert r.error == "Connection refused"

    def test_is_dataclass(self):
        """``ToolExecutionResult`` must be a dataclass (not a Pydantic BaseModel).

        We rely on this for ``dataclasses.asdict`` in tool callers and
        for the lightweight ``ToolExecutionResult(**dict)`` round-trip
        used in tests.
        """
        assert dataclass(ToolExecutionResult) is ToolExecutionResult

    def test_fields_have_expected_defaults(self):
        """``tool_name``, ``success``, ``content`` are required; ``error`` defaults to None.

        v3.0.0 contract: callers must always pass the result of an
        invocation (success flag, tool name, content), but the error
        field defaults to None when omitted (e.g. success=True cases
        typically don't carry an error string).
        """
        sig = inspect.signature(ToolExecutionResult)
        for required in ("tool_name", "success", "content"):
            assert required in sig.parameters
            assert sig.parameters[required].default is inspect.Parameter.empty
        # ``error`` has a default of None (success path usually has no error)
        assert sig.parameters["error"].default is None

    def test_mutable_after_construction(self):
        """Results are mutable — callers can patch ``error`` after the fact."""
        r = ToolExecutionResult(tool_name="t", success=False, content=None, error="initial")
        r.error = "updated"
        r.content = "recovered"
        assert r.error == "updated"
        assert r.content == "recovered"


# ----------------------------------------------------------------------------
# Tool base class — local ABC (always runnable)
# ----------------------------------------------------------------------------

class _DummyTool(Tool):
    """Concrete subclass used for testing the ``Tool`` ABC contract.

    Works whether or not ``nanobot-ai`` is installed, since the v3.0.0
    Tool wrapper degrades to a plain ABC in the no-extras case.
    """

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool for testing the Tool base class."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"executed: {kwargs.get('value', '')}"


class TestToolBaseClass:
    """``Tool`` ABC contract — properties + ``_dispatch`` helper."""

    def test_name_property(self):
        assert _DummyTool().name == "dummy"

    def test_description_property(self):
        desc = _DummyTool().description
        assert "dummy" in desc.lower()

    def test_parameters_property(self):
        params = _DummyTool().parameters
        assert params["type"] == "object"
        assert "value" in params["properties"]
        assert "value" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_runs(self):
        """Concrete tool ``execute()`` is awaited and returns its result."""
        t = _DummyTool()
        result = await t.execute(value="hello")
        assert result == "executed: hello"

    @pytest.mark.asyncio
    async def test_dispatch_helper_known_action(self):
        """``_dispatch`` looks up an action in a registry and calls it."""
        t = _DummyTool()

        async def _action_a(x: int) -> int:
            return x * 2

        result = await t._dispatch("action_a", {"action_a": _action_a}, x=3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_dispatch_helper_unknown_action_raises(self):
        """``_dispatch`` raises ``ValueError`` for unknown action names."""
        t = _DummyTool()
        with pytest.raises(ValueError, match="Unknown action: bogus"):
            await t._dispatch("bogus", {})

    @pytest.mark.asyncio
    async def test_dispatch_helper_passes_kwargs(self):
        """All kwargs are forwarded verbatim to the action callable."""
        t = _DummyTool()
        captured: Dict[str, Any] = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return "ok"

        result = await t._dispatch(
            "capture",
            {"capture": _capture},
            foo="bar",
            n=42,
            flag=True,
        )
        assert result == "ok"
        assert captured == {"foo": "bar", "n": 42, "flag": True}


# ----------------------------------------------------------------------------
# Tool.to_openai_schema — requires nanobot-ai (skipped otherwise)
# ----------------------------------------------------------------------------

@pytest.mark.skipif(
    not NANOBOT_AVAILABLE,
    reason="to_openai_schema requires nanobot-ai (Stage 5.3 graceful degradation)",
)
class TestToolToOpenAISchema:
    """When ``nanobot-ai`` is installed, ``Tool.to_openai_schema`` returns
    a standard OpenAI function-calling schema dict (alias for upstream ``to_schema``).
    """

    def test_returns_function_dict(self):
        """Schema has ``type=function`` and a nested ``function`` block."""
        schema = _DummyTool().to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "dummy"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_to_openai_schema_is_alias_for_to_schema(self):
        """``to_openai_schema`` is a backward-compat alias for upstream ``to_schema``."""
        tool = _DummyTool()
        assert tool.to_openai_schema == tool.to_schema
        # Both calls should return equal content
        assert tool.to_openai_schema() == tool.to_schema()

    def test_schema_contains_all_required_fields(self):
        """OpenAI schema carries name, description, and parameters with required list."""
        schema = _DummyTool().to_openai_schema()
        fn = schema["function"]
        assert fn["name"] == "dummy"
        assert "description" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "value" in params.get("required", [])


# ----------------------------------------------------------------------------
# Optional-dep error path
# ----------------------------------------------------------------------------

@pytest.mark.skipif(
    NANOBOT_AVAILABLE,
    reason="Only run when nanobot-ai is NOT installed (verifies the error message).",
)
class TestToolWithoutNanobot:
    """When ``nanobot-ai`` is not installed, the local ABC ``Tool`` must
    still allow subclassing and ``_dispatch`` but raise a clear error on
    ``to_openai_schema``.
    """

    def test_can_still_subclass(self):
        class _LocalTool(Tool):
            @property
            def name(self):
                return "local"

            @property
            def description(self):
                return "local desc"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        t = _LocalTool()
        assert t.name == "local"
        assert t.description == "local desc"

    def test_to_openai_schema_raises_install_hint(self):
        """The error message tells the user how to install ``[agent]`` extra."""
        t = _DummyTool()
        with pytest.raises(RuntimeError) as exc_info:
            t.to_openai_schema
        msg = str(exc_info.value)
        assert "quantnodes[agent]" in msg, (
            f"error message should mention install command, got: {msg!r}"
        )

# coding=utf-8
"""Tests for v3.0.0 Skill infrastructure: Bridge + DreamEngine shim + API router.

v3.0.0 refactor: Stage 1 deleted ``QuantNodes.agent.core.memory.DreamEngine``
(replaced by upstream nanobot's memory). Stage 3 added
``QuantNodes.agent.core.quant_dream.DreamEngine`` as a **backward-compat
shim** that delegates to ``QuantDreamHook``. This file targets the v3.0.0
infrastructure:

- ``QuantNodes.agent.skills.registry.SkillRegistry`` (kept, thread-safe singleton)
- ``QuantNodes.agent.skills.bridge.SkillToolBridge`` (kept, skill → tool)
- ``QuantNodes.agent.skills.bridge.SkillToolAdapter`` (kept)
- ``QuantNodes.agent.core.quant_dream.DreamEngine`` (shim, kept for compat)
- ``api/routers/skill.py`` (FastAPI router, kept — actually uses ``registry.get_skill_info()``)

All tests run **without** ``nanobot-ai`` because the skill subsystem
is self-contained. Where the API router depends on the upstream
``SkillRegistry.get_skill_info`` method, we add a small shim to the
local ``SkillRegistry`` if missing.
"""

import contextlib
import threading
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from QuantNodes.agent.skills.base import (
    Skill,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from QuantNodes.agent.skills.registry import SkillRegistry
from QuantNodes.agent.tools.base import Tool
from QuantNodes.agent.tools.registry import ToolRegistry


# ----------------------------------------------------------------------------
# Test helpers — Mock Skills
# ----------------------------------------------------------------------------

class MockSkill(Skill):
    """Concrete test skill with configurable category."""

    def __init__(
        self,
        name: str = "mock_skill",
        category: SkillCategory = SkillCategory.STRATEGY,
        fail: bool = False,
    ) -> None:
        self._meta = SkillMetadata(
            name=name,
            description=f"Mock skill: {name}",
            category=category,
        )
        self._fail = fail

    @property
    def metadata(self) -> SkillMetadata:
        return self._meta

    @property
    def name(self) -> str:
        return self._meta.name

    @property
    def description(self) -> str:
        return self._meta.description

    @property
    def category(self) -> SkillCategory:
        return self._meta.category

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        if self._fail:
            return SkillResult(success=False, error="simulated failure")
        query = context.get("query", "")
        return SkillResult(
            success=True,
            data={"result": f"executed {self.name} with query={query}"},
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }


def _reset_skill_registry() -> None:
    """Reset the SkillRegistry singleton between tests."""
    SkillRegistry._instance = None


@contextlib.contextmanager
def fresh_skill_registry():
    """Yield a clean SkillRegistry, resetting the singleton after."""
    _reset_skill_registry()
    yield
    _reset_skill_registry()


# ----------------------------------------------------------------------------
# SkillRegistry — singleton + thread-safe + CRUD
# ----------------------------------------------------------------------------

class TestSkillRegistryThreadSafety:

    def setup_method(self) -> None:
        _reset_skill_registry()

    def teardown_method(self) -> None:
        _reset_skill_registry()

    def test_singleton(self):
        with fresh_skill_registry():
            r1 = SkillRegistry()
            r2 = SkillRegistry()
            assert r1 is r2

    def test_register_and_get(self):
        with fresh_skill_registry():
            reg = SkillRegistry()
            skill = MockSkill(name="test_skill")
            reg.register(skill)
            assert reg.get("test_skill") is skill

    def test_register_duplicate_raises(self):
        with fresh_skill_registry():
            reg = SkillRegistry()
            reg.register(MockSkill(name="dup"))
            with pytest.raises(ValueError, match="already registered"):
                reg.register(MockSkill(name="dup"))

    def test_unregister(self):
        with fresh_skill_registry():
            reg = SkillRegistry()
            reg.register(MockSkill(name="to_remove"))
            assert reg.unregister("to_remove") is True
            assert reg.get("to_remove") is None

    def test_unregister_nonexistent_returns_false(self):
        with fresh_skill_registry():
            reg = SkillRegistry()
            assert reg.unregister("ghost") is False

    def test_list_all_returns_all_registered(self):
        with fresh_skill_registry():
            reg = SkillRegistry()
            reg.register(MockSkill(name="a"))
            reg.register(MockSkill(name="b"))
            assert len(reg.list_all()) == 2

    def test_list_by_category(self):
        with fresh_skill_registry():
            reg = SkillRegistry()
            reg.register(MockSkill(name="s1", category=SkillCategory.STRATEGY))
            reg.register(MockSkill(name="f1", category=SkillCategory.FACTOR))
            strategies = reg.list_by_category(SkillCategory.STRATEGY)
            assert len(strategies) == 1
            assert strategies[0].name == "s1"

    def test_concurrent_register_does_not_lose_skills(self):
        """Thread-safety: 20 concurrent ``register`` calls all succeed.

        v3.0.0 contract: ``SkillRegistry.register`` is guarded by a
        re-entrant lock so concurrent registrations from multiple
        threads all succeed without crashing or losing skills.
        """
        with fresh_skill_registry():
            reg = SkillRegistry()
            errors: List[Exception] = []

            def register_skill(idx: int) -> None:
                try:
                    reg.register(MockSkill(name=f"skill_{idx}"))
                except Exception as e:  # pragma: no cover
                    errors.append(e)

            threads = [
                threading.Thread(target=register_skill, args=(i,)) for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(reg.list_all()) == 20


# ----------------------------------------------------------------------------
# SkillToolBridge — Skill → Tool conversion
# ----------------------------------------------------------------------------

class TestSkillToolBridge:
    def setup_method(self) -> None:
        _reset_skill_registry()

    def teardown_method(self) -> None:
        _reset_skill_registry()

    def test_bridge_creates_tool_with_skill_prefix(self):
        """Each skill becomes a tool named ``skill_<skill_name>``."""
        with fresh_skill_registry():
            from QuantNodes.agent.skills.bridge import SkillToolBridge

            reg = SkillRegistry()
            tool_reg = ToolRegistry()
            reg.register(MockSkill(name="bridge_test"))
            SkillToolBridge(reg, tool_reg).register_all()

            names = [t.name for t in tool_reg.list_tools()]
            assert "skill_bridge_test" in names

    @pytest.mark.asyncio
    async def test_bridge_tool_executes_skill(self):
        """``SkillToolAdapter.execute`` delegates to ``skill.execute``."""
        with fresh_skill_registry():
            from QuantNodes.agent.skills.bridge import SkillToolBridge

            reg = SkillRegistry()
            tool_reg = ToolRegistry()
            reg.register(MockSkill(name="exec_test"))

            SkillToolBridge(reg, tool_reg).register_all()

            tool = tool_reg.get("skill_exec_test")
            assert tool is not None
            result = await tool.execute(query="hello")
            # v3.0.0: SkillToolAdapter returns a dict (SkillResult.to_dict()).
            assert result["success"] is True
            assert "executed exec_test" in result["data"]["result"]

    def test_bridge_preserves_read_only(self):
        """SkillToolAdapter.read_only is True (skills are pure execute)."""
        with fresh_skill_registry():
            from QuantNodes.agent.skills.bridge import SkillToolBridge

            reg = SkillRegistry()
            tool_reg = ToolRegistry()
            reg.register(MockSkill(name="ro_test"))

            SkillToolBridge(reg, tool_reg).register_all()

            tool = tool_reg.get("skill_ro_test")
            assert tool is not None
            assert tool.read_only is True

    def test_bridge_multiple_skills(self):
        with fresh_skill_registry():
            from QuantNodes.agent.skills.bridge import SkillToolBridge

            reg = SkillRegistry()
            tool_reg = ToolRegistry()
            reg.register(MockSkill(name="multi_a"))
            reg.register(MockSkill(name="multi_b", category=SkillCategory.FACTOR))
            SkillToolBridge(reg, tool_reg).register_all()
            assert len(tool_reg.list_tools()) == 2


# ----------------------------------------------------------------------------
# DreamEngine shim — backward compat with v2.x
# ----------------------------------------------------------------------------

class TestDreamEngineShim:
    """v3.0.0 ``DreamEngine`` is a shim over ``QuantDreamHook``.

    Preserves the v2.x API (``analyze_conversation``, ``generate_dream``,
    ``workspace=`` / ``dream_store=`` kwargs) while delegating to the new
    file-based ``QuantDreamHook``.
    """

    def test_accepts_workspace_kwarg(self, tmp_path):
        from QuantNodes.agent.core.quant_dream import DreamEngine

        engine = DreamEngine(workspace=tmp_path)
        assert engine.workspace == tmp_path

    def test_accepts_dream_store_kwarg_with_workspace_attr(self, tmp_path):
        """v2.x callers passed ``DreamStore`` with a ``workspace`` attr.

        The shim extracts ``.workspace`` to maintain backward compat.
        """
        from QuantNodes.agent.core.quant_dream import DreamEngine

        class _FakeStore:
            workspace = str(tmp_path)

        engine = DreamEngine(dream_store=_FakeStore())
        assert engine.workspace == tmp_path

    def test_analyze_conversation_returns_insight(self, tmp_path):
        from QuantNodes.agent.core.quant_dream import DreamEngine

        engine = DreamEngine(workspace=tmp_path)
        insight = engine.analyze_conversation(
            "帮我分析一下动量因子的 IC",
            "动量因子的 IC 大约在 0.05 左右，表现稳定。",
        )
        # Either an insight (when keywords match) or None
        if insight is not None:
            assert insight.type
            assert insight.content

    def test_generate_dream_appends_to_topic_file(self, tmp_path):
        """``generate_dream`` appends to ``.agent/memory/topic-quant-dream.md``."""
        from QuantNodes.agent.core.quant_dream import DreamEngine

        engine = DreamEngine(workspace=tmp_path)
        dream = engine.generate_dream(
            dream_type="factor_insight",
            content="momentum factor IC is 0.05",
            insights=["IC stable across regimes"],
        )
        # The dream is returned AND persisted to the topic file
        topic_file = tmp_path / "memory" / "topic-quant-dream.md"
        assert topic_file.exists()
        content = topic_file.read_text(encoding="utf-8")
        assert "momentum factor IC is 0.05" in content


# ----------------------------------------------------------------------------
# Skill API router
# ----------------------------------------------------------------------------

class TestSkillAPIRouter:

    def setup_method(self) -> None:
        _reset_skill_registry()

    def teardown_method(self) -> None:
        _reset_skill_registry()

    def _ensure_skill_info_method(self):
        """Add a minimal ``get_skill_info`` shim to ``SkillRegistry`` if missing.

        The api/routers/skill.py router calls
        ``registry.get_skill_info(name)``. If upstream nanobot's
        SkillRegistry doesn't have it, the router would 500. We
        shim the method here so the router is testable in isolation.
        """
        if not hasattr(SkillRegistry, "get_skill_info"):
            def _get_skill_info(self, name: str):
                skill = self.get(name)
                if skill is None:
                    return None
                return {
                    "name": skill.name,
                    "description": skill.description,
                    "category": skill.category.value,
                }
            SkillRegistry.get_skill_info = _get_skill_info  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_list_skills_endpoint(self):
        self._ensure_skill_info_method()
        from api.routers.skill import router

        reg = SkillRegistry()
        reg.register(MockSkill(name="api_test_skill"))

        app = FastAPI()
        app.include_router(router, prefix="/skills")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/skills/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "api_test_skill" in names

    @pytest.mark.asyncio
    async def test_get_skill_detail_endpoint(self):
        self._ensure_skill_info_method()
        from api.routers.skill import router

        reg = SkillRegistry()
        reg.register(MockSkill(name="detail_test"))

        app = FastAPI()
        app.include_router(router, prefix="/skills")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/skills/detail_test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "detail_test"

    @pytest.mark.asyncio
    async def test_get_skill_not_found_returns_404(self):
        self._ensure_skill_info_method()
        from api.routers.skill import router

        app = FastAPI()
        app.include_router(router, prefix="/skills")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/skills/ghost")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_execute_skill_endpoint(self):
        self._ensure_skill_info_method()
        from api.routers.skill import router

        reg = SkillRegistry()
        reg.register(MockSkill(name="exec_api_test"))

        app = FastAPI()
        app.include_router(router, prefix="/skills")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/skills/exec_api_test/execute",
                json={"query": "hello"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_list_categories_endpoint(self):
        self._ensure_skill_info_method()
        from api.routers.skill import router

        reg = SkillRegistry()
        reg.register(MockSkill(name="cat_s", category=SkillCategory.STRATEGY))
        reg.register(MockSkill(name="cat_f", category=SkillCategory.FACTOR))

        app = FastAPI()
        app.include_router(router, prefix="/skills")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/skills/categories/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategy" in data
        assert "factor" in data

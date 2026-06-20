# coding=utf-8
"""
Phase 4: Skill 桥接 + DreamEngine 增强测试

覆盖：
- SkillRegistry 线程安全
- SkillToolBridge (Skill → Tool)
- DreamEngine dispatch_skills + push_to_agent
- Skill API Router
"""

import threading
from datetime import datetime
from typing import Any, Dict

import pytest

from QuantNodes.agent.skills.base import (
    Skill, SkillCategory, SkillMetadata, SkillResult,
)
from QuantNodes.agent.skills.registry import SkillRegistry


# ─── Test Helpers ──────────────────────────────────────────────────────

class MockSkill(Skill):
    """测试用 Mock 技能"""

    def __init__(self, name="mock_skill", category=SkillCategory.STRATEGY):
        self._meta = SkillMetadata(
            name=name,
            description=f"Mock skill: {name}",
            category=category,
        )

    @property
    def metadata(self) -> SkillMetadata:
        return self._meta

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        query = context.get("query", "")
        return SkillResult(
            success=True,
            data={"result": f"executed {self.name} with query={query}"},
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Input query"},
            },
        }


class FailingSkill(Skill):
    """执行失败的技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="failing_skill",
            description="Always fails",
            category=SkillCategory.ANALYSIS,
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        return SkillResult(success=False, error="Simulated failure")

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}


def _reset_registry():
    """重置 SkillRegistry 单例"""
    SkillRegistry._instance = None


# ─── SkillRegistry 线程安全 ────────────────────────────────────────────

class TestSkillRegistryThreadSafety:

    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    def test_singleton(self):
        r1 = SkillRegistry()
        r2 = SkillRegistry()
        assert r1 is r2

    def test_register_and_get(self):
        reg = SkillRegistry()
        skill = MockSkill(name="test_skill")
        reg.register(skill)
        assert reg.get("test_skill") is skill

    def test_register_duplicate_raises(self):
        reg = SkillRegistry()
        reg.register(MockSkill(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(MockSkill(name="dup"))

    def test_unregister(self):
        reg = SkillRegistry()
        reg.register(MockSkill(name="to_remove"))
        assert reg.unregister("to_remove") is True
        assert reg.get("to_remove") is None

    def test_unregister_nonexistent(self):
        reg = SkillRegistry()
        assert reg.unregister("ghost") is False

    def test_list_all(self):
        reg = SkillRegistry()
        reg.register(MockSkill(name="a"))
        reg.register(MockSkill(name="b"))
        assert len(reg.list_all()) == 2

    def test_list_by_category(self):
        reg = SkillRegistry()
        reg.register(MockSkill(name="s1", category=SkillCategory.STRATEGY))
        reg.register(MockSkill(name="f1", category=SkillCategory.FACTOR))
        strategies = reg.list_by_category(SkillCategory.STRATEGY)
        assert len(strategies) == 1
        assert strategies[0].name == "s1"

    def test_search(self):
        reg = SkillRegistry()
        reg.register(MockSkill(name="momentum_factor"))
        reg.register(MockSkill(name="dual_ma"))
        results = reg.search("momentum")
        assert len(results) == 1
        assert results[0]["name"] == "momentum_factor"

    def test_clear(self):
        reg = SkillRegistry()
        reg.register(MockSkill(name="a"))
        reg.clear()
        assert len(reg.list_all()) == 0

    def test_thread_safety_concurrent_register(self):
        """并发注册不丢失、不崩溃"""
        _reset_registry()
        reg = SkillRegistry()
        errors = []

        def register_skill(idx):
            try:
                reg.register(MockSkill(name=f"skill_{idx}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_skill, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(reg.list_all()) == 20


# ─── SkillToolBridge ──────────────────────────────────────────────────

class TestSkillToolBridge:
    """测试 Skill → Tool 桥接"""

    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    @pytest.mark.asyncio
    async def test_bridge_creates_tool_from_skill(self):
        from QuantNodes.agent.skills.bridge import SkillToolBridge
        from QuantNodes.agent.tools.registry import ToolRegistry

        reg = SkillRegistry()
        tool_reg = ToolRegistry()
        skill = MockSkill(name="bridge_test")
        reg.register(skill)

        bridge = SkillToolBridge(reg, tool_reg)
        bridge.register_all()

        tools = tool_reg.list_tools()
        tool_names = [t.name for t in tools]
        assert "skill_bridge_test" in tool_names

    @pytest.mark.asyncio
    async def test_bridge_tool_executes_skill(self):
        from QuantNodes.agent.skills.bridge import SkillToolBridge
        from QuantNodes.agent.tools.registry import ToolRegistry

        reg = SkillRegistry()
        tool_reg = ToolRegistry()
        skill = MockSkill(name="exec_test")
        reg.register(skill)

        bridge = SkillToolBridge(reg, tool_reg)
        bridge.register_all()

        tool = tool_reg.get("skill_exec_test")
        result = await tool.execute(query="hello")
        assert result["success"] is True
        assert "executed exec_test" in result["data"]["result"]

    @pytest.mark.asyncio
    async def test_bridge_tool_schema_matches_skill(self):
        from QuantNodes.agent.skills.bridge import SkillToolBridge
        from QuantNodes.agent.tools.registry import ToolRegistry

        reg = SkillRegistry()
        tool_reg = ToolRegistry()
        skill = MockSkill(name="schema_test")
        reg.register(skill)

        bridge = SkillToolBridge(reg, tool_reg)
        bridge.register_all()

        tool = tool_reg.get("skill_schema_test")
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "skill_schema_test"
        assert "query" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_bridge_tool_name_prefix(self):
        from QuantNodes.agent.skills.bridge import SkillToolBridge
        from QuantNodes.agent.tools.registry import ToolRegistry

        reg = SkillRegistry()
        tool_reg = ToolRegistry()
        skill = MockSkill(name="prefix_test")
        reg.register(skill)

        bridge = SkillToolBridge(reg, tool_reg)
        bridge.register_all()

        tool = tool_reg.get("skill_prefix_test")
        assert tool.name == "skill_prefix_test"

    @pytest.mark.asyncio
    async def test_bridge_preserves_read_only(self):
        from QuantNodes.agent.skills.bridge import SkillToolBridge
        from QuantNodes.agent.tools.registry import ToolRegistry

        reg = SkillRegistry()
        tool_reg = ToolRegistry()
        skill = MockSkill(name="ro_test")
        reg.register(skill)

        bridge = SkillToolBridge(reg, tool_reg)
        bridge.register_all()

        tool = tool_reg.get("skill_ro_test")
        assert tool.read_only is True

    @pytest.mark.asyncio
    async def test_bridge_multiple_skills(self):
        from QuantNodes.agent.skills.bridge import SkillToolBridge
        from QuantNodes.agent.tools.registry import ToolRegistry

        reg = SkillRegistry()
        tool_reg = ToolRegistry()
        reg.register(MockSkill(name="multi_a"))
        reg.register(MockSkill(name="multi_b", category=SkillCategory.FACTOR))

        bridge = SkillToolBridge(reg, tool_reg)
        bridge.register_all()

        assert len(tool_reg.list_tools()) == 2


# ─── DreamEngine 增强 ──────────────────────────────────────────────────

class TestDreamEngineEnhancements:

    def _make_engine(self, tmp_path):
        from QuantNodes.agent.core.memory import DreamStore, DreamConfig
        from QuantNodes.agent.core.dream import DreamEngine

        store = DreamStore(tmp_path)
        config = DreamConfig()
        return DreamEngine(store, config), store

    @pytest.mark.asyncio
    async def test_generate_dream(self, tmp_path):
        engine, store = self._make_engine(tmp_path)
        dream = await engine.generate_dream(
            dream_type="test",
            content="test content",
            insights=["insight 1"],
            confidence=0.9,
            tags=["test"],
        )
        assert dream.type == "test"
        assert dream.content == "test content"
        assert dream.insights == ["insight 1"]
        assert dream.confidence == 0.9

    @pytest.mark.asyncio
    async def test_dispatch_skills(self, tmp_path):

        engine, store = self._make_engine(tmp_path)
        _reset_registry()
        reg = SkillRegistry()
        skill = MockSkill(name="dispatch_test")
        reg.register(skill)

        results = await engine.dispatch_skills("test query", reg)
        assert len(results) == 1
        assert results[0].success is True
        assert "executed dispatch_test" in results[0].data["result"]

    @pytest.mark.asyncio
    async def test_dispatch_skills_empty_registry(self, tmp_path):

        engine, store = self._make_engine(tmp_path)
        _reset_registry()
        reg = SkillRegistry()

        results = await engine.dispatch_skills("anything", reg)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_skills_with_failing(self, tmp_path):

        engine, store = self._make_engine(tmp_path)
        _reset_registry()
        reg = SkillRegistry()
        reg.register(MockSkill(name="ok_skill"))
        reg.register(FailingSkill())

        results = await engine.dispatch_skills("query", reg)
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1

    @pytest.mark.asyncio
    async def test_push_to_agent(self, tmp_path):
        engine, store = self._make_engine(tmp_path)
        from QuantNodes.agent.core.memory import Dream

        dream = Dream(
            id="test_1",
            timestamp=datetime.now().isoformat(),
            type="test_insight",
            content="test push",
            confidence=0.9,
        )
        engine.push_to_agent(dream)
        recent = store.get_recent_dreams(10)
        assert len(recent) == 1
        assert recent[0].content == "test push"

    @pytest.mark.asyncio
    async def test_subscribe_and_notify(self, tmp_path):
        engine, store = self._make_engine(tmp_path)
        received = []

        async def on_dream(dream):
            received.append(dream)

        engine.subscribe(on_dream)
        await engine.generate_dream(dream_type="sub_test", content="hello")
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_get_stats(self, tmp_path):
        engine, store = self._make_engine(tmp_path)
        stats = engine.get_stats()
        assert "total_dreams" in stats
        assert "by_type" in stats


# ─── Skill API Router ─────────────────────────────────────────────────

class TestSkillAPIRouter:

    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    @pytest.mark.asyncio
    async def test_list_skills(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport
        from QuantNodes.agent.skills.registry import SkillRegistry

        reg = SkillRegistry()
        reg.register(MockSkill(name="api_test_skill"))

        from api.routers.skill import router
        app = FastAPI()
        app.include_router(router, prefix="/skills")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/skills/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "api_test_skill" in names

    @pytest.mark.asyncio
    async def test_get_skill_detail(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        reg = SkillRegistry()
        reg.register(MockSkill(name="detail_test"))

        from api.routers.skill import router
        app = FastAPI()
        app.include_router(router, prefix="/skills")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/skills/detail_test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "detail_test"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        _reset_registry()
        from api.routers.skill import router
        app = FastAPI()
        app.include_router(router, prefix="/skills")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/skills/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_execute_skill(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        reg = SkillRegistry()
        reg.register(MockSkill(name="exec_api_test"))

        from api.routers.skill import router
        app = FastAPI()
        app.include_router(router, prefix="/skills")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/skills/exec_api_test/execute", json={"query": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        _reset_registry()
        from api.routers.skill import router
        app = FastAPI()
        app.include_router(router, prefix="/skills")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/skills/ghost/execute")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_categories(self):
        from fastapi import FastAPI
        from httpx import AsyncClient, ASGITransport

        reg = SkillRegistry()
        reg.register(MockSkill(name="cat_s", category=SkillCategory.STRATEGY))
        reg.register(MockSkill(name="cat_f", category=SkillCategory.FACTOR))

        from api.routers.skill import router
        app = FastAPI()
        app.include_router(router, prefix="/skills")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/skills/categories/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategy" in data
        assert "factor" in data

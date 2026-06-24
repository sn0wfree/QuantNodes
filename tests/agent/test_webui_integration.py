# coding=utf-8
"""Tests for v3.0.0 Stage 5.3 WebUI integration (frontend iframe + runtime wiring).

Verifies the contract between the frontend iframe and the backend:

- ``/agent-chat`` route is registered in the Vue router
- ``VITE_AGENT_ENABLED`` toggles the sidebar entry
- The iframe target URL matches the gateway port (default 18080)
- The backend ``/api/agent/status`` contract that the iframe polls
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_router_has_agent_chat_route():
    """router/index.ts must register /agent-chat pointing to AgentChat.vue."""
    router_path = REPO_ROOT / "frontend" / "src" / "router" / "index.ts"
    text = router_path.read_text(encoding="utf-8")
    assert "path: 'agent-chat'" in text, "router missing /agent-chat path"
    assert "AgentChat.vue" in text, "router missing AgentChat.vue import"
    assert "name: 'AgentChat'" in text, "router missing AgentChat name"


def test_sidebar_uses_agent_chat_path():
    """AppSidebar.vue must use /agent-chat (not the stale /chat)."""
    sidebar = (REPO_ROOT / "frontend" / "src" / "components" / "Layout" / "AppSidebar.vue").read_text(encoding="utf-8")
    assert "/agent-chat" in sidebar, "sidebar must link to /agent-chat"
    assert 'key="/chat"' not in sidebar, "stale /chat key still present"


def test_sidebar_gates_agent_chat_by_env_flag():
    """Agent Chat entry must be hidden when VITE_AGENT_ENABLED=false."""
    sidebar = (REPO_ROOT / "frontend" / "src" / "components" / "Layout" / "AppSidebar.vue").read_text(encoding="utf-8")
    assert "agentEnabled" in sidebar, "sidebar must define agentEnabled flag"
    assert "VITE_AGENT_ENABLED" in sidebar, "sidebar must read VITE_AGENT_ENABLED env"


def test_agent_chat_view_uses_status_polling():
    """AgentChat.vue must poll /api/agent/status to render the right state."""
    view = (REPO_ROOT / "frontend" / "src" / "views" / "AgentChat.vue").read_text(encoding="utf-8")
    assert "/api/agent/status" in view
    assert "VITE_NANOBOT_GATEWAY_URL" in view
    # iframe target with sandbox attribute for safety
    assert "sandbox=" in view
    assert "<iframe" in view


def test_agent_chat_view_handles_unavailable_state():
    """AgentChat.vue must render an install prompt when nanobot-ai is missing."""
    view = (REPO_ROOT / "frontend" / "src" / "views" / "AgentChat.vue").read_text(encoding="utf-8")
    assert "pip install" in view
    assert "quantnodes[agent]" in view
    assert "未启用" in view or "unavailable" in view


def test_env_development_has_agent_flags():
    """frontend/.env.development must set VITE_AGENT_ENABLED=true + gateway URL."""
    env = (REPO_ROOT / "frontend" / ".env.development").read_text(encoding="utf-8")
    assert "VITE_AGENT_ENABLED=true" in env
    assert "VITE_NANOBOT_GATEWAY_URL" in env
    assert "18080" in env


def test_env_template_has_nanobot_section():
    """.env.template must document the new NANOBOT_GATEWAY_* env vars."""
    env = (REPO_ROOT / ".env.template").read_text(encoding="utf-8")
    assert "NANOBOT_GATEWAY_HOST" in env
    assert "NANOBOT_GATEWAY_PORT" in env
    assert "NANOBOT_WORKSPACE" in env
    assert "18080" in env
    assert "Stage 5.3" in env or "agent" in env.lower()


def test_pyproject_marks_nanobot_as_optional():
    """pyproject.toml must move nanobot-ai from dependencies to [agent] extra."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # nanobot-ai should NOT be in the main dependencies block
    deps_match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", pyproject, re.MULTILINE | re.DOTALL)
    assert deps_match, "pyproject must have a dependencies = [...] block"
    deps_block = deps_match.group(1)
    assert "nanobot-ai" not in deps_block, "nanobot-ai must NOT be in main dependencies"

    # nanobot-ai must be in [agent] extras
    agent_match = re.search(r"^agent\s*=\s*\[(.*?)\]", pyproject, re.MULTILINE | re.DOTALL)
    assert agent_match, "pyproject must have an agent = [...] extras block"
    agent_block = agent_match.group(1)
    assert "nanobot-ai" in agent_block, "nanobot-ai must be in [agent] extra"

    # all meta-extra
    assert "all" in pyproject, "pyproject should define an [all] meta-extra"


def test_api_main_wires_runtime_into_lifespan():
    """api/main.py must import + start the NanobotRuntime in lifespan."""
    main = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    assert "init_runtime" in main
    assert "shutdown_runtime" in main
    assert "nanobot_runtime" in main
    assert "agent_router" in main or "agent as agent_router" in main

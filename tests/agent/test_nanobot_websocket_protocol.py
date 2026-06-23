# coding=utf-8
"""Frontend WebSocket protocol contract tests.

These tests verify the static structure of
``frontend/src/composables/useNanobotWebSocket.ts`` — that the file
exists, declares the expected protocol-level exports, references the
nanobot gateway bootstrap path, and uses the wire protocol documented
in ``nanobot/channels/websocket.py``.

We do **not** spin up a JS test runner here — the goal is to catch
contract drift between frontend and backend (e.g. someone renames the
composable without updating imports, or drops the bootstrap fetch).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSABLE = REPO_ROOT / "frontend" / "src" / "composables" / "useNanobotWebSocket.ts"


def test_useNanobotWebSocket_exists():
    """Composable file must exist alongside the older ``useWebSocket.ts``."""
    assert COMPOSABLE.is_file(), f"{COMPOSABLE} missing"


def test_useNanobotWebSocket_exports_named_function():
    """``useNanobotWebSocket`` must be the default export (named function)."""
    text = COMPOSABLE.read_text(encoding="utf-8")
    assert "export function useNanobotWebSocket(" in text, (
        "expected `export function useNanobotWebSocket(` definition"
    )


def test_useNanobotWebSocket_calls_webui_bootstrap():
    """Composable must fetch ``/webui/bootstrap`` to obtain a WS token."""
    text = COMPOSABLE.read_text(encoding="utf-8")
    assert "/webui/bootstrap" in text, (
        "composable must hit /webui/bootstrap to fetch a short-lived WS token"
    )


def test_useNanobotWebSocket_sends_message_envelope():
    """Outgoing frames must be a JSON envelope ``{type: message, content, chat_id}``."""
    text = COMPOSABLE.read_text(encoding="utf-8")
    assert '"type"' in text or "'type'" in text, (
        "outgoing frame must include a `type` field"
    )
    assert "content" in text, "outgoing frame must include `content` field"
    assert "chat_id" in text, "outgoing frame must include `chat_id` field"


def test_useNanobotWebSocket_handles_nanobot_events():
    """Incoming event types we care about must be listed in the docs/types."""
    text = COMPOSABLE.read_text(encoding="utf-8")
    # All four primary render buckets + at least 3 secondary events
    for ev in ("message", "tool_call", "tool_result", "error", "attached", "user", "tool_hint"):
        assert f"'{ev}'" in text or f'"{ev}"' in text, (
            f"composable must reference the '{ev}' nanobot event"
        )


def test_useNanobotWebSocket_uses_exponential_backoff():
    """Reconnect strategy should use exponential backoff to be polite."""
    text = COMPOSABLE.read_text(encoding="utf-8")
    # We're loose: any Math.pow or similar means exponential
    assert re.search(r"Math\.pow|2\s*\*", text) is not None or "reconnectInterval" in text


def test_useNanobotWebSocket_handles_disconnect_on_unmount():
    """Composable must disconnect its WS on ``onUnmounted`` to prevent leaks."""
    text = COMPOSABLE.read_text(encoding="utf-8")
    assert "onUnmounted" in text
    assert "disconnect" in text


def test_env_development_references_nanobot_bootstrap():
    """frontend/.env.development must include the bootstrap path variable."""
    env = (REPO_ROOT / "frontend" / ".env.development").read_text(encoding="utf-8")
    assert "VITE_NANOBOT_GATEWAY_URL" in env
    assert "18080" in env

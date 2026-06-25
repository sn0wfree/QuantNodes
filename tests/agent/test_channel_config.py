# coding=utf-8
"""Tests for v3.0.0 Stage 5.4 — channel configuration (websocket + feishu).

Verifies that ``config_mapper.build_nanobot_config`` correctly emits
the ``channels`` block, that:

- ``websocket`` channel is enabled by default with sensible defaults
- ``feishu`` channel is **disabled** when ``FEISHU_APP_ID`` /
  ``FEISHU_APP_SECRET`` env vars are absent
- ``feishu`` channel is enabled and configured when both env vars are set
- ``channel_overrides`` lets callers (the FastAPI runtime) inject the
  gateway host/port into the websocket block
- Optional Feishu knobs (domain, group policy, reply-to-message) flow
  through from env to the produced config dict

These tests do **not** require ``nanobot-ai`` to be installed — they
exercise pure config-mapper logic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def clean_feishu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no FEISHU_* env vars leak in from the test runner."""
    for k in list(os.environ):
        if k.startswith("FEISHU_"):
            monkeypatch.delenv(k, raising=False)


def test_websocket_channel_enabled_by_default(tmp_path: Path, clean_feishu_env: None) -> None:
    """Without any channel overrides, websocket should be enabled on 127.0.0.1:8765."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    cfg = build_nanobot_config(tmp_path, {})

    assert "channels" in cfg
    channels = cfg["channels"]
    assert "websocket" in channels
    ws = channels["websocket"]
    assert ws["enabled"] is True
    assert ws["host"] == "127.0.0.1"
    assert ws["port"] == 8765
    assert ws["path"] == "/"
    assert ws["streaming"] is True
    assert ws["allowFrom"] == ["*"]


def test_feishu_channel_disabled_when_env_missing(tmp_path: Path, clean_feishu_env: None) -> None:
    """Without FEISHU_APP_ID/SECRET env vars, the channel must be disabled."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    cfg = build_nanobot_config(tmp_path, {})

    assert cfg["channels"]["feishu"]["enabled"] is False


def test_feishu_channel_enabled_when_env_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_feishu_env: None
) -> None:
    """With FEISHU_APP_ID + FEISHU_APP_SECRET, the channel must be enabled."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_xyz")

    cfg = build_nanobot_config(tmp_path, {})

    feishu = cfg["channels"]["feishu"]
    assert feishu["enabled"] is True
    assert feishu["appId"] == "cli_test_app"
    assert feishu["appSecret"] == "secret_xyz"
    assert feishu["domain"] == "feishu"
    assert feishu["groupPolicy"] == "mention"
    assert feishu["replyToMessage"] is False
    assert feishu["streaming"] is True


def test_feishu_channel_partial_env_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_feishu_env: None
) -> None:
    """Only ``APP_ID`` without ``APP_SECRET`` must NOT enable the channel."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
    # No FEISHU_APP_SECRET

    cfg = build_nanobot_config(tmp_path, {})

    assert cfg["channels"]["feishu"]["enabled"] is False


def test_feishu_channel_optional_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_feishu_env: None
) -> None:
    """Optional Feishu env vars should flow through to the config block."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    monkeypatch.setenv("FEISHU_APP_ID", "app1")
    monkeypatch.setenv("FEISHU_APP_SECRET", "sec1")
    monkeypatch.setenv("FEISHU_DOMAIN", "lark")
    monkeypatch.setenv("FEISHU_GROUP_POLICY", "open")
    monkeypatch.setenv("FEISHU_REPLY_TO_MESSAGE", "true")
    monkeypatch.setenv("FEISHU_ENCRYPT_KEY", "enc123")
    monkeypatch.setenv("FEISHU_VERIFICATION_TOKEN", "vt456")
    monkeypatch.setenv("FEISHU_ALLOW_FROM", "ou_aaa, ou_bbb")

    cfg = build_nanobot_config(tmp_path, {})

    feishu = cfg["channels"]["feishu"]
    assert feishu["enabled"] is True
    assert feishu["domain"] == "lark"
    assert feishu["groupPolicy"] == "open"
    assert feishu["replyToMessage"] is True
    assert feishu["encryptKey"] == "enc123"
    assert feishu["verificationToken"] == "vt456"
    assert feishu["allowFrom"] == ["ou_aaa", "ou_bbb"]


def test_channel_overrides_propagate_to_websocket(tmp_path: Path, clean_feishu_env: None) -> None:
    """``channel_overrides`` is the FastAPI runtime's hook to set ws host/port."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    overrides = {
        "websocket": {
            "enabled": True,
            "host": "10.0.0.5",
            "port": 19999,
        },
    }
    cfg = build_nanobot_config(tmp_path, {}, channel_overrides=overrides)

    ws = cfg["channels"]["websocket"]
    assert ws["host"] == "10.0.0.5"
    assert ws["port"] == 19999
    assert ws["enabled"] is True


def test_channel_overrides_can_disable_websocket(tmp_path: Path, clean_feishu_env: None) -> None:
    """Caller can force websocket off via ``channel_overrides``."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    overrides = {"websocket": {"enabled": False}}
    cfg = build_nanobot_config(tmp_path, {}, channel_overrides=overrides)

    assert cfg["channels"]["websocket"]["enabled"] is False


def test_mcp_server_block_removed(tmp_path: Path, clean_feishu_env: None) -> None:
    """v3.1.0: MCP servers removed from auto-generated config.

    QuantTools are registered directly on the agent's ToolRegistry via
    ``register_all_quant_tools()``. The MCP server can be started
    independently for external clients (``quantnodes serve --mcp``).
    """
    from QuantNodes.agent.config_mapper import build_nanobot_config

    cfg = build_nanobot_config(tmp_path, {})

    # MCP config no longer in auto-generated config
    assert "tools" not in cfg
    # Core keys still present
    assert "agents" in cfg
    assert "providers" in cfg
    assert "channels" in cfg


def test_config_blocks_have_known_keys(tmp_path: Path, clean_feishu_env: None) -> None:
    """Sanity: top-level structure has agents / providers / channels."""
    from QuantNodes.agent.config_mapper import build_nanobot_config

    cfg = build_nanobot_config(tmp_path, {})

    assert set(cfg.keys()) >= {"agents", "providers", "channels"}
    assert "defaults" in cfg["agents"]
    assert "workspace" in cfg["agents"]["defaults"]
    assert "model" in cfg["agents"]["defaults"]

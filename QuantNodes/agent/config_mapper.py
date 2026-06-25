# coding=utf-8
"""Translate ``.env`` QUANTNODES__* vars into HKUDS nanobot config.json.

Nanobot 0.2.1 reads a JSON config (``.agent/nanobot_config.json`` by
convention) with shape (verified against ``nanobot.config.schema.Config``)::

    {
      "agents": {"defaults": {"workspace": "<path>", "model": "...", "provider": "openai"}},
      "providers": {
        "<slot_name>": {
          "api_key": "sk-...",
          "api_base": "https://api.openai.com/v1",
          "api_type": "chat_completions" | "responses" | "auto"
        },
        ...
      },
      "channels": {
        "websocket": {"enabled": true, "host": "127.0.0.1", "port": 8765, ...},
        "feishu":    {"enabled": false, "app_id": "...", "app_secret": "...", ...}
      }
    }

Provider slots (defined in ProvidersConfig schema):
- ``openai``            — OpenAI direct
- ``anthropic``         — Anthropic direct
- ``azure_openai``      — Azure OpenAI
- ``bedrock``           — AWS Bedrock
- ``custom``            — Generic OpenAI-compatible endpoint
- ``ollama``            — Ollama local (api_base=http://localhost:11434/v1)
- ``lm_studio``         — LM Studio local
- ``vllm``              — vLLM local
- ``openrouter``        — OpenRouter aggregator
- ``deepseek``          — DeepSeek direct
- ``groq``              — Groq
- ``gemini``            — Google Gemini
- ``minimax``           — MiniMax (provider model aliases)
- ... (30+ total)

Channels (defined as ``ChannelsConfig`` extras in upstream schema):

- ``websocket`` — nanobot WebSocket channel + WebUI SPA host. Default port 18080.
- ``feishu``    — Feishu/Lark bot (WebSocket long connection, no public IP needed).
                  Requires ``FEISHU_APP_ID`` + ``FEISHU_APP_SECRET`` env vars.

Slot resolution (URL-based heuristics):
- base URL contains ``anthropic`` -> ``anthropic`` (or ``minimax_anthropic`` if MiniMax endpoint)
- base URL contains ``azure``    -> ``azure_openai``
- base URL contains ``ollama``   -> ``ollama``
- otherwise (incl. local OpenAI-compatible) -> ``custom``

``agents.defaults.provider`` is set to the resolved slot name so that the
upstream ``_match_provider`` helper finds the right provider config.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


CONFIG_FILENAME = "nanobot_config.json"


def _resolve_slot(base_url: str, api_key: str) -> tuple[str, str]:
    """Return (slot_name, api_type) inferred from URL/key hints.

    ``slot_name`` matches one of the predefined slots in ProvidersConfig
    schema (e.g. ``openai``, ``anthropic``, ``azure_openai``, ``custom``).
    ``api_type`` is the request API surface (``chat_completions``,
    ``responses``, or ``auto``).
    """
    u = (base_url or "").lower()
    if "anthropic" in u and "minimax" not in u and "minimaxi" not in u:
        # Anthropic native endpoint (api.anthropic.com or proxied clones).
        return "anthropic", "auto"
    if "azure" in u:
        return "azure_openai", "responses"
    if "ollama" in u or ":11434" in u:
        return "ollama", "auto"
    # v3.0.0: MiniMax provider (``api.minimaxi.com`` /
    # ``api.minimax.com``) exposes an OpenAI-compatible ``/v1/chat/completions``
    # endpoint, NOT the Anthropic ``/v1/messages`` surface — so the
    # ``minimax_anthropic`` slot (which forces Anthropic schema) would 404.
    # Use the ``custom`` slot with ``chat_completions`` API type instead.
    if "minimaxi" in u or "minimax" in u:
        return "custom", "chat_completions"
    if api_key and api_key.startswith("sk-ant-"):
        return "anthropic", "auto"
    # Default: OpenAI-compatible endpoint goes to ``custom`` slot
    # (which is the documented fallback for OpenAI-compatible URLs).
    return "custom", "chat_completions"


def _build_websocket_config(
    host: str = "127.0.0.1",
    port: int = 8765,
    ws_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the WebSocket channel config block.

    The WebSocket channel is the upstream nanobot gateway. It serves:

    - The WebUI SPA on port 18080 (mounted via ``ChannelManager(webui_static_dist=True)``)
    - The WebSocket endpoint at ``ws://{host}:{port}/{path}?token=...``
    - REST surface for sessions / messages / commands / sidebar state

    Configuration knobs (all map to ``nanobot.channels.websocket.WebSocketConfig``):

    - ``host``               — bind address (default 127.0.0.1, loopback only)
    - ``port``               — WS port (default 8765; the SPA lives on gateway.port)
    - ``path``               — WS upgrade path (default ``/``)
    - ``websocketRequiresToken`` — require token for handshake (default True)
    - ``streaming``          — enable streaming responses (default True)
    - ``allowFrom``          — list of allowed ``client_id`` values (default ``["*"]``)
    """
    return {
        "enabled": True,
        "host": host,
        "port": port,
        "path": "/",
        "tokenIssuePath": "/webui/token",
        "websocketRequiresToken": True,
        "allowFrom": ["*"],
        "streaming": True,
        # v3.0.0: when binding to 0.0.0.0 (all interfaces), nanobot's
        # WebSocketConfig validates that a token is set — this prevents
        # unauthenticated access from the LAN. Use NANOBOT_WS_TOKEN env
        # var if provided; otherwise auto-generate one on startup.
        **({"token": ws_token} if ws_token else {}),
    }


def _build_feishu_config(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Build the Feishu channel config block from env vars.

    Required env vars (Stage 5.4):

    - ``FEISHU_APP_ID``     — application ID from Feishu Open Platform
    - ``FEISHU_APP_SECRET`` — application secret

    Optional:

    - ``FEISHU_ENCRYPT_KEY``         — event encryption key
    - ``FEISHU_VERIFICATION_TOKEN``  — event verification token
    - ``FEISHU_DOMAIN`` (``feishu`` / ``lark``) — domain selector
    - ``FEISHU_GROUP_POLICY`` (``mention`` / ``open``) — group chat policy
    - ``FEISHU_REPLY_TO_MESSAGE``     — whether to quote the user's message

    If required env vars are missing the channel is disabled (returns
    ``{"enabled": False}``). The Feishu SDK (``lark_oapi``) is an *optional*
    runtime dep — even when this config block is enabled, the channel will
    refuse to start if the SDK isn't installed.

    Reference: ``nanobot.channels.feishu.FeishuConfig``
    """
    env = env or os.environ
    app_id = env.get("FEISHU_APP_ID", "").strip()
    app_secret = env.get("FEISHU_APP_SECRET", "").strip()
    if not (app_id and app_secret):
        return {"enabled": False}

    allow_from_raw = env.get("FEISHU_ALLOW_FROM", "").strip()
    allow_from: List[str] = (
        [s.strip() for s in allow_from_raw.split(",") if s.strip()]
        if allow_from_raw else []
    )

    return {
        "enabled": True,
        "appId": app_id,
        "appSecret": app_secret,
        "encryptKey": env.get("FEISHU_ENCRYPT_KEY", "").strip(),
        "verificationToken": env.get("FEISHU_VERIFICATION_TOKEN", "").strip(),
        "allowFrom": allow_from,
        "domain": env.get("FEISHU_DOMAIN", "feishu").strip() or "feishu",
        "groupPolicy": env.get("FEISHU_GROUP_POLICY", "mention").strip() or "mention",
        "replyToMessage": env.get("FEISHU_REPLY_TO_MESSAGE", "false").lower() in ("1", "true", "yes"),
        "streaming": True,
    }


def build_nanobot_config(
    workspace: Path,
    user_config: Optional[Dict[str, Any]] = None,
    channel_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a nanobot-compatible config dict from env + user_config.

    ``channel_overrides`` lets the caller (typically the FastAPI runtime)
    force specific channel settings — e.g. websocket host/port derived
    from ``NANOBOT_GATEWAY_HOST``/``NANOBOT_GATEWAY_PORT``.
    """
    user_config = user_config or {}
    channel_overrides = channel_overrides or {}

    api_key = user_config.get("api_key") or os.environ.get("QUANTNODES__LLM__API_KEY", "")
    base_url = user_config.get("api_base") or os.environ.get("QUANTNODES__LLM__BASE_URL", "")
    model = user_config.get("model") or os.environ.get("QUANTNODES__LLM__MODEL", "gpt-4o")

    slot, api_type = _resolve_slot(base_url, api_key)

    provider_block: Dict[str, Any] = {}
    # ``api_type`` is only allowed on the ``openai`` slot (upstream schema
    # validation); other slots always use the provider-default API surface.
    if slot == "openai":
        provider_block["apiType"] = api_type
    if api_key:
        provider_block["apiKey"] = api_key
    if base_url:
        provider_block["apiBase"] = base_url

    # ── Channels ────────────────────────────────────────────────────────
    channels: Dict[str, Any] = {}

    ws_override = channel_overrides.get("websocket") or {}
    ws_enabled = ws_override.get("enabled", True)
    if ws_enabled:
        ws_host = ws_override.get("host", "127.0.0.1")
        # v3.0.0: when binding to 0.0.0.0, nanobot requires a token for security.
        # Use NANOBOT_WS_TOKEN env var if set; otherwise auto-generate one.
        ws_token = os.environ.get("NANOBOT_WS_TOKEN", "")
        if not ws_token and ws_host in ("0.0.0.0", "::"):
            import secrets
            ws_token = secrets.token_urlsafe(32)
            logger.info("Auto-generated WebSocket token for LAN access: %s...", ws_token[:12])
        channels["websocket"] = _build_websocket_config(
            host=ws_host,
            port=int(ws_override.get("port", 8765)),
            ws_token=ws_token or None,
        )
    else:
        # Caller explicitly disabled websocket — emit the block anyway so
        # nanobot sees ``enabled: false`` and skips it deterministically.
        channels["websocket"] = {"enabled": False}

    fs_override = channel_overrides.get("feishu") or {}
    feishu_config = _build_feishu_config()
    if fs_override:
        # Allow caller to force enabled/disabled even when env vars are
        # missing (useful for tests).
        feishu_config.update(fs_override)
    channels["feishu"] = feishu_config

    # ── Top-level ───────────────────────────────────────────────────────
    # v3.0.0: MCP servers live under ``tools.mcp_servers`` in nanobot 0.2.1
    # (verified against ``nanobot.config.schema.ToolsConfig``). The upstream
    # schema rejects top-level ``mcpServers`` (extra_forbidden), so we must
    # nest it here. Field mapping:
    #   - ``transport: stdio``  →  ``type: stdio``  (upstream enum)
    #   - ``description`` removed (not in MCPServerConfig schema)
    config: Dict[str, Any] = {
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "model": model,
                "provider": slot,
            },
        },
        "providers": {
            slot: provider_block,
        },
        "channels": channels,
        "tools": {
            "mcpServers": {
                "quant": {
                    "type": "stdio",
                    # v3.0.0: use ``sys.executable`` to ensure the same
                    # Python interpreter that runs the FastAPI process
                    # (which may be ``python3.11`` on systems where
                    # ``python`` isn't on PATH) launches the MCP server.
                    "command": sys.executable,
                    "args": ["-m", "QuantNodes.mcp_server"],
                },
            },
        },
    }

    if "max_tokens" in user_config:
        config["agents"]["defaults"]["maxTokens"] = int(user_config["max_tokens"])

    return config


def write_nanobot_config(workspace: Path, config: Dict[str, Any]) -> Path:
    """Persist ``config`` as JSON inside ``workspace`` and return the path."""
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / CONFIG_FILENAME
    target.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote nanobot config: %s", target)
    return target


__all__ = [
    "build_nanobot_config",
    "write_nanobot_config",
    "CONFIG_FILENAME",
    "_build_websocket_config",
    "_build_feishu_config",
]

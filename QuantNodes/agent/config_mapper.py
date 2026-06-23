# coding=utf-8
"""Translate ``.env`` QUANTNODES__* vars into HKUDS nanobot config.json.

Nanobot 0.2.1 reads a JSON config (``.agent/nanobot_config.json`` by
convention) with shape::

    {
      "agents": {"defaults": {"workspace": "<path>", "model": "..."}},
      "llmProviders": {"<name>": {"dialect": "...", "apiKey": "...", "baseURL": "..."}},
      "mcpServers": {...},
      "cron": {...},
      "channels": {...}
    }

We map:
- ``QUANTNODES__LLM__API_KEY`` -> ``llmProviders.openai.apiKey`` (or anthropic if
  the key prefix suggests Anthropic, or azureAnthropic if base URL is Azure).
- ``QUANTNODES__LLM__BASE_URL`` -> ``llmProviders.<name>.baseURL``
- ``QUANTNODES__LLM__MODEL`` -> ``agents.defaults.model``

Dialect inference:
- base URL contains ``anthropic`` -> ``AnthropicMessages``
- base URL contains ``azure``    -> ``OpenAIResponses`` (Azure OpenAI)
- base URL contains ``ollama``  -> ``OpenResponses``
- otherwise (incl. local OpenAI-compatible) -> ``OpenAIChatCompletions``
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


CONFIG_FILENAME = "nanobot_config.json"


def _infer_dialect(base_url: str, api_key: str) -> tuple[str, str]:
    """Return (dialect, provider_name) inferred from URL/key hints."""
    u = (base_url or "").lower()
    if "anthropic" in u:
        if "azure" in u:
            return "AnthropicMessages", "azureAnthropic"
        return "AnthropicMessages", "anthropic"
    if "azure" in u:
        return "OpenAIResponses", "azureOpenAI"
    if "ollama" in u or ":11434" in u:
        return "OpenResponses", "ollama"
    if api_key and api_key.startswith("sk-ant-"):
        return "AnthropicMessages", "anthropic"
    return "OpenAIChatCompletions", "openai"


def build_nanobot_config(workspace: Path, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a nanobot-compatible config dict from env + user_config."""
    user_config = user_config or {}
    api_key = user_config.get("api_key") or os.environ.get("QUANTNODES__LLM__API_KEY", "")
    base_url = user_config.get("api_base") or os.environ.get("QUANTNODES__LLM__BASE_URL", "")
    model = user_config.get("model") or os.environ.get("QUANTNODES__LLM__MODEL", "gpt-4o")

    dialect, provider_name = _infer_dialect(base_url, api_key)

    provider_block: Dict[str, Any] = {
        "dialect": dialect,
        "apiKey": api_key,
    }
    if base_url:
        provider_block["baseURL"] = base_url

    config = {
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "model": model,
            },
        },
        "llmProviders": {
            provider_name: provider_block,
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


__all__ = ["build_nanobot_config", "write_nanobot_config", "CONFIG_FILENAME"]

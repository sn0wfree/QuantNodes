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

We map:
- ``QUANTNODES__LLM__API_KEY``  ->  ``providers.<slot>.api_key``
- ``QUANTNODES__LLM__BASE_URL`` ->  ``providers.<slot>.api_base``
- ``QUANTNODES__LLM__MODEL``    ->  ``agents.defaults.model`` (+ provider slot)

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
from pathlib import Path
from typing import Any, Dict, Optional

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
    if "anthropic" in u:
        if "minimax" in u or "MiniMax" in u:
            return "minimax_anthropic", "auto"
        return "anthropic", "auto"
    if "azure" in u:
        return "azure_openai", "responses"
    if "ollama" in u or ":11434" in u:
        return "ollama", "auto"
    if api_key and api_key.startswith("sk-ant-"):
        return "anthropic", "auto"
    # Default: OpenAI-compatible endpoint goes to ``custom`` slot
    # (which is the documented fallback for OpenAI-compatible URLs).
    return "custom", "chat_completions"


def build_nanobot_config(workspace: Path, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a nanobot-compatible config dict from env + user_config."""
    user_config = user_config or {}
    api_key = user_config.get("api_key") or os.environ.get("QUANTNODES__LLM__API_KEY", "")
    base_url = user_config.get("api_base") or os.environ.get("QUANTNODES__LLM__BASE_URL", "")
    model = user_config.get("model") or os.environ.get("QUANTNODES__LLM__MODEL", "gpt-4o")

    slot, api_type = _resolve_slot(base_url, api_key)

    provider_block: Dict[str, Any] = {}
    # ``api_type`` is only allowed on the ``openai`` slot (upstream schema
    # validation); other slots always use the provider-default API surface.
    if slot == "openai":
        provider_block["api_type"] = api_type
    if api_key:
        provider_block["api_key"] = api_key
    if base_url:
        provider_block["api_base"] = base_url

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
    }

    if "max_tokens" in user_config:
        config["agents"]["defaults"]["max_tokens"] = int(user_config["max_tokens"])

    return config


def write_nanobot_config(workspace: Path, config: Dict[str, Any]) -> Path:
    """Persist ``config`` as JSON inside ``workspace`` and return the path."""
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / CONFIG_FILENAME
    target.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote nanobot config: %s", target)
    return target


__all__ = ["build_nanobot_config", "write_nanobot_config", "CONFIG_FILENAME"]

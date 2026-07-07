"""foundation/llm/client — LLM client construction.

Supports a multi-tier config resolution:

  Tier 1: ``~/.quantnodes/llm.json`` (single canonical, M4.2 hardcoded)
  Tier 2: ``QUANTNODES__LLM__*`` env vars (override file values, M3.2)
  Tier 3: hard-coded defaults (provider-internal)

M4.2 (PR6.7): legacy ``~/.llmwikify/llmwikify.json`` is no longer auto-detected.
Run ``scripts/migrate_llmwikify_paths.py`` once to migrate legacy config.

See ``docs/refactor/REFACTOR_PLAN.md`` for M3.2 / M4.2 rationale.

Canonical imports:
    from QuantNodes.research.common.llm.client import build_llm_client, load_llm_config
    from QuantNodes.research.common.llm.client import CONFIG_PATHS, CONFIG_PATH
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── Config location (hardcoded M4.2 / PR6.7) ──────────────────────────
#
# M4.2: ``~/.quantnodes/llm.json`` is the single canonical config location.
# Legacy ``~/.llmwikify/llmwikify.json`` is no longer auto-detected by code.
# Run ``scripts/migrate_llmwikify_paths.py`` once to copy legacy config
# into the new location (or symlink it).
#
# Schema: a top-level ``"llm"`` object with keys ``enabled``, ``provider``,
# ``model``, ``base_url``, ``api_key``, ``timeout``, etc.
# ``load_llm_config`` reads the file directly and applies env-var overrides
# (Tier 3, QUANTNODES__LLM__*).

CONFIG_PATH: Path = Path.home() / ".quantnodes" / "llm.json"

# Back-compat: expose CONFIG_PATHS as a 1-tuple for callers that iterate.
CONFIG_PATHS: tuple[Path, ...] = (CONFIG_PATH,)


# ─── Env-var override mapping (Tier 3) ─────────────────────────────────
#
# M3.2: ``QUANTNODES__LLM__*`` env vars override the value loaded from
# any config file. This makes it possible to configure the LLM client
# purely via environment variables (e.g. in Docker / CI) without writing
# a JSON file.
#
# Existing callers that already use ``LLM_API_KEY`` / ``LLM_BASE_URL`` /
# ``LLM_MODEL`` / ``LLM_PROVIDER`` (the LAL resolver's namespace) keep
# working through ``QuantNodes.research.common.llm.resolver`` — those
# env vars are not duplicated here to keep this layer's surface minimal.

_ENV_OVERRIDE_KEYS: tuple[tuple[str, str], ...] = (
    ("provider", "QUANTNODES__LLM__PROVIDER"),
    ("model", "QUANTNODES__LLM__MODEL"),
    ("base_url", "QUANTNODES__LLM__BASE_URL"),
    ("api_key", "QUANTNODES__LLM__API_KEY"),
    ("enabled", "QUANTNODES__LLM__ENABLED"),
)


# ─── Provider info table (C2: replaces hardcoded "minimax" / "bearer") ─
#
# Maps provider name → (default_base_url, auth_header).
# Used by build_llm_client to fill in config gaps. Adding a new provider?
# Add a row here. (Full provider metadata lives in apps/chat/providers/
# for the chat agent; for simple client construction this is enough.)

_PROVIDER_INFO: dict[str, tuple[str, str]] = {
    # provider_name → (default_base_url, auth_header)
    "minimax": ("https://api.minimaxi.com/v1", "bearer"),
    "xiaomi": ("https://api.xiaomi.com/v1", "bearer"),
    "openai": ("https://api.openai.com/v1", "bearer"),
    "anthropic": ("https://api.anthropic.com/v1", "x-api-key"),
}


# ─── Config loading ──────────────────────────────────────────────────


def _load_single_path(path: Path) -> dict[str, Any] | None:
    """Load the ``[llm]`` section from a single config file.

    Returns:
        - ``None`` if the file does not exist (so caller can try the next
          path in ``CONFIG_PATHS``).
        - ``{}`` if the file exists but has no ``"llm"`` top-level key
          (deliberate empty config — caller treats this as "found").
        - The ``llm`` dict otherwise.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("%s: top-level is not a dict (%s)", path, type(data).__name__)
        return {}
    llm_section = data.get("llm")
    if not isinstance(llm_section, dict):
        return {}
    return llm_section


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply ``QUANTNODES__LLM__*`` env var overrides (Tier 3).

    Returns a new dict (does not mutate input). Empty / unset env vars
    are ignored — config values are preserved.
    """
    result = dict(config)
    for key, env_key in _ENV_OVERRIDE_KEYS:
        env_val = os.environ.get(env_key)
        if env_val:  # ignore empty / unset
            # Coerce boolean-ish strings for the "enabled" key.
            if key == "enabled":
                result[key] = env_val.strip().lower() in ("1", "true", "yes", "on")
            else:
                result[key] = env_val
    return result


def load_llm_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load LLM config from the canonical path (M4.2 hardcode).

    Resolution order (first match wins):
      1. ``config_path`` argument (if provided) — overrides everything;
         used by tests and ``build_llm_client(config=...)`` callers.
      2. ``CONFIG_PATH`` = ``~/.quantnodes/llm.json`` (single canonical)
      3. Returns ``{}`` if not found.

    Legacy ``~/.llmwikify/llmwikify.json`` is NOT consulted by code.
    Run ``scripts/migrate_llmwikify_paths.py`` once to migrate legacy config.

    A path is "found" if the file exists (even if its ``[llm]`` section
    is empty — that means the user deliberately created an empty config).

    Args:
        config_path: Override config file path (mainly for tests).
                     When provided, only this path is consulted.

    Returns:
        The ``llm`` section as a dict, or ``{}`` if no config found.
    """
    if config_path is not None:
        result = _load_single_path(config_path)
        if result is None:
            logger.warning("LLM config not found at %s", config_path)
            return {}
        logger.debug("[llm_client] config loaded from explicit %s", config_path)
        return result

    for path in CONFIG_PATHS:
        result = _load_single_path(path)
        if result is not None:
            logger.info("[llm_client] config loaded from %s", path)
            return result

    logger.warning(
        "LLM config not found in any of: %s",
        [str(p) for p in CONFIG_PATHS],
    )
    return {}


# ─── Client construction ─────────────────────────────────────────────


def _resolve_provider_info(provider: str) -> tuple[str, str]:
    """Look up (default_base_url, auth_header) for a provider.

    C2: replaces the old hardcoded `auth_header = "bearer" if ... else "bearer"`
    no-op with a real lookup.

    Args:
        provider: Provider name from config (e.g., "minimax", "xiaomi").

    Returns:
        (default_base_url, auth_header) tuple. Falls back to
        (generic OpenAI URL, "bearer") if the provider is unknown.
        Logs a warning for unknown providers (don't silently default).
    """
    if provider in _PROVIDER_INFO:
        return _PROVIDER_INFO[provider]
    logger.warning(
        "[llm_client] unknown provider %r; falling back to OpenAI defaults. "
        "Add an entry to _PROVIDER_INFO if you want custom base_url/auth_header.",
        provider,
    )
    return _PROVIDER_INFO["openai"]


def build_llm_client(
    config: dict[str, Any] | None = None,
    model: str | None = None,
    config_path: Path | None = None,
) -> Any:
    """Build a ``StreamableLLMClient`` from user config.

    Resolution order:
      1. Explicit ``config`` argument (if provided)
      2. ``load_llm_config(config_path=config_path)`` — Tier 1 → Tier 2 file lookup
      3. ``QUANTNODES__LLM__*`` env overrides applied to (1) or (2)
      4. Provider-internal defaults from ``_PROVIDER_INFO`` (Tier 4)

    Args:
        config: Pre-loaded config dict. If None, loads from disk.
        model: Override model name (default: config's ``model`` field).
        config_path: Override config file path (mainly for tests).

    Returns:
        Configured ``StreamableLLMClient`` instance.

    Raises:
        RuntimeError: If LLM is disabled in config, provider is missing,
            or api_key is not configured.
    """
    from .streamable import StreamableLLMClient

    if config is None:
        config = load_llm_config(config_path=config_path)

    # M3.2: Tier 3 env override (QUANTNODES__LLM__*)
    config = _apply_env_overrides(config)

    if not config.get("enabled"):
        raise RuntimeError(
            f"LLM is disabled in {config_path or CONFIG_PATHS}. "
            "Set llm.enabled=true to enable."
        )

    # C2: provider is REQUIRED (was hardcoded to "minimax" in pre-C2).
    # If the config has no provider, fail loudly rather than silently
    # fall back to minimax.
    provider = config.get("provider")
    if not provider:
        raise RuntimeError(
            f"Missing 'provider' in {config_path or CONFIG_PATHS}. "
            f"Set llm.provider to one of: {', '.join(_PROVIDER_INFO.keys())}"
        )

    # C2: auth_header from provider-info table (was no-op hardcoded "bearer").
    default_base_url, auth_header = _resolve_provider_info(provider)
    base_url = config.get("base_url") or default_base_url
    chosen_model = model or config.get("model") or "MiniMax-M2.7"
    api_key = config.get("api_key", "")
    timeout = config.get("timeout", 600)

    if not api_key:
        raise RuntimeError(
            f"Missing api_key in {config_path or CONFIG_PATHS}. Set llm.api_key first."
        )

    logger.info(
        "[llm_client] provider=%s model=%s base_url=%s timeout=%s",
        provider,
        chosen_model,
        base_url,
        timeout,
    )
    return StreamableLLMClient(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=chosen_model,
        auth_header=auth_header,
        reasoning_split=True,
        request_timeout_seconds=float(timeout),
    )


__all__ = [
    "CONFIG_PATHS",
    "CONFIG_PATH",
    "load_llm_config",
    "build_llm_client",
    "_PROVIDER_INFO",
    "_ENV_OVERRIDE_KEYS",
]
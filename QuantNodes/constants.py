# coding=utf-8
"""QuantNodes infrastructure defaults — single source of truth.

Import from here, NOT from cli._helpers or api.config.
This module has ZERO external dependencies (no numpy, no fastapi, no nanobot).
"""

# ── Network ──────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 19380
DEFAULT_GATEWAY_PORT = 18090
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_WEBSOCKET_PORT = 8765  # internal WS channel (not user-facing)

# ── LLM Provider ────────────────────────────────────────
DEFAULT_LLM_MODEL = "gpt-4"
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"

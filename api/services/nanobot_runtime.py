# coding=utf-8
"""Single-process nanobot runtime wrapper.

v3.0.0 Stage 5.3 architecture:

```
QuantNodes 进程（单一 Python 进程）
├─ uvicorn 8000           ← FastAPI + 量化 REST API
├─ nanobot 18090          ← WebSocket + WebUI SPA（同一进程内启动）
│   ├─ WS /               ← chat 通信
│   ├─ /api/sessions      ← WebUI 后端 REST
│   └─ / (index.html)     ← WebUI SPA
├─ QuantDream             ← asyncio.Task
├─ CronService            ← asyncio.Task
└─ AgentLoop              ← asyncio.Task
```

This module owns the lifecycle of the in-process nanobot runtime:

1. :func:`start` — called from FastAPI lifespan; manually wires
   ``AgentLoop + ChannelManager + CronService`` and schedules them as
   ``asyncio.create_task`` (NOT ``asyncio.run``).
2. :func:`stop` — called on shutdown; cancels background tasks in
   correct order (cron → agent → channels → mcp).
3. :func:`status` — returns a serializable snapshot for
   ``GET /api/agent/status`` and the frontend.

If ``nanobot-ai`` is not installed (NANOBOT_AVAILABLE=False), all
operations degrade gracefully:

- :func:`start` is a no-op (returns immediately, sets state to
  ``"unavailable"``)
- :func:`stop` is a no-op
- :func:`status` returns ``{available: false, hint: "pip install
  'quantnodes[agent]'"}`` for the frontend to display an install page
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Public configuration
# ----------------------------------------------------------------------------

# Re-export from QuantNodes.constants (single source of truth)
from QuantNodes.constants import (  # noqa: E402
    DEFAULT_HOST as DEFAULT_GATEWAY_HOST,
    DEFAULT_GATEWAY_PORT,
)

DEFAULT_WORKSPACE = ".agent"
DEFAULT_NANOBOT_CONFIG = "nanobot_config.json"


# ----------------------------------------------------------------------------
# Runtime state
# ----------------------------------------------------------------------------

@dataclass
class RuntimeState:
    """Snapshot of the nanobot runtime lifecycle state."""

    state: str = "uninitialized"  # uninitialized | starting | running | stopping | stopped | error | unavailable
    available: bool = False
    hint: Optional[str] = None
    error: Optional[str] = None
    gateway_host: str = DEFAULT_GATEWAY_HOST
    gateway_port: int = DEFAULT_GATEWAY_PORT
    workspace: str = DEFAULT_WORKSPACE
    components: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "available": self.available,
            "hint": self.hint,
            "error": self.error,
            "gateway_host": self.gateway_host,
            "gateway_port": self.gateway_port,
            "workspace": self.workspace,
            "components": self.components,
        }


def _is_nanobot_available() -> bool:
    """Probe ``NANOBOT_AVAILABLE`` at call time (not import time).

    Reading the flag lazily lets test code toggle it via
    ``unittest.mock.patch.object`` between test cases without re-importing
    the whole ``QuantNodes.agent`` module.
    """
    try:
        from QuantNodes.agent import NANOBOT_AVAILABLE
        return bool(NANOBOT_AVAILABLE)
    except ImportError:  # pragma: no cover - defensive
        return False


class NanobotRuntime:
    """Manages the in-process nanobot agent runtime lifecycle.

    This is a singleton; the FastAPI lifespan creates one instance per
    process and stores it on ``app.state.nanobot_runtime``. The HTTP
    router (``/api/agent/*``) reads from this instance.
    """

    def __init__(
        self,
        workspace: str = DEFAULT_WORKSPACE,
        gateway_host: str = DEFAULT_GATEWAY_HOST,
        gateway_port: int = DEFAULT_GATEWAY_PORT,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.state = RuntimeState(
            available=_is_nanobot_available(),
            gateway_host=gateway_host,
            gateway_port=gateway_port,
            workspace=str(self.workspace),
            components={},
        )

        # Component handles (None until start() succeeds)
        self._bus: Any = None
        self._session_manager: Any = None
        self._cron: Any = None
        self._agent: Any = None
        self._channels: Any = None

        # Background asyncio.Task handles
        self._tasks: List[asyncio.Task] = []

    # -- Public API --------------------------------------------------------

    async def start(self) -> None:
        """Start the nanobot agent runtime in this process.

        Steps (all wrapped in try/except so a partial failure doesn't
        crash the FastAPI app):

        1. Build nanobot config from .env via config_mapper
        2. Create MessageBus, SessionManager, CronService
        3. Create AgentLoop via ``AgentLoop.from_config(cfg, bus, ...)``
        4. Register 14 quant tools on the agent's ToolRegistry
        5. Create ChannelManager (with webui_static_dist=True)
        6. Schedule ``cron.start()``, ``agent.run()``,
           ``channels.start_all()`` as ``asyncio.create_task``s

        On any failure, the state transitions to ``"error"`` and a
        ``hint`` is set; the FastAPI app keeps running normally with
        ``/api/agent/*`` returning a 503 + install/run hint.
        """
        # Re-evaluate the flag at call time so tests can toggle it.
        if not _is_nanobot_available():
            self.state.state = "unavailable"
            self.state.available = False
            self.state.hint = (
                "nanobot-ai not installed. Run:  "
                "pip install 'quantnodes[agent]'  (or  'quantnodes[all]')"
            )
            logger.info("nanobot-ai not installed — runtime unavailable")
            return

        self.state.state = "starting"
        self.state.available = True
        try:
            await self._build_components()
            await self._schedule_tasks()
            self.state.state = "running"
            self.state.components = {
                "bus": self._bus is not None,
                "session_manager": self._session_manager is not None,
                "cron": self._cron is not None,
                "agent": self._agent is not None,
                "channels": self._channels is not None,
            }
            logger.info(
                "nanobot runtime started (workspace=%s, gateway=%s:%d)",
                self.workspace, self.gateway_host, self.gateway_port,
            )
            # Log the gateway tokenIssueSecret so users can access the WebUI
            try:
                _cfg = json.loads(self.config_path.read_text())
                _token = _cfg.get("channels", {}).get("websocket", {}).get("token", "")
                if _token:
                    logger.info("Gateway tokenIssueSecret: %s", _token)
            except Exception:
                pass
        except ImportError as e:
            # ``NanobotNotInstalled`` is a subclass of ImportError. We
            # surface its message as a hint for the frontend.
            try:
                from QuantNodes.agent import NanobotNotInstalled as _NNI
                if isinstance(e, _NNI):
                    self.state.state = "unavailable"
                    self.state.available = False
                    self.state.hint = str(e)
                    logger.warning("nanobot runtime unavailable: %s", e)
                    return
            except ImportError:
                pass
            self.state.state = "error"
            self.state.error = f"{type(e).__name__}: {e}"
            self.state.hint = (
                "nanobot runtime failed to start. Check .env QUANTNODES__LLM__* "
                "vars and 'pip install quantnodes[agent]'."
            )
            logger.exception("nanobot runtime start failed (ImportError)")
        except Exception as e:  # pragma: no cover - defensive
            self.state.state = "error"
            self.state.error = f"{type(e).__name__}: {e}"
            self.state.hint = (
                "nanobot runtime failed to start. Check .env QUANTNODES__LLM__* "
                "vars and 'pip install quantnodes[agent]'."
            )
            logger.exception("nanobot runtime start failed")

    async def stop(self) -> None:
        """Stop the nanobot agent runtime.

        Order: cron → agent → channels → mcp. Each step is best-effort:
        if it raises, we log and move on (we still want to release
        resources on the way out).
        """
        if not self._tasks and self._agent is None:
            self.state.state = "stopped"
            return

        self.state.state = "stopping"
        try:
            if self._cron is not None:
                try:
                    self._cron.stop()
                except Exception:  # pragma: no cover
                    logger.exception("cron.stop() failed")

            if self._agent is not None:
                try:
                    self._agent.stop()
                except Exception:  # pragma: no cover
                    logger.exception("agent.stop() failed")

            if self._channels is not None:
                try:
                    await self._channels.stop_all()
                except Exception:  # pragma: no cover
                    logger.exception("channels.stop_all() failed")

            if self._agent is not None:
                try:
                    await self._agent.close_mcp()
                except Exception:  # pragma: no cover
                    logger.exception("agent.close_mcp() failed")

            for task in self._tasks:
                if not task.done():
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            self._tasks = []
            self._bus = self._session_manager = self._cron = None
            self._agent = self._channels = None
            self.state.state = "stopped"
            self.state.components = {}
            logger.info("nanobot runtime stopped")

    def status(self) -> Dict[str, Any]:
        """Return a serializable snapshot for HTTP and the frontend."""
        return self.state.to_dict()

    @property
    def agent(self) -> Any:
        """Underlying AgentLoop (or None if not running)."""
        return self._agent

    @property
    def bus(self) -> Any:
        """Underlying MessageBus (or None if not running)."""
        return self._bus

    @property
    def channels(self) -> Any:
        """Underlying ChannelManager (or None if not running)."""
        return self._channels

    @property
    def session_manager(self) -> Any:
        """Underlying SessionManager (or None if not running)."""
        return self._session_manager

    # -- Internal helpers --------------------------------------------------

    async def _build_components(self) -> None:
        """Construct all nanobot components (no async I/O yet)."""
        # Late imports: only when nanobot is actually available. This keeps
        # import-time fast and prevents circular dependencies.
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus
        from nanobot.channels.manager import ChannelManager
        from nanobot.config.schema import Config
        from nanobot.cron.service import CronService
        from nanobot.providers.factory import build_provider_snapshot
        from nanobot.session.manager import SessionManager

        from QuantNodes.agent.config_mapper import (
            build_nanobot_config,
            write_nanobot_config,
        )
        from QuantNodes.agent.tools import register_all_quant_tools

        # 1. Ensure workspace exists with config.json.
        # Channel overrides let the runtime inject NANOBOT_GATEWAY_HOST/PORT
        # so the websocket channel binds to the correct address.
        self.workspace.mkdir(parents=True, exist_ok=True)
        channel_overrides = {
            "websocket": {
                "enabled": True,
                "host": self.gateway_host,
                # The websocket channel lives on the gateway port (default 18090).
                # It serves both the WebSocket API and the SPA static files.
                "port": self.gateway_port,
            },
        }
        settings = build_nanobot_config(
            self.workspace, {}, channel_overrides=channel_overrides
        )
        self.config_path = write_nanobot_config(self.workspace, settings)

        # 2. Build runtime Config from the JSON we just wrote
        cfg = Config.model_validate(json.loads(self.config_path.read_text()))
        # v3.0.0: ``workspace_path`` is a derived @property on Config
        # (read from ``agents.defaults.workspace``). Don't try to set it
        # directly — it's read-only. The workspace was already written into
        # the JSON by config_mapper.py at line above, so this is enough.
        cfg.gateway.host = self.gateway_host
        cfg.gateway.port = self.gateway_port
        cfg.agents.defaults.workspace = str(self.workspace)
        # v3.0.0: WebSocket channel port must also be updated to match
        # gateway_port (in case env NANOBOT_GATEWAY_PORT differs from the
        # default 18090 written by config_mapper). ChannelsConfig uses
        # ``extra="allow"``, so ``cfg.channels.websocket`` is a plain dict
        # (each channel parses its own config in __init__).
        if isinstance(cfg.channels.websocket, dict):
            cfg.channels.websocket["port"] = self.gateway_port
            cfg.channels.websocket["host"] = self.gateway_host

        # 3. Build components
        self._bus = MessageBus()
        self._session_manager = SessionManager(self.workspace)
        self._cron = CronService(self.workspace / "cron" / "jobs.json")

        try:
            provider_snapshot = build_provider_snapshot(cfg)
        except Exception as exc:  # pragma: no cover - config issue
            raise RuntimeError(
                f"Failed to build provider snapshot (check QUANTNODES__LLM__* env): {exc}"
            ) from exc

        self._agent = AgentLoop.from_config(
            cfg,
            self._bus,
            provider=provider_snapshot.provider,
            model=provider_snapshot.model,
            context_window_tokens=provider_snapshot.context_window_tokens,
            cron_service=self._cron,
            session_manager=self._session_manager,
        )

        # 4. Register our 14 quant tools on the agent's ToolRegistry
        register_all_quant_tools(self._agent.tools, workspace=self.workspace)

        # 4b. Register quant-domain cron jobs (daily recap / weekly review /
        # monthly strategy-pool). These are idempotent: ``register_system_job``
        # replaces any job with the same id on restart. The cron service was
        # constructed in step 3 (above) so we can pass it here.
        try:
            from QuantNodes.agent.cron_jobs import register_quant_cron_jobs
            registered_ids = register_quant_cron_jobs(self._cron)
            logger.info(
                "Registered %d quant cron jobs: %s",
                len(registered_ids),
                ", ".join(registered_ids),
            )
        except ImportError as e:
            # nanobot cron types not available — graceful skip. This branch
            # is exercised when nanobot-ai is partially installed or when
            # an older nanobot version is in use.
            logger.warning("Could not register quant cron jobs: %s", e)

        # 5. ChannelManager with webui_static_dist=True (serves SPA + WS API).
        # Channels that were enabled in the JSON config (websocket by
        # default; feishu when FEISHU_APP_ID/SECRET are set) will start
        # their background tasks via channels.start_all().
        self._channels = ChannelManager(
            cfg,
            self._bus,
            session_manager=self._session_manager,
            webui_static_dist=True,
            webui_runtime_surface="browser",
            webui_runtime_model_name=lambda: getattr(self._agent, "model", None),
        )

    async def _schedule_tasks(self) -> None:
        """Schedule background tasks for cron, agent, channels.

        These run concurrently with the FastAPI event loop. Each task
        is named so it's easy to identify in asyncio debug output.
        """
        self._tasks = [
            asyncio.create_task(self._cron.start(), name="nanobot-cron"),
            asyncio.create_task(self._agent.run(), name="nanobot-agent"),
            asyncio.create_task(self._channels.start_all(), name="nanobot-channels"),
        ]


# ----------------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------------

_runtime: Optional[NanobotRuntime] = None


def get_runtime() -> Optional[NanobotRuntime]:
    """Return the active NanobotRuntime singleton, or None if not started.

    The singleton is created in :func:`init_runtime` (called from the
    FastAPI lifespan startup phase) and cleared in :func:`shutdown_runtime`
    (called from lifespan shutdown).
    """
    return _runtime


def init_runtime(
    workspace: str = DEFAULT_WORKSPACE,
    gateway_host: str = DEFAULT_GATEWAY_HOST,
    gateway_port: int = DEFAULT_GATEWAY_PORT,
) -> NanobotRuntime:
    """Create the runtime singleton (does not start it)."""
    global _runtime
    if _runtime is not None:
        return _runtime
    port = int(os.environ.get("NANOBOT_GATEWAY_PORT", gateway_port))
    host = os.environ.get("NANOBOT_GATEWAY_HOST", gateway_host)
    ws = os.environ.get("NANOBOT_WORKSPACE", workspace)
    _runtime = NanobotRuntime(workspace=ws, gateway_host=host, gateway_port=port)
    return _runtime


async def shutdown_runtime() -> None:
    """Stop and clear the runtime singleton."""
    global _runtime
    if _runtime is None:
        return
    try:
        await _runtime.stop()
    finally:
        _runtime = None

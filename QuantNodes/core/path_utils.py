"""Path utilities — common pattern extraction.

Consolidates:
  1. `Path(x).expanduser()` and the env-var override variant
  2. `Path(x).mkdir(parents=True, exist_ok=True)` (38 sites)
  3. `path.parent.mkdir(parents=True, exist_ok=True)` (parent-for-file pattern)

Used by:
  - QuantNodes/core/trajectory/pool.py
  - QuantNodes/core/monitoring/{collector,dashboard}.py
  - QuantNodes/core/feedback/dataclass.py
  - QuantNodes/core/knowledge/knowledge_base.py
  - QuantNodes/core/visualization/report.py
  - QuantNodes/core/parallel/worker_process.py
  - QuantNodes/core/quality_gate/zoo.py
  - QuantNodes/agent/tools/{file_ops,task}.py
  - QuantNodes/agent/cli/main.py
  - QuantNodes/agent/core/memory.py
  - QuantNodes/agent/session/manager.py
  - QuantNodes/cache_node/{cache_store,metadata}.py
  - QuantNodes/monitor/{version/version_manager.py,scheduler/scheduler.py,storage/repository.py}
  - QuantNodes/cli/_helpers.py + commands/{run,init}.py
  - QuantNodes/backtest/config_runner.py
  - QuantNodes/research/factor_test/{nodes,ifind_db,e2e}/...py

P-1 priority is preserved: env_var > expanduser(default).
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["resolve_path", "ensure_dir", "ensure_parent"]


def resolve_path(default: str, env_var: str | None = None) -> Path:
    """Resolve a path with priority: env_var > expanduser(default).

    Args:
        default: fallback path string (may contain ~).
        env_var: optional environment variable name. If set and non-empty,
            its value (after expanduser) wins.

    Returns:
        Path (always expanded, never None).
    """
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            return Path(env_val).expanduser()
    return Path(default).expanduser()


def ensure_dir(path: str | Path) -> Path:
    """Create directory at `path` if missing. Idempotent. Returns Path.

    Replaces: `Path(x).mkdir(parents=True, exist_ok=True)`.
    Also replaces bare `os.makedirs(x, exist_ok=True)`.
    """
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    """Ensure the *parent* of `path` exists (i.e. prepare to write a file).

    Replaces: `Path(x).parent.mkdir(parents=True, exist_ok=True)`.
    Returns the *original* `path` (not the parent), so callers can write to it.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

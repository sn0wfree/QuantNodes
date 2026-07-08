# coding=utf-8
"""Helper functions and constants used by QuantNodes CLI commands."""

import argparse
import functools
import os
import signal
import socket
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

from QuantNodes.research.wiki.init_factor_wiki import init_factor_wiki
from QuantNodes.core.path_utils import ensure_dir

PROG_NAME = "quantnodes"

# Re-export from QuantNodes.constants (single source of truth)
from QuantNodes.constants import (  # noqa: E402, F401
    DEFAULT_API_PORT,
    DEFAULT_FRONTEND_PORT,
    DEFAULT_HOST,
    DEFAULT_GATEWAY_PORT,
)
PIDFILE_NAME = ".quantnodes.pid"


def is_initialized() -> bool:
    """Check if current directory is already initialized."""
    return Path(".env").exists() or Path("conn.ini").exists() or Path("wiki/index.md").exists()


def get_project_root() -> Path:
    """Get the QuantNodes project root directory.

    Walks up from this file to find the repo root (containing pyproject.toml).
    Works from any working directory.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def create_directory_structure():
    """Create necessary directories.

    Two output roots are used by different subsystems:
    - outputs/    backtest engine results (equity / signals / trades parquets)
    - output/     factor_test pipeline artefacts (per-node parquets + final report)
    """
    dirs = [
        "data",
        ".agent/memory",
        ".agent/dream",
        "output",
        "outputs",
        "logs",
    ]

    for d in dirs:
        ensure_dir(Path(d))
        print(f"  ✓ 创建目录: {d}/")


def init_llmwikify_wiki(force: bool = False) -> bool:
    """Initialize QuantNodes wiki structure."""
    print("\n  初始化 QuantNodes Wiki...")

    wiki_exists = Path("wiki/index.md").exists()

    if wiki_exists and not force:
        print("  ⏭️  Wiki 已存在，跳过初始化")
        try:
            init_factor_wiki("wiki", force=False)
            print("  ✓ Wiki 配置已更新")
            return True
        except Exception as e:
            print(f"  ⚠ 配置更新失败: {e}")
            return False

    if wiki_exists and force:
        print("  🔄 强制重新初始化 Wiki")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "llmwikify", "init"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ llmwikify 初始化失败: {result.stderr}")
            return False
        print("  ✓ llmwikify 基础结构创建完成")
    except Exception as e:
        print(f"  ⚠ llmwikify 初始化失败: {e}")
        return False

    try:
        init_factor_wiki("wiki", force=True)
        print("  ✓ QuantNodes 专用配置写入完成")
        return True
    except Exception as e:
        print(f"  ⚠ 配置写入失败: {e}")
        return False


def get_input_with_default(prompt: str, default: str, required: bool = False) -> str:
    """Get user input with a default value."""
    while True:
        try:
            response = input(f"{prompt} [{default}]: ").strip()
            if not response:
                return default
            return response
        except EOFError:
            if required:
                continue
            return default


# ============================================================================
# Phase I1: argparse builders for repeated argument groups (2026-06-20)
# ============================================================================
# Each builder attaches a common set of CLI args to the given parser.
# Used by _build_parser() to dedup 6+ repeated patterns across factor-* cmds:
#   --pool-dir (6 sites), --top (2), --metric (2), --title (2),
#   --ancestor-depth/--descendant-depth (2),
#   --min-ipo-days/--min-group-size/--groups (2).


def add_pool_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Attach --pool-dir <path> (required)."""
    parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")


def add_top_arg(parser: argparse.ArgumentParser, default: int = 5) -> None:
    """Attach --top <N>."""
    parser.add_argument("--top", type=int, default=default, help=f"Top-N (默认 {default})")


def add_metric_arg(parser: argparse.ArgumentParser, default: str = "sharpe") -> None:
    """Attach --metric <name>."""
    parser.add_argument("--metric", default=default, help=f"排序指标 (默认 {default})")


def add_title_arg(parser: argparse.ArgumentParser) -> None:
    """Attach --title <text>."""
    parser.add_argument("--title", default=None, help="报告标题")


def add_output_arg(parser: argparse.ArgumentParser, default: str | None = None) -> None:
    """Attach --output <path>."""
    parser.add_argument("--output", default=default, help="HTML 输出路径")


def add_lineage_depth_args(parser: argparse.ArgumentParser) -> None:
    """Attach --ancestor-depth / --descendant-depth (default 2 each)."""
    parser.add_argument("--ancestor-depth", type=int, default=2, help="祖先深度 (默认 2)")
    parser.add_argument("--descendant-depth", type=int, default=2, help="后裔深度 (默认 2)")


def add_cli_overrides(parser: argparse.ArgumentParser) -> None:
    """Attach --min-ipo-days / --min-group-size / --groups (CLI overrides)."""
    parser.add_argument("--min-ipo-days", type=int, default=None,
                        help="剔除上市不足 N 日新股 (覆盖 config 默认值 360)")
    parser.add_argument("--min-group-size", type=int, default=None,
                        help="计算 IC 最少样本数 (覆盖 config 默认值 5)")
    parser.add_argument("--groups", type=int, default=None,
                        help="分组分析分组数 (覆盖 config 默认值 5)")


# ============================================================================
# Phase I2: cli_safe_run decorator (2026-06-20)
# ============================================================================

def cli_safe_run(func):
    """Decorator: catch Exception, print 错误 + return 1.

    Replaces the manual ``try: ... except Exception as e: print(f"错误: {e}"); return 1``
    wrapper that was repeated 7+ times in factor.py and evolve.py.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"错误: {e}")
            return 1
    return wrapper


# ============================================================================
# Phase I3: confirm_section helper (2026-06-20)
# ============================================================================

def confirm_section(
    name: str,
    fields: Sequence[tuple[str, str]],
    default: bool = False,
) -> dict | None:
    """Prompt "是否配置 {name}" → on Yes, ask each (label, default) tuple.

    Used by init.py for ClickHouse / MySQL config blocks. Each block
    was 7 lines of boilerplate (print header + 5 get_input_with_default calls);
    confirm_section collapses to a single call with a field list.

    Args:
        name: section name (e.g. "ClickHouse").
        fields: sequence of (label, default) pairs to prompt for.
        default: default for the yes/no confirm.

    Returns:
        Dict of {label: value} on confirmation, or None if declined.
    """
    if not get_yes_no(f"是否配置 {name}", default=default):
        return None
    print(f"\n  {name} 配置:")
    return {label: get_input_with_default(f"    {label}", default) for label, default in fields}


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    while True:
        try:
            response = input(f"{prompt} ({default_str}): ").strip().lower()
            if not response:
                return default
            if response in ("y", "yes"):
                return True
            if response in ("n", "no"):
                return False
            print("  请输入 y 或 n")
        except EOFError:
            return default


def get_model_choice() -> str:
    """Get model choice from user."""
    models = {
        "1": "gpt-4",
        "2": "gpt-3.5-turbo",
        "3": "gpt-4-turbo",
        "4": "custom",
    }

    print("  选择模型:")
    print("    1) gpt-4 (默认)")
    print("    2) gpt-3.5-turbo")
    print("    3) gpt-4-turbo")
    print("    4) 自定义")

    while True:
        try:
            choice = input("  选择 [1]: ").strip()
            if not choice:
                return "gpt-4"
            if choice in models:
                return models[choice]
            print("  无效选择，请输入 1-4")
        except EOFError:
            return "gpt-4"


def write_env_file(api_key: str, base_url: str, model: str, duckdb_path: str,
                   clickhouse_config: dict, mysql_config: dict) -> None:
    """Write .env configuration file."""
    env_content = f"""# LLM 配置
QUANTNODES__LLM__API_KEY={api_key}
QUANTNODES__LLM__BASE_URL={base_url}

# 模型配置
QUANTNODES__LLM__MODEL={model}

# DuckDB (默认本地文件)
QUANTNODES__DUCKDB__PATH={duckdb_path}

# 可选数据源
"""

    if clickhouse_config:
        env_content += f"""QUANTNODES__CLICKHOUSE__HOST={clickhouse_config['host']}
QUANTNODES__CLICKHOUSE__PORT={clickhouse_config['port']}
QUANTNODES__CLICKHOUSE__USER={clickhouse_config['user']}
QUANTNODES__CLICKHOUSE__PASSWORD={clickhouse_config['passwd']}
QUANTNODES__CLICKHOUSE__DATABASE={clickhouse_config['db']}

"""

    if mysql_config:
        env_content += f"""QUANTNODES__MYSQL__HOST={mysql_config['host']}
QUANTNODES__MYSQL__PORT={mysql_config['port']}
QUANTNODES__MYSQL__USER={mysql_config['user']}
QUANTNODES__MYSQL__PASSWORD={mysql_config['passwd']}
QUANTNODES__MYSQL__DATABASE={mysql_config['db']}

"""

    env_content += """# 缓存配置 (H18: 与 IFindFetcher.DEFAULT_CACHE_TTL_S=604800 对齐)
QUANTNODES__CACHE_ENABLED=true
QUANTNODES__CACHE_TTL=604800
"""

    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    print("  ✓ 创建文件: .env")


def write_conn_ini(duckdb_path: str, clickhouse_config: dict, mysql_config: dict) -> None:
    """Write conn.ini configuration file."""
    conn_content = f"""[DuckDB]
path = {duckdb_path}
read_only = False

"""

    if clickhouse_config:
        conn_content += f"""[ClickHouse]
host = {clickhouse_config['host']}
port = {clickhouse_config['port']}
user = {clickhouse_config['user']}
passwd = {clickhouse_config['passwd']}
db = {clickhouse_config['db']}

"""

    if mysql_config:
        conn_content += f"""[MySQL]
host = {mysql_config['host']}
port = {mysql_config['port']}
user = {mysql_config['user']}
passwd = {mysql_config['passwd']}
db = {mysql_config['db']}
"""

    with open("conn.ini", "w", encoding="utf-8") as f:
        f.write(conn_content)
    print("  ✓ 创建文件: conn.ini")


def install_talib() -> bool:
    """Install TA-Lib (optional)."""
    print("\n  安装 TA-Lib...")
    try:
        result = subprocess.run(
            ["pip", "install", "TA-Lib>=0.6.0"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ TA-Lib 安装完成")
            return True
        else:
            print("  ⚠ TA-Lib 安装失败 (可能需要系统库)，跳过继续...")
            return False
    except Exception as e:
        print(f"  ⚠ TA-Lib 安装失败: {e}，跳过继续...")
        return False


# ============================================================================
# v3.0.0 Stage 7 — serve/stop/status lifecycle helpers
# ============================================================================
#
# These helpers back the new ``quantnodes serve`` / ``stop`` / ``status`` /
# ``agent`` subcommands. Design goals:
#   - Reuse from both CLI (foreground or daemon) and FastAPI startup
#     (api/main.py uses ``load_env_file`` to populate os.environ before
#     reading QUANTNODES__LLM__*).
#   - Avoid new dependencies: use only stdlib + httpx (already required).
#   - Keep pidfile simple: one file at project root, single int PID.


def load_env_file(env_path: Optional[Path] = None) -> bool:
    """Load ``.env`` into ``os.environ`` (no override).

    Returns True if the file existed and was loaded, False otherwise.
    Used by both ``quantnodes serve`` (CLI) and ``api/main.py`` (FastAPI
    lifespan startup) so a single source of truth populates
    ``QUANTNODES__LLM__*`` / ``NANOBOT_GATEWAY_*`` / ``FEISHU_*`` for
    ``os.environ.get(...)`` consumers (config_mapper.py, nanobot_runtime.py).

    Idempotent: safe to call multiple times. ``override=False`` so an
    explicit shell env var always wins (e.g. ``NANOBOT_GATEWAY_PORT=18090
    quantnodes serve``).
    """
    env_path = Path(env_path) if env_path else Path.cwd() / ".env"
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return True
    except ImportError:
        # python-dotenv is optional. Without it the user must source .env
        # manually (or use shell exports).
        return False


def write_pidfile(pid: int, path: Optional[Path] = None) -> Path:
    """Write ``pid`` to ``.quantnodes.pid`` (project root by default)."""
    path = Path(path) if path else Path.cwd() / PIDFILE_NAME
    path.write_text(str(pid), encoding="utf-8")
    return path


def read_pidfile(path: Optional[Path] = None) -> Optional[int]:
    """Read PID from ``.quantnodes.pid``. Returns None if missing/invalid."""
    path = Path(path) if path else Path.cwd() / PIDFILE_NAME
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def remove_pidfile(path: Optional[Path] = None) -> None:
    """Remove ``.quantnodes.pid`` if it exists (best-effort)."""
    path = Path(path) if path else Path.cwd() / PIDFILE_NAME
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is running.

    Uses ``kill(pid, 0)`` which doesn't actually send a signal but raises
    ``ProcessLookupError`` if the process is gone. Works on Linux/macOS.
    """
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if ``port`` on ``host`` is currently bindable.

    Tries to bind a TCP socket to the address. ``OSError`` (EADDRINUSE)
    means the port is taken; anything else means it's free. We don't
    actually listen (SO_REUSEADDR + immediate close).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def wait_for_health(api_url: str, timeout_s: int = 30,
                    poll_interval_s: float = 1.0) -> bool:
    """Poll ``GET {api_url}/api/agent/status`` until ``state == "running"``.

    Returns True on success, False on timeout. Used by ``quantnodes serve``
    to block until the FastAPI lifespan has wired the nanobot runtime.
    """
    import httpx  # local import: avoid hard dep at module-import time
    deadline = time.time() + timeout_s
    last_state = "unreachable"
    while time.time() < deadline:
        try:
            r = httpx.get(f"{api_url}/api/agent/status", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                last_state = data.get("state", "unknown")
                if last_state == "running":
                    return True
        except Exception:
            pass
        time.sleep(poll_interval_s)
    print(f"  ⚠ health check timeout after {timeout_s}s (last state={last_state})")
    return False


def is_nanobot_installed() -> bool:
    """Return True if ``nanobot-ai`` is importable.

    Used by ``quantnodes init`` and ``serve --check-env`` to give a
    friendly non-blocking warning when the optional ``[agent]`` extra
    is not installed. Quant tool library (Wiki/Factor/Backtest/Strategy)
    works without it; only Agent Chat / WebUI / MCP / Feishu need it.
    """
    try:
        from nanobot.agent.tools.base import Tool  # noqa: F401
        return True
    except ImportError:
        return False


def print_nanobot_install_hint() -> None:
    """Print a friendly hint for installing the optional ``[agent]`` extra."""
    if is_nanobot_installed():
        return
    print()
    print("ℹ 未检测到 nanobot-ai（量化 agent 可选依赖）")
    print("  安装后可启用 Agent Chat / WebUI / MCP / 飞书:")
    print("    pip install 'quantnodes[agent]'    # 或 'quantnodes[all]'")
    print("  未安装时量化工具库（Wiki / Factor / Backtest / Strategy）完全可用")

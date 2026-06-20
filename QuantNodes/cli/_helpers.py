# coding=utf-8
"""Helper functions and constants used by QuantNodes CLI commands."""

import sys
import subprocess
from pathlib import Path

from QuantNodes.research.wiki import init_factor_wiki
from QuantNodes.core.path_utils import ensure_dir

PROG_NAME = "quantnodes"
DEFAULT_API_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_HOST = "localhost"


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
        ".quant_agent/memory",
        ".quant_agent/dream",
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

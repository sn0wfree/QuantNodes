# coding=utf-8
"""
QuantNodes CLI - Command Line Interface

Commands:
    init    - Initialize current directory
    run     - Start services
    version - Show version
    help    - Show this help message
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Any

from QuantNodes import __version__
from QuantNodes.research.wiki import init_factor_wiki

PROG_NAME = "quantnodes"
DEFAULT_API_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_HOST = "localhost"


def is_initialized() -> bool:
    """Check if current directory is already initialized."""
    return Path(".env").exists() or Path("conn.ini").exists() or Path("wiki/index.md").exists()


def get_project_root() -> Path:
    """Get the QuantNodes package root directory."""
    return Path(__file__).parent.parent


def create_directory_structure():
    """Create necessary directories."""
    dirs = [
        "data",
        ".quant_agent/memory",
        ".quant_agent/dream",
        "outputs",
        "logs",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
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
    
    env_content += """# 缓存配置
QUANTNODES__CACHE_ENABLED=true
QUANTNODES__CACHE_TTL=3600
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


def cmd_init(args) -> int:
    """Initialize current directory for QuantNodes."""
    print()
    print("=" * 50)
    print("QuantNodes 初始化向导")
    print("=" * 50)
    
    current_dir = Path.cwd()
    print(f"\n✓ 检测当前目录: {current_dir}")
    print("✓ 检查初始化状态...")
    
    already_init = False
    if Path(".env").exists():
        print("  ✗ .env 已存在")
        already_init = True
    
    if Path("conn.ini").exists():
        print("  ✗ conn.ini 已存在")
        already_init = True
    
    if already_init and not args.force:
        print("\n错误: 当前目录已初始化")
        print("请先 cd 到其他目录，或使用 --force 强制重新初始化")
        return 1
    
    if already_init and args.force:
        print("  (强制模式: 将覆盖现有配置)")
    
    print()
    
    print("-" * 50)
    print("配置 LLM")
    print("-" * 50)
    
    api_key = get_input_with_default(
        "请输入 OpenAI API Key (sk-...)",
        "",
        required=True
    )
    while not api_key.startswith("sk-"):
        print("  错误: API Key 必须以 sk- 开头")
        api_key = get_input_with_default(
            "请输入 OpenAI API Key (sk-...)",
            "",
            required=True
        )
    
    base_url = get_input_with_default(
        "请输入 API Base URL",
        "https://api.openai.com/v1"
    )
    
    model = get_model_choice()
    if model == "custom":
        model = get_input_with_default("请输入自定义模型名称", "gpt-4")
    
    print()
    print("-" * 50)
    print("配置数据源")
    print("-" * 50)
    
    duckdb_path = get_input_with_default(
        "DuckDB 数据库路径",
        "data/quantnodes.db"
    )
    
    configure_clickhouse = get_yes_no("是否配置 ClickHouse", default=False)
    configure_mysql = get_yes_no("是否配置 MySQL", default=False)
    
    clickhouse_config = {}
    mysql_config = {}
    
    if configure_clickhouse:
        print("\n  ClickHouse 配置:")
        clickhouse_config["host"] = get_input_with_default("    Host", "localhost")
        clickhouse_config["port"] = get_input_with_default("    Port", "8123")
        clickhouse_config["user"] = get_input_with_default("    User", "default")
        clickhouse_config["passwd"] = get_input_with_default("    Password", "")
        clickhouse_config["db"] = get_input_with_default("    Database", "default")
    
    if configure_mysql:
        print("\n  MySQL 配置:")
        mysql_config["host"] = get_input_with_default("    Host", "localhost")
        mysql_config["port"] = get_input_with_default("    Port", "3306")
        mysql_config["user"] = get_input_with_default("    User", "root")
        mysql_config["passwd"] = get_input_with_default("    Password", "")
        mysql_config["db"] = get_input_with_default("    Database", "quant")
    
    print()
    
    print("-" * 50)
    print("初始化 llmwikify Wiki")
    print("-" * 50)
    init_llmwikify_wiki(force=args.force)
    print()
    
    print("-" * 50)
    print("创建目录结构")
    print("-" * 50)
    create_directory_structure()
    print()
    
    print("-" * 50)
    print("创建配置文件")
    print("-" * 50)
    
    write_env_file(api_key, base_url, model, duckdb_path, clickhouse_config, mysql_config)
    write_conn_ini(duckdb_path, clickhouse_config, mysql_config)
    
    print()
    
    install_talib_option = get_yes_no("是否安装 TA-Lib 技术分析库 (可选)", default=True)
    if install_talib_option:
        install_talib()
    
    print()
    print("=" * 50)
    print("✓ 初始化完成!")
    print("=" * 50)
    print()
    print("快速启动:")
    print("  # 启动后端")
    print("  python -m uvicorn api.main:app --reload --port 8000")
    print()
    print("  # 启动前端 (新终端)")
    print("  cd frontend && npm run dev")
    print()
    print("  # 或使用 quantnodes run 启动全部服务")
    print()
    print("访问 http://localhost:5173")
    print()
    
    return 0


def start_api_server(host: str, port: int, log_file: Optional[Path] = None) -> Tuple[subprocess.Popen, Optional[Any]]:
    """Start the API server. Returns (process, log_file_handle)."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
        "--reload"
    ]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            cwd=get_project_root()
        )
        return proc, log_fd
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=get_project_root()
        )
        return proc, None


def start_frontend_server(host: str, port: int, api_port: int = 8000, log_file: Optional[Path] = None) -> Tuple[subprocess.Popen, Optional[Any]]:
    """Start the frontend server. Returns (process, log_file_handle)."""
    cmd = ["npm", "run", "dev"]
    
    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(port)
    env["API_PORT"] = str(api_port)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(get_project_root() / "frontend"),
            env=env
        )
        return proc, log_fd
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(get_project_root() / "frontend"),
            env=env
        )
        return proc, None


def cmd_run(args) -> int:
    """Start QuantNodes services."""
    if not is_initialized():
        print("错误: 当前目录未初始化")
        print("请先运行: quantnodes init")
        return 1
    
    host = args.host or DEFAULT_HOST
    frontend_port = args.port or DEFAULT_FRONTEND_PORT
    # 联动：如果只设置 --port，则 api_port = port + 1000
    if args.port and not args.api_port:
        api_port = args.port + 1000
    else:
        api_port = args.api_port or DEFAULT_API_PORT
    
    if args.daemon:
        if sys.platform != "linux":
            print("错误: daemon 模式仅支持 Linux")
            return 1
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        api_log = log_dir / f"quantnodes_api_{timestamp}.log"
        frontend_log = log_dir / f"quantnodes_frontend_{timestamp}.log"
        
        print("=" * 50)
        print("QuantNodes 服务 (后台运行)")
        print("=" * 50)
        print(f"  后端: http://{host}:{api_port}")
        print(f"  前端: http://{host}:{frontend_port}")
        print(f"  API 日志: {api_log}")
        print(f"  前端日志: {frontend_log}")
        print()
        
        api_proc, api_fd = start_api_server(host, api_port, api_log)
        frontend_proc, frontend_fd = start_frontend_server(host, frontend_port, api_port, frontend_log)
        
        print(f"✓ 服务已后台启动")
        print(f"  API 进程: {api_proc.pid}")
        print(f"  前端进程: {frontend_proc.pid}")
        print()
        print("查看日志:")
        print(f"  tail -f {api_log}")
        print(f"  tail -f {frontend_log}")
        print()
        print("停止服务:")
        print(f"  kill {api_proc.pid} {frontend_proc.pid}")
        
        return 0
    
    print("=" * 50)
    print("QuantNodes 服务")
    print("=" * 50)
    
    processes: List[Tuple[str, subprocess.Popen]] = []
    log_fds: List[Any] = []
    
    try:
        if not args.frontend_only:
            print(f"\n启动后端: http://{host}:{api_port}")
            api_proc, api_fd = start_api_server(host, api_port)
            processes.append(("API", api_proc))
            log_fds.append(api_fd)
            print(f"  进程 PID: {api_proc.pid}")
        
        if not args.api_only:
            print(f"\n启动前端: http://{host}:{frontend_port}")
            frontend_proc, frontend_fd = start_frontend_server(host, frontend_port, api_port)
            processes.append(("Frontend", frontend_proc))
            log_fds.append(frontend_fd)
            print(f"  进程 PID: {frontend_proc.pid}")
        
        print()
        print("=" * 50)
        print("✓ 服务已启动")
        print("=" * 50)
        print()
        print("访问:")
        if not args.frontend_only:
            print(f"  后端: http://localhost:{api_port}/docs")
        if not args.api_only:
            print(f"  前端: http://localhost:{frontend_port}")
        print()
        print("按 Ctrl+C 停止服务")
        print()
        
        try:
            for name, proc in processes:
                proc.wait()
        except KeyboardInterrupt:
            print("\n\n正在停止服务...")
            for name, proc in processes:
                proc.terminate()
                proc.wait()
            for fd in log_fds:
                if fd:
                    fd.close()
            print("✓ 服务已停止")
        
    except Exception as e:
        print(f"错误: {e}")
        for name, proc in processes:
            proc.terminate()
        for fd in log_fds:
            if fd:
                fd.close()
        return 1
    
    return 0


def cmd_version(args) -> int:
    """Show version information."""
    print(f"QuantNodes version {__version__}")
    return 0


def cmd_help(args) -> int:
    """Show help message."""
    print("""
QuantNodes CLI - 量化研究节点架构命令行工具

用法:
    quantnodes <命令> [选项]

命令:
    init        初始化当前目录
    run         启动服务
    version     显示版本
    help        显示帮助

init 选项:
    --force           强制重新初始化 (覆盖现有配置)

run 选项:
    --host HOST         绑定主机 (默认: localhost)
    --port PORT         前端端口 (默认: 5173)，设置后后端端口自动设为 PORT+1000
    --api-port PORT     后端端口 (默认: 8000)，优先级高于 --port 联动
    --daemon           后台运行 (仅 Linux)
    --api-only         仅启动后端
    --frontend-only    仅启动前端

示例:
    quantnodes init
    quantnodes run                          # 前端:5173, 后端:8000
    quantnodes run --port 18380             # 前端:18380, 后端:19380 (联动)
    quantnodes run --port 18380 --api-port 9000  # 前端:18380, 后端:9000 (指定后端)
    quantnodes run --daemon
    quantnodes run --api-only
    quantnodes version

详细文档: docs/QuickStart.md
""")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog=PROG_NAME,
        description="QuantNodes CLI - 量化研究节点架构命令行工具",
        add_help=False
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    init_parser = subparsers.add_parser("init", help="初始化当前目录")
    init_parser.add_argument("--force", action="store_true", help="强制重新初始化")
    
    run_parser = subparsers.add_parser("run", help="启动服务")
    run_parser.add_argument("--host", help="绑定主机")
    run_parser.add_argument("--port", type=int, help="前端端口")
    run_parser.add_argument("--api-port", type=int, dest="api_port", help="后端端口")
    run_parser.add_argument("--daemon", action="store_true", help="后台运行 (仅 Linux)")
    run_parser.add_argument("--api-only", action="store_true", dest="api_only", help="仅启动后端")
    run_parser.add_argument("--frontend-only", action="store_true", dest="frontend_only", help="仅启动前端")
    
    subparsers.add_parser("version", help="显示版本")
    subparsers.add_parser("help", help="显示帮助")
    
    args = parser.parse_args()
    
    if args.command == "init":
        return cmd_init(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "version":
        return cmd_version(args)
    elif args.command == "help":
        return cmd_help(args)
    else:
        cmd_help(args)
        return 0


if __name__ == "__main__":
    sys.exit(main())

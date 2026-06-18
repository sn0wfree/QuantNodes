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
import json
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
            # Wait for backend to be ready before starting frontend
            import time
            import urllib.request
            import urllib.error
            print("  等待后端就绪...")
            for i in range(30):
                try:
                    urllib.request.urlopen(f"http://localhost:{api_port}/docs", timeout=2)
                    print("  ✓ 后端已就绪")
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(1)
            else:
                print("  ⚠ 后端未就绪，继续启动前端")
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
    chat        启动 Agent 对话模式
    evolve      多轮演化主入口 (Week 5)
    factor-info 显示 TrajectoryPool 统计 (Week 5)
    factor-best 显示 Top-N 最佳 entry (Week 5)
    factor-visual 生成可视化 HTML 报告 (Week 6)
    factor-rag-show RAG 检索演示 (Week 7)
    factor-rag-eval 批量评估 RAG 质量 (Week 10)
    factor-data-fetch 从 iFinD 拉取数据 + 写 H5 (Week 12)
    factor-dashboard 生成 3 类指标 dashboard (Week 13)
    version     显示版本
    help        显示帮助

evolve 选项:
    --config PATH          YAML 配置文件路径 (必填)
    --directions LIST      逗号分隔的研究方向
    --initial-json JSON    初始 candidates JSON
    --max-rounds N         覆盖 config.evolution.max_rounds
    --early-stop N         覆盖 config.evolution.early_stop_patience

factor-info / factor-best / factor-visual / factor-rag-show 选项:
    --pool-dir PATH        TrajectoryPool 目录
    --top N                Top-N (默认 5, 仅 factor-best/rag-show)
    --metric NAME          排序指标 (默认 sharpe, 仅 factor-best/visual)
    --output PATH          HTML 输出路径 (仅 factor-visual)
    --title TITLE          报告标题 (仅 factor-visual)
    --query TEXT           查询文本 (仅 factor-rag-show)
    --compress             启用谱系压缩 (Week 9, 仅 factor-rag-show)
    --ancestor-depth N     祖先深度 (默认 2, 仅 --compress)
    --descendant-depth N   后裔深度 (默认 2, 仅 --compress)
    --max-tokens N         压缩最大字符数 (默认 200, 仅 --compress)

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
    quantnodes evolve --config configs/evolve.yaml --directions momentum,reversal --max-rounds 3 --workers 4
    quantnodes factor-info --pool-dir output/trajectory
    quantnodes factor-best --pool-dir output/trajectory --top 10 --metric sharpe
    quantnodes factor-visual --pool-dir output/trajectory --output report.html
    quantnodes factor-rag-show --pool-dir output/trajectory --query "momentum effect" --top 5
    quantnodes factor-rag-eval --pool-dir output/trajectory --queries "momentum,reversal" --top 5
    quantnodes factor-data-fetch --output-dir /tmp/real_data/ --date-beg 20260101 --factors momentum_20d,reversal_5d
    quantnodes factor-dashboard --pool-dir output/trajectory --output dashboard.html
    quantnodes version

详细文档: docs/QuickStart.md
""")
    return 0


def cmd_chat(args):
    """启动 Agent 对话模式"""
    from .enhanced import chat, chat_single

    workspace = args.workspace
    if args.message:
        chat_single(args.message, workspace=workspace)
    else:
        chat(workspace=workspace)
    return 0


# ============================================================
# Week 5: 演化实验 CLI (evolve / factor-info / factor-best)
# ============================================================

def _load_runner_from_config(config_path: str):
    """从 YAML 配置构造 PipelineRunner (延迟 import 避免顶层依赖)。"""
    from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
    return PipelineRunner.from_yaml(config_path)


def cmd_evolve(args) -> int:
    """多轮演化主入口。

    用法:
        quantnodes evolve --config configs/single_factor.yaml \\
                         --directions momentum,reversal,volatility \\
                         --max-rounds 3
    """
    from QuantNodes.core.evolution import FactorCandidate

    config_path = args.config
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return 1

    directions = [d.strip() for d in (args.directions or "").split(",") if d.strip()]
    initial = None
    if args.initial_json:
        try:
            raw = json.loads(args.initial_json)
            initial = [FactorCandidate(**c) for c in raw]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"错误: --initial-json 解析失败: {e}")
            return 1

    try:
        runner = _load_runner_from_config(config_path)
    except Exception as e:
        print(f"错误: 加载配置失败: {e}")
        return 1

    # CLI 参数覆盖 config (如指定)
    if args.max_rounds is not None:
        runner.config.evolution.max_rounds = args.max_rounds
    if args.early_stop is not None:
        runner.config.evolution.early_stop_patience = args.early_stop
    # M13-M15: CLI overrides for common defaults
    if args.min_ipo_days is not None:
        runner.config.preprocess.tradable.min_ipo_days = args.min_ipo_days
    if args.min_group_size is not None:
        runner.config.ic.min_group_size = args.min_group_size
    if args.groups is not None:
        runner.config.group.groups = args.groups

    print("=" * 60)
    print(f"演化实验: {config_path}")
    print(f"  方向: {directions or '(无, 走 initial_candidates)'}")
    print(f"  max_rounds: {runner.config.evolution.max_rounds}")
    print(f"  early_stop: {runner.config.evolution.early_stop_patience}")
    if args.min_ipo_days is not None:
        print(f"  min_ipo_days: {runner.config.preprocess.tradable.min_ipo_days} (CLI override)")
    if args.min_group_size is not None:
        print(f"  min_group_size: {runner.config.ic.min_group_size} (CLI override)")
    if args.groups is not None:
        print(f"  groups: {runner.config.group.groups} (CLI override)")
    print("=" * 60)

    try:
        result = runner.run_evolution(
            initial_directions=directions or None,
            initial_candidates=initial,
            workers=getattr(args, 'workers', 1),
        )
    except Exception as e:
        print(f"错误: 演化失败: {e}")
        return 1

    print()
    print("=" * 60)
    print(f"演化完成: {result.rounds_completed} 轮")
    print(f"  总数: {result.total_count}, 拒绝: {result.rejected_count}")
    print(f"  Top {len(result.best_entries)} entries:")
    for i, e in enumerate(result.best_entries[:5], 1):
        metric_val = e.metrics.get(runner.config.evolution.metric, 0)
        name = e.feedback.factor_name if e.feedback else e.entry_id[:8]
        print(f"    {i}. {name} [{e.operation} r{e.round_idx}] "
              f"{runner.config.evolution.metric}={metric_val:.4f}")
    print("=" * 60)
    return 0


def cmd_factor_info(args) -> int:
    """显示 TrajectoryPool 统计信息。

    用法:
        quantnodes factor-info --pool-dir output/trajectory/
    """
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    print("=" * 60)
    print(f"TrajectoryPool: {pool_dir}")
    print(f"  size: {pool.size}")
    by_round: dict = {}
    for e in pool.all():
        by_round.setdefault(e.round_idx, 0)
        by_round[e.round_idx] += 1
    print(f"  by_round: {by_round}")
    by_op: dict = {}
    for e in pool.all():
        by_op.setdefault(e.operation, 0)
        by_op[e.operation] += 1
    print(f"  by_operation: {by_op}")
    n_passed = sum(1 for e in pool.all() if e.feedback and e.feedback.decision)
    print(f"  passed: {n_passed} / {pool.size}")
    print("=" * 60)
    return 0


def cmd_factor_best(args) -> int:
    """显示 Top-N 最佳 entry (按 metric 排序)。

    用法:
        quantnodes factor-best --pool-dir output/trajectory/ --top 5 --metric sharpe
    """
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    top = pool.best(top_n=args.top, metric=args.metric)
    print("=" * 60)
    print(f"Top {len(top)} entries by {args.metric}:")
    for i, e in enumerate(top, 1):
        metric_val = e.metrics.get(args.metric, 0)
        name = e.feedback.factor_name if e.feedback else e.entry_id[:8]
        print(f"  {i}. {name} [{e.operation} r{e.round_idx}] "
              f"{args.metric}={metric_val:.4f}")
    print("=" * 60)
    return 0


def cmd_factor_visual(args) -> int:
    """生成可视化 HTML 报告 (谱系 DAG + 指标分布 + 拦截率 + 趋势)。

    用法:
        quantnodes factor-visual --pool-dir output/trajectory/ \\
                                --output report.html --metric sharpe
    """
    from QuantNodes.core.trajectory import TrajectoryPool
    from QuantNodes.core.visualization import generate_html

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无 entry 可视化")
        return 1

    output = args.output or str(Path(pool_dir).parent / f"{Path(pool_dir).name}_report.html")
    title = args.title or f"QuantNodes 演化报告: {pool_dir}"
    try:
        generate_html(pool, metric=args.metric, title=title, output_path=output)
    except Exception as e:
        print(f"错误: 生成报告失败: {e}")
        return 1
    print(f"✓ HTML 报告已生成: {output}")
    print(f"  size: {pool.size}, metric: {args.metric}")
    return 0


def cmd_factor_dashboard(args) -> int:
    """生成 3 类指标 dashboard (Week 13)。

    从 TrajectoryPool 提取 RAG + Evolution + Quality Gate 指标,
    生成 Plotly 6 图 + 概览表 HTML 报告。

    用法:
        quantnodes factor-dashboard --pool-dir output/trajectory/ \\
                                     --output dashboard.html
    """
    from QuantNodes.core.feedback import FactorFeedback, FeedbackChannel, ChannelFeedback
    from QuantNodes.core.monitoring import (
        MetricCollector,
        generate_dashboard_html,
    )
    from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无指标可显示")
        return 1

    output = args.output or str(Path(pool_dir).parent / f"{Path(pool_dir).name}_dashboard.html")
    title = args.title or f"QuantNodes 演化 Dashboard: {Path(pool_dir).name}"

    # 收集 3 类指标
    collector = MetricCollector()

    # RAG: 从 TrajectoryEntry.feedback 元数据中提取 (兼容 rag_metrics_history 缺失)
    rounds = sorted({e.round_idx for e in pool.all()})
    for r in rounds:
        round_entries = [e for e in pool.all() if e.round_idx == r]
        n_total = len(round_entries)
        n_passed = sum(1 for e in round_entries if e.feedback and e.feedback.decision)
        # RAG 指标的简单代理: pass rate 作为 HR@5
        if n_total > 0:
            from QuantNodes.core.monitoring import RagMetrics
            collector.add_rag(RagMetrics(
                round=r, n_queries=n_total,
                hit_at_5=n_passed / n_total, hit_at_10=n_passed / n_total,
                ndcg_at_5=n_passed / n_total, ndcg_at_10=n_passed / n_total,
                mrr=n_passed / n_total,
                lineage_coverage=0.0,
                diversity=1.0,
            ))

    # Evolution: 累积统计
    from QuantNodes.core.monitoring import EvolutionMetrics, QualityMetrics
    for r in rounds:
        round_entries = [e for e in pool.all() if e.round_idx <= r]
        n_passed = sum(1 for e in round_entries if e.feedback and e.feedback.decision)
        n_total = len(round_entries)
        n_rejected = n_total - n_passed
        best_metric = 0.0
        best_name = ""
        for e in round_entries:
            sharpe = (e.metrics or {}).get("sharpe", 0)
            if sharpe > best_metric:
                best_metric = sharpe
                if e.feedback:
                    best_name = e.feedback.factor_name
        collector.add_evolution(EvolutionMetrics(
            round=r, pool_size=n_total,
            total_count=n_passed, rejected_count=n_rejected,
            best_metric=best_metric, best_factor_name=best_name,
        ))

    # Quality: 每 round 通道统计
    for r in rounds:
        collector.update_quality_from_pool(pool, round_idx=r)

    print("=" * 60)
    print(f"Dashboard 收集 ({len(collector)} metrics):")
    print(f"  RAG:    {len(collector.rag_history)} rounds")
    print(f"  Evo:    {len(collector.evolution_history)} rounds")
    print(f"  Quality: {len(collector.quality_history)} rounds")
    print("=" * 60)

    try:
        streaming = getattr(args, "streaming", False) or getattr(args, "watch", False)
        refresh_sec = getattr(args, "refresh", 10)
        generate_dashboard_html(
            collector, title=title, output_path=output,
            streaming=streaming, refresh_interval_sec=refresh_sec,
        )
    except Exception as e:
        print(f"错误: 生成 dashboard 失败: {e}")
        return 1

    # 同时保存 JSON (供后续分析)
    metrics_json = output.replace(".html", "_metrics.json")
    collector.save(metrics_json)
    print(f"✓ Dashboard: {output}")
    print(f"✓ Metrics JSON: {metrics_json}")

    # Watch 模式: 后台定时刷新
    if getattr(args, "watch", False):
        import time as _time
        refresh_sec = getattr(args, "refresh", 10)
        print(f"\n[Watch] 每 {refresh_sec}s 刷新 dashboard (Ctrl+C 退出)...")
        try:
            while True:
                _time.sleep(refresh_sec)
                # 重载 pool + 重新生成
                try:
                    pool = TrajectoryPool(pool_dir)
                    collector = MetricCollector()
                    for r in rounds:
                        round_entries = [e for e in pool.all() if e.round_idx == r]
                        if not round_entries:
                            continue
                        collector.update_quality_from_pool(pool, round_idx=r)
                    generate_dashboard_html(
                        collector, title=title, output_path=output,
                        streaming=True, refresh_interval_sec=refresh_sec,
                    )
                except Exception:
                    pass  # pool 可能被其他进程写入, 忽略暂时错误
        except KeyboardInterrupt:
            print("\n[Watch] 停止监控")
    return 0


def cmd_factor_data_fetch(args) -> int:
    """从 iFinD 拉取数据 + 写为 HDF5 格式 (Week 12)。

    用法:
        quantnodes factor-data-fetch --output-dir /tmp/real_data/ \\
                                    --universe all \\
                                    --date-beg 20260101 --date-end 20260630 \\
                                    --factors momentum_20d,reversal_5d
    """
    try:
        from QuantNodes.research.factor_test.ifind_db import IFinDDatabase
    except (ImportError, FileNotFoundError) as e:
        print(f"错误: 无法导入 IFinDDatabase: {e}")
        return 1

    output_dir = Path(args.output_dir)
    try:
        db = IFinDDatabase(
            date_beg=args.date_beg,
            date_end=args.date_end,
            universe=args.universe,
        )
    except FileNotFoundError as e:
        print(f"错误: iFinD 配置缺失: {e}")
        return 1
    except ValueError as e:
        print(f"错误: iFinD auth_token 无效: {e}")
        return 1

    factor_names = [f.strip() for f in (args.factors or "").split(",") if f.strip()]

    print("=" * 60)
    print(f"iFinD 数据拉取")
    print(f"  universe: {args.universe}")
    print(f"  date range: {args.date_beg} ~ {args.date_end}")
    print(f"  output_dir: {output_dir}")
    print(f"  factors: {factor_names or '(none)'}")
    print("=" * 60)

    try:
        stats = db.fetch_to_h5(output_dir, factor_names=factor_names)
    except Exception as e:
        print(f"错误: 拉取失败: {e}")
        return 1

    print()
    print("=" * 60)
    print(f"✓ 完成, 统计:")
    for fname, file_stats in stats.items():
        if isinstance(file_stats, dict):
            keys_info = ", ".join(
                f"{k}={v}" for k, v in file_stats.items() if v
            )
            print(f"  {fname}: {keys_info or '(empty)'}")
    print("=" * 60)
    return 0


def cmd_factor_rag_eval(args) -> int:
    """批量评估 RAG 检索质量 (Week 10)。

    用法:
        quantnodes factor-rag-eval --pool-dir output/trajectory/ \\
                                   --queries "momentum,reversal,volatility" \\
                                   --top 5 \\
                                   --output eval.json
    """
    from QuantNodes.core.knowledge import (
        IdentityRetriever,
        KnowledgeBase,
        RAGEvaluator,
        expand_lineage,
    )
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无可评估内容")
        return 1

    queries = [q.strip() for q in (args.queries or "").split(",") if q.strip()]
    if not queries:
        print("错误: --queries 至少需要 1 个 query")
        return 1

    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    n = kb.sync_from_pool()

    # 构造评估输入
    all_ids = {e.entry_id for e in pool.all()}
    retrieved: list[list[str]] = []
    relevant: list[list[str]] = []
    relevance_scores: list[dict[str, float]] = []
    lineage_ids: list[list[str]] = []
    token_lists: list[list[list[str]]] = []

    for q in queries:
        results = kb.query(q, top_k=args.top)
        ids = [e.entry_id for e, _ in results]
        retrieved.append(ids)
        relevant.append(list(all_ids))
        relevance_scores.append({eid: 1.0 for eid in ids})
        lin_set: set[str] = set()
        tokens_per_entry: list[list[str]] = []
        for e, _ in results:
            expanded = expand_lineage(
                pool, e.entry_id,
                max_ancestor_depth=args.ancestor_depth,
                max_descendant_depth=args.descendant_depth,
            )
            for _, ee in expanded["ancestors"] + expanded["descendants"]:
                lin_set.add(ee.entry_id)
            cfg = (e.config_snapshot or {}).get("factor", {}) if e else {}
            toks = []
            if cfg.get("name"):
                toks += cfg["name"].lower().split("_")
            if cfg.get("hypothesis"):
                toks += cfg["hypothesis"].lower().split()
            tokens_per_entry.append(toks)
        lineage_ids.append(list(lin_set))
        token_lists.append(tokens_per_entry)

    ev = RAGEvaluator()
    report = ev.evaluate(
        queries=queries,
        retrieved=retrieved,
        relevant=relevant,
        relevance_scores=relevance_scores,
        lineage_ids=lineage_ids,
        token_lists=token_lists,
    )

    if args.output:
        ev.save(report, args.output)
        print(f"✓ EvalReport 已保存: {args.output}")

    print("=" * 60)
    print(f"RAG 评估报告 ({report.n_queries} queries, indexed {n} entries)")
    print(f"  HitRate@5:   {report.hit_at_5:.3f}")
    print(f"  HitRate@10:  {report.hit_at_10:.3f}")
    print(f"  NDCG@5:      {report.ndcg_at_5:.3f}")
    print(f"  NDCG@10:     {report.ndcg_at_10:.3f}")
    print(f"  MRR:         {report.mrr:.3f}")
    print(f"  LineageCov:  {report.lineage_coverage:.3f}")
    print(f"  Diversity:   {report.diversity:.3f}")
    print("=" * 60)
    return 0


def cmd_factor_rag_show(args) -> int:
    """从 TrajectoryPool 检索相似因子 (RAG demo)。

    用法:
        quantnodes factor-rag-show --pool-dir output/trajectory/ \\
                                   --query "momentum effect" --top 5
    """
    from QuantNodes.core.knowledge import (
        Compressor,
        IdentityRetriever,
        KnowledgeBase,
        expand_lineage,
    )
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无可检索内容")
        return 1

    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    n = kb.sync_from_pool()
    print(f"索引了 {n} 个 entry")

    results = kb.query(args.query, top_k=args.top)
    if not results:
        print(f"无匹配结果 (query: {args.query!r})")
        return 0

    use_compress = getattr(args, "compress", False)
    compressor = Compressor(model="mock", max_tokens=args.max_tokens) if use_compress else None

    print("=" * 60)
    print(f"Top {len(results)} 检索结果 (query: {args.query!r}):")
    for i, (entry, score) in enumerate(results, 1):
        cfg = entry.config_snapshot or {}
        factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        name = factor_cfg.get("name", entry.entry_id[:8])
        expr = factor_cfg.get("expression", "")[:50]
        sharpe = (entry.metrics or {}).get("sharpe", 0)
        print(f"  {i}. {name}  score={score:.3f}  sharpe={sharpe:.2f}")
        print(f"     expression: {expr}")
        if use_compress and compressor is not None:
            expanded = expand_lineage(
                pool, entry.entry_id,
                max_ancestor_depth=args.ancestor_depth,
                max_descendant_depth=args.descendant_depth,
            )
            c_anc = compressor.compress(expanded["ancestors"], relation="ancestors")
            c_desc = compressor.compress(expanded["descendants"], relation="descendants")
            print(f"     ↑ ancestors ({c_anc.original_count}): {c_anc.summary[:80]}")
            print(f"     ↓ descendants ({c_desc.original_count}): {c_desc.summary[:80]}")
    print("=" * 60)
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
    
    chat_parser = subparsers.add_parser("chat", help="启动 Agent 对话模式")
    chat_parser.add_argument("message", nargs="?", help="单次提问（不指定则进入交互模式）")
    chat_parser.add_argument("--workspace", default=".", help="工作目录")
    
    # Week 5: 演化实验子命令
    evolve_parser = subparsers.add_parser("evolve", help="多轮演化主入口")
    evolve_parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    evolve_parser.add_argument("--directions", default="", help="逗号分隔的研究方向")
    evolve_parser.add_argument("--initial-json", default=None, help="初始 candidates JSON")
    evolve_parser.add_argument("--max-rounds", type=int, default=None, help="覆盖 config.max_rounds")
    evolve_parser.add_argument("--early-stop", type=int, default=None, help="覆盖 config.early_stop_patience")
    evolve_parser.add_argument("--workers", type=int, default=1, help="并行评估数 (默认 1=串行, >1=ThreadPool)")
    # M13-M15: 增加 CLI 覆盖配置默认值
    evolve_parser.add_argument("--min-ipo-days", type=int, default=None, help="剔除上市不足 N 日新股 (覆盖 config 默认值 360)")
    evolve_parser.add_argument("--min-group-size", type=int, default=None, help="计算 IC 最少样本数 (覆盖 config 默认值 5)")
    evolve_parser.add_argument("--groups", type=int, default=None, help="分组分析分组数 (覆盖 config 默认值 5)")
    
    info_parser = subparsers.add_parser("factor-info", help="显示 TrajectoryPool 统计")
    info_parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")
    
    best_parser = subparsers.add_parser("factor-best", help="显示 Top-N 最佳 entry")
    best_parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")
    best_parser.add_argument("--top", type=int, default=5, help="Top-N (默认 5)")
    best_parser.add_argument("--metric", default="sharpe", help="排序指标 (默认 sharpe)")

    visual_parser = subparsers.add_parser("factor-visual", help="生成可视化 HTML 报告 (Week 6)")
    visual_parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")
    visual_parser.add_argument("--output", default=None, help="HTML 输出路径 (默认 <pool-dir>_report.html)")
    visual_parser.add_argument("--metric", default="sharpe", help="用于可视化的指标 (默认 sharpe)")
    visual_parser.add_argument("--title", default=None, help="报告标题")

    rag_parser = subparsers.add_parser("factor-rag-show", help="RAG 检索演示 (Week 7)")
    rag_parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")
    rag_parser.add_argument("--query", required=True, help="查询文本")
    rag_parser.add_argument("--top", type=int, default=5, help="Top-K (默认 5)")
    rag_parser.add_argument("--compress", action="store_true", help="启用谱系压缩 (Week 9)")
    rag_parser.add_argument("--ancestor-depth", type=int, default=2, help="祖先深度 (默认 2)")
    rag_parser.add_argument("--descendant-depth", type=int, default=2, help="后裔深度 (默认 2)")
    rag_parser.add_argument("--max-tokens", type=int, default=200, help="压缩最大字符数 (默认 200)")

    rag_eval_parser = subparsers.add_parser("factor-rag-eval", help="批量评估 RAG 检索质量 (Week 10)")
    rag_eval_parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")
    rag_eval_parser.add_argument("--queries", required=True, help="逗号分隔的 query 列表")
    rag_eval_parser.add_argument("--top", type=int, default=5, help="Top-K (默认 5)")
    rag_eval_parser.add_argument("--ancestor-depth", type=int, default=2, help="祖先深度")
    rag_eval_parser.add_argument("--descendant-depth", type=int, default=2, help="后裔深度")
    rag_eval_parser.add_argument("--output", default=None, help="EvalReport JSON 输出路径")

    fetch_parser = subparsers.add_parser("factor-data-fetch", help="从 iFinD 拉取数据 + 写 H5 (Week 12)")
    fetch_parser.add_argument("--output-dir", required=True, help="HDF5 输出目录")
    fetch_parser.add_argument("--date-beg", required=True, help="起始日期 (YYYYMMDD)")
    fetch_parser.add_argument("--date-end", default="", help="截止日期 (空=今天)")
    fetch_parser.add_argument("--universe", default="all", help="股票池 (默认 all, 与 iFinD API 兼容)")
    fetch_parser.add_argument("--factors", default="", help="逗号分隔的因子列表")
    # M13-M15: 增加 CLI 覆盖配置默认值
    fetch_parser.add_argument("--min-ipo-days", type=int, default=None, help="剔除上市不足 N 日新股 (默认 360)")
    fetch_parser.add_argument("--min-group-size", type=int, default=None, help="计算 IC 最少样本数 (默认 5)")
    fetch_parser.add_argument("--groups", type=int, default=None, help="分组分析分组数 (默认 5)")

    dash_parser = subparsers.add_parser("factor-dashboard", help="生成 3 类指标 dashboard (Week 13/16)")
    dash_parser.add_argument("--pool-dir", required=True, help="Pool 目录路径")
    dash_parser.add_argument("--output", default=None, help="HTML 输出路径")
    dash_parser.add_argument("--title", default=None, help="报告标题")
    dash_parser.add_argument("--streaming", action="store_true", help="启用 streaming 模式 (自动刷新 10s)")
    dash_parser.add_argument("--refresh", type=int, default=10, help="streaming 刷新间隔秒数 (默认 10)")
    dash_parser.add_argument("--watch", action="store_true", help="后台模式: 每 10s 刷新 dashboard")
    
    subparsers.add_parser("version", help="显示版本")
    subparsers.add_parser("help", help="显示帮助")
    
    args = parser.parse_args()
    
    if args.command == "init":
        return cmd_init(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "chat":
        return cmd_chat(args)
    elif args.command == "evolve":
        return cmd_evolve(args)
    elif args.command == "factor-info":
        return cmd_factor_info(args)
    elif args.command == "factor-best":
        return cmd_factor_best(args)
    elif args.command == "factor-visual":
        return cmd_factor_visual(args)
    elif args.command == "factor-rag-show":
        return cmd_factor_rag_show(args)
    elif args.command == "factor-rag-eval":
        return cmd_factor_rag_eval(args)
    elif args.command == "factor-data-fetch":
        return cmd_factor_data_fetch(args)
    elif args.command == "factor-dashboard":
        return cmd_factor_dashboard(args)
    elif args.command == "version":
        return cmd_version(args)
    elif args.command == "help":
        return cmd_help(args)
    else:
        cmd_help(args)
        return 0


if __name__ == "__main__":
    sys.exit(main())

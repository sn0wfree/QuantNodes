# coding=utf-8
"""``quantnodes init`` command."""

from pathlib import Path

from QuantNodes.cli.command import Command
from .._helpers import (
    confirm_section,
    create_directory_structure,
    get_input_with_default,
    get_model_choice,
    get_yes_no,
    init_llmwikify_wiki,
    install_talib,
    print_nanobot_install_hint,
    write_conn_ini,
    write_env_file,
)


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
        clickhouse_config = confirm_section(
            "ClickHouse",
            [
                ("Host", "localhost"),
                ("Port", "8123"),
                ("User", "default"),
                ("Password", ""),
                ("Database", "default"),
            ],
        ) or {}

    if configure_mysql:
        mysql_config = confirm_section(
            "MySQL",
            [
                ("Host", "localhost"),
                ("Port", "3306"),
                ("User", "root"),
                ("Password", ""),
                ("Database", "quant"),
            ],
        ) or {}

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
    print("  # 启动后端（推荐）")
    print("  quantnodes serve                  # 前台，Ctrl+C 停止")
    print("  quantnodes serve --daemon         # 后台，写 .quantnodes.pid")
    print("  quantnodes serve --frontend       # 同时启动 Vite dev server")
    print("  quantnodes serve --check-env     # 启动前校验 API key")
    print()
    print("  # 服务管理")
    print("  quantnodes status                 # health + agent state")
    print("  quantnodes logs -f                # 实时日志")
    print("  quantnodes stop                   # 停止后台 serve")
    print()
    print("  # Agent Chat（HTTP 模式，需后端在跑）")
    print("  quantnodes agent chat '一句话回答动量因子'")
    print()
    print("  # 启动前端 (新终端)")
    print("  cd frontend && npm run dev")
    print()
    print("  # 或使用 quantnodes run 启动全部服务（旧接口，兼容保留）")
    print()
    print("访问 http://localhost:5173")
    print()

    # v3.0.0 Stage 7: 友好提示 nanobot-ai 可选依赖
    print_nanobot_install_hint()

    return 0


class InitCommand(Command):
    """``quantnodes init`` subcommand."""

    name = "init"
    description = "初始化当前目录"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("--force", action="store_true", help="强制重新初始化")

    def run(self, args) -> int:
        return cmd_init(args)

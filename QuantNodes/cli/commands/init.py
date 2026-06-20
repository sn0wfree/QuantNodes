# coding=utf-8
"""``quantnodes init`` command."""

from pathlib import Path

from .._helpers import (
    create_directory_structure,
    get_input_with_default,
    get_model_choice,
    get_yes_no,
    init_llmwikify_wiki,
    install_talib,
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

# coding=utf-8
"""
配置驱动的回测工具

接受 YAML 配置，通过 ConfigBacktestRunner 直接执行回测。
"""

from typing import Any, Dict
import yaml

from QuantNodes.agent.tools.base import Tool


class ConfigBacktestTool(Tool):
    """配置驱动的回测工具

    通过 YAML 配置文件定义策略，直接执行回测。

    工作流程:
    1. 加载 YAML 配置
    2. 检查算子覆盖度
    3. 调用 ConfigBacktestRunner 执行回测
    4. 返回结果
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "config_backtest"

    @property
    def description(self) -> str:
        return (
            "通过 YAML 配置文件执行策略回测。"
            "支持因子定义、运算配置、组合因子和回测参数。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "config_yaml": {
                    "type": "string",
                    "description": "YAML 格式的策略配置字符串"
                },
                "config_path": {
                    "type": "string",
                    "description": "YAML 配置文件路径"
                },
                "data_path": {
                    "type": "string",
                    "description": "数据文件路径 (csv/parquet)"
                },
                "start_date": {
                    "type": "string",
                    "description": "覆盖配置中的开始日期"
                },
                "end_date": {
                    "type": "string",
                    "description": "覆盖配置中的结束日期"
                },
                "initial_cash": {
                    "type": "number",
                    "description": "覆盖配置中的初始资金"
                }
            },
            "required": []
        }

    @property
    def read_only(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def execute(
        self,
        config_yaml: str = None,
        config_path: str = None,
        data_path: str = None,
        start_date: str = None,
        end_date: str = None,
        initial_cash: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        result = {
            "status": "success",
            "summary": {},
            "config_info": {},
        }

        try:
            # 1. 加载配置
            strategy_config = self._load_config(config_yaml, config_path)

            if strategy_config is None:
                result["status"] = "error"
                result["errors"] = ["Need config_yaml or config_path"]
                return result

            # 2. 覆盖参数
            if start_date and strategy_config.backtest:
                strategy_config.backtest.start_date = start_date
            if end_date and strategy_config.backtest:
                strategy_config.backtest.end_date = end_date
            if initial_cash and strategy_config.backtest:
                strategy_config.backtest.initial_cash = initial_cash

            # 3. 检查覆盖度
            from QuantNodes.agent.config.loader import ConfigLoader
            loader = ConfigLoader()
            coverage = loader.check_coverage(strategy_config)

            if not coverage.is_complete:
                result["status"] = "warning"
                result["warnings"] = [
                    f"Unresolved operators: {coverage.unresolved}"
                ]

            # 4. 加载数据
            data = self._load_data(strategy_config, data_path)

            # 5. 调用 ConfigBacktestRunner
            from QuantNodes.backtest.config_runner import ConfigBacktestRunner

            runner = ConfigBacktestRunner()
            bt_result = runner.run(strategy_config, data)

            # 6. 保存输出文件
            saved_files = {}
            if strategy_config.output is not None:
                signals_df = None
                if bt_result.statistics.get("total_trades", 0) > 0:
                    signals_df = bt_result.trades
                saved_files = runner.save_output(
                    bt_result, strategy_config, signals_df=signals_df
                )

            # 7. 格式化返回结果
            result["status"] = "success"
            result["summary"] = {
                "total_trades": bt_result.statistics.get("total_trades", 0),
                "final_cash": bt_result.final_cash,
                "total_commission": bt_result.statistics.get("total_commission", 0),
                "total_return": bt_result.total_return,
                "sharpe_ratio": bt_result.sharpe_ratio,
                "max_drawdown": bt_result.max_drawdown,
                "win_rate": bt_result.win_rate,
                "annualized_return": bt_result.statistics.get("annualized_return", 0),
                "annualized_volatility": bt_result.statistics.get("annualized_volatility", 0),
                "sortino_ratio": bt_result.statistics.get("sortino_ratio", 0),
                "calmar_ratio": bt_result.statistics.get("calmar_ratio", 0),
                "profit_factor": bt_result.statistics.get("profit_factor", 0),
                "avg_trade_pnl": bt_result.statistics.get("avg_trade_pnl", 0),
                "trading_days": bt_result.statistics.get("trading_days", 0),
            }

            # 8. 附加配置信息
            result["config_info"] = {
                "name": strategy_config.name,
                "description": strategy_config.description,
                "factors": len(strategy_config.factors),
                "operations": len(strategy_config.operations),
                "composites": len(strategy_config.composite),
                "has_backtest": strategy_config.backtest is not None,
            }

            if saved_files:
                result["output_files"] = saved_files

        except Exception as e:
            result["status"] = "error"
            result["errors"] = [str(e)]

        return result

    def _load_config(self, config_yaml: str = None, config_path: str = None):
        """加载配置"""
        from QuantNodes.agent.config.loader import ConfigLoader

        if config_yaml:
            try:
                data = yaml.safe_load(config_yaml)
                if data is None:
                    return None
                loader = ConfigLoader()
                return loader._parse(data)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML: {e}")

        if config_path:
            loader = ConfigLoader()
            return loader.load(config_path)

        return None

    def _load_data(self, config, data_path: str = None):
        """加载数据为 Polars LazyFrame
        
        数据加载分发逻辑:
        1. data_path 参数 → 直接读文件（向后兼容）
        2. config.data.source == "csv"/"parquet" → 读文件
        3. config.data.source == "clickhouse"/"mysql" → _load_from_db()
        4. config.data.source == "sqlite"/"duckdb" → _load_from_db() (path-based)
        """

        # 优先使用 data_path 参数
        if data_path:
            return self._read_data_file(data_path)

        if not config.data:
            raise ValueError("No data configuration provided. Set config.data or data_path")

        source = config.data.source

        # 文件类数据源
        if source in ("csv", "parquet"):
            if config.data.path:
                lf = self._read_data_file(config.data.path)
                # 应用列名映射
                if config.data.column_mapping:
                    lf = lf.rename(config.data.column_mapping)
                return lf
            raise ValueError(f"source='{source}' requires config.data.path")

        # 数据库类数据源
        if source in ("clickhouse", "mysql", "sqlite", "duckdb"):
            return self._load_from_db(config)

        raise ValueError(f"Unsupported data source: {source}")

    def _load_from_db(self, config):
        """从 database_node 加载数据
        
        流程: 读取 conn.ini → 构建 SQL → 查询 → 列名映射 → 返回 LazyFrame
        
        如果 cache_enabled=True, 优先使用 MarketDataCacheNode 缓存查询结果。
        
        注意: DateTime 转换使用映射后的 date_column（而非 db_date_column）。
        """
        import polars as pl

        data_cfg = config.data
        source = data_cfg.source

        # 1. 构建 Node 实例
        node = self._build_db_node(source, data_cfg)

        # 2. 缓存逻辑
        if data_cfg.cache_enabled:
            df = self._load_with_cache(data_cfg, node)
        else:
            df = self._load_from_db_direct(data_cfg, node)

        # 3. 列名映射
        if data_cfg.column_mapping:
            df = df.rename(columns=data_cfg.column_mapping)

        # 4. DateTime → Date 类型转换（使用映射后的 date_column）
        date_col = data_cfg.date_column
        if date_col in df.columns:
            try:
                import polars as pl_polars
                pdf = pl_polars.from_pandas(df)
                if pdf.schema.get(date_col) == pl_polars.Datetime:
                    pdf = pdf.with_columns(pl_polars.col(date_col).cast(pl_polars.Date))
                    df = pdf.to_pandas()
            except Exception:
                pass

        return pl.from_pandas(df).lazy()

    def _load_with_cache(self, data_cfg, node):
        """使用缓存加载数据"""
        from QuantNodes.cache_node import MarketDataCacheNode

        cache_node = MarketDataCacheNode(config={
            "cache_dir": data_cfg.cache_dir,
            "ttl_days": data_cfg.cache_ttl_days,
            "force_refresh": data_cfg.cache_force_refresh,
        })

        return cache_node.execute({
            "source": data_cfg.source,
            "table": data_cfg.table,
            "columns": data_cfg.columns,
            "query_filter": data_cfg.query_filter,
            "node": node,
            "date_column": data_cfg.db_date_column or data_cfg.date_column,
        })

    def _load_from_db_direct(self, data_cfg, node):
        """直接从数据库加载 (不使用缓存)"""
        try:
            node.connect()
            sql = self._build_query(data_cfg)
            df = node.query(sql)
        finally:
            node.disconnect()
        return df

    def _build_db_node(self, source, data_cfg):
        """构建 database_node 实例"""
        from pathlib import Path
        from QuantNodes.conf_node.ini_config import IniConfigNode

        if source in ("sqlite", "duckdb"):
            return self._build_embedded_node(source, data_cfg)

        # clickhouse / mysql: 从 conn.ini 读取连接参数
        if not data_cfg.conn_ini:
            raise ValueError(f"source='{source}' requires conn_ini")

        ini_path = Path(data_cfg.conn_ini)
        if not ini_path.exists():
            raise FileNotFoundError(f"conn.ini not found: {ini_path}")

        ini = IniConfigNode(str(ini_path), section=data_cfg.conn_section)
        conn_params = ini.execute()

        if source == "clickhouse":
            from QuantNodes.database_node import ClickHouseNode
            return ClickHouseNode(
                host=conn_params.get("host", "localhost"),
                port=int(conn_params.get("port", 8123)),
                user=conn_params.get("user", "default"),
                passwd=conn_params.get("passwd", ""),
                database=conn_params.get("db", "default"),
            )
        elif source == "mysql":
            from QuantNodes.database_node import MySQLNode
            return MySQLNode(
                host=conn_params.get("host", "localhost"),
                port=int(conn_params.get("port", 3306)),
                user=conn_params.get("user", "root"),
                passwd=conn_params.get("passwd", ""),
                db=conn_params.get("db", ""),
            )

    def _build_embedded_node(self, source, data_cfg):
        """构建嵌入式数据库 Node (sqlite/duckdb)"""
        path = data_cfg.path
        if not path:
            raise ValueError(f"source='{source}' requires config.data.path")

        if source == "sqlite":
            from QuantNodes.database_node import SQLiteNode
            return SQLiteNode(database=path)
        elif source == "duckdb":
            from QuantNodes.database_node import DuckDBNode
            return DuckDBNode(database=path)

    def _build_query(self, data_cfg):
        """从 DataConfig 构建 SQL 查询
        
        使用 db_*_column 作为 SQL 标识符（数据库原始列名）。
        如果 db_*_column 为空，则 fallback 到 date_column/code_column。
        """
        cols = data_cfg.columns or ["*"]
        cols_str = ", ".join(cols)
        table = data_cfg.table

        if not table:
            raise ValueError("DataConfig.table is required for database sources")

        sql = f"SELECT {cols_str} FROM {table}"

        where_parts = []
        if data_cfg.query_filter:
            where_parts.append(data_cfg.query_filter.lstrip("WHERE "))

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        code_col = data_cfg.db_code_column or data_cfg.code_column
        date_col = data_cfg.db_date_column or data_cfg.date_column
        sql += f" ORDER BY {code_col}, {date_col}"

        return sql

    def _read_data_file(self, path: str):
        """读取数据文件"""
        import polars as pl

        if path.endswith(".csv"):
            return pl.scan_csv(path)
        elif path.endswith(".parquet"):
            return pl.scan_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

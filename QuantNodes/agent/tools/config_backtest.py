# coding=utf-8
"""
配置驱动的回测工具

接受 YAML 配置，通过 ConfigBacktestRunner 直接执行回测。
"""

from typing import Any, Dict, Optional
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
            import polars as pl
            from QuantNodes.backtest.config_runner import ConfigBacktestRunner

            runner = ConfigBacktestRunner()
            bt_result = runner.run(strategy_config, data)

            # 6. 格式化返回结果
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

            # 7. 附加配置信息
            result["config_info"] = {
                "name": strategy_config.name,
                "description": strategy_config.description,
                "factors": len(strategy_config.factors),
                "operations": len(strategy_config.operations),
                "composites": len(strategy_config.composite),
                "has_backtest": strategy_config.backtest is not None,
            }

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
        """加载数据为 Polars LazyFrame"""
        import polars as pl

        # 优先使用 data_path 参数
        if data_path:
            return self._read_data_file(data_path)

        # 使用 config.data.path
        if config.data and config.data.path:
            return self._read_data_file(config.data.path)

        raise ValueError("No data path provided. Set data_path or config.data.path")

    def _read_data_file(self, path: str):
        """读取数据文件"""
        import polars as pl

        if path.endswith(".csv"):
            return pl.scan_csv(path)
        elif path.endswith(".parquet"):
            return pl.scan_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

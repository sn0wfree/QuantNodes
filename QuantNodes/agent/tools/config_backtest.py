# coding=utf-8
"""
配置驱动的回测工具

接受 YAML 配置，自动生成代码并执行回测。
"""

from typing import Any, Dict, Optional
import yaml

from QuantNodes.agent.tools.base import Tool


class ConfigBacktestTool(Tool):
    """配置驱动的回测工具
    
    通过 YAML 配置文件定义策略，自动生成代码并执行回测。
    
    工作流程:
    1. 加载 YAML 配置
    2. 检查算子覆盖度
    3. 生成 Python 代码
    4. 调用 BacktestTool 执行回测
    5. 返回结果
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
        start_date: str = None,
        end_date: str = None,
        initial_cash: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        result = {
            "status": "success",
            "summary": {},
            "config_info": {},
            "generated_code": "",
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
            
            # 4. 生成代码
            from QuantNodes.agent.config.generator import ConfigCodeGenerator
            generator = ConfigCodeGenerator()
            code = generator.generate(strategy_config)
            result["generated_code"] = code
            
            # 5. 调用 BacktestTool
            from QuantNodes.agent.tools.backtest import BacktestTool
            backtest_tool = BacktestTool()
            
            bt_params = {
                "pipeline_code": code,
            }
            
            if strategy_config.backtest:
                bt_params["start_date"] = strategy_config.backtest.start_date
                bt_params["end_date"] = strategy_config.backtest.end_date
                bt_params["initial_cash"] = strategy_config.backtest.initial_cash
                bt_params["commission"] = strategy_config.backtest.commission
            
            bt_result = await backtest_tool.execute(**bt_params)
            
            # 6. 合并结果
            result["status"] = bt_result.get("status", "error")
            result["summary"] = bt_result.get("summary", {})
            result["errors"] = bt_result.get("errors", [])
            
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

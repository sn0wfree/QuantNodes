# coding=utf-8
"""
配置加载器

解析 YAML 配置文件为 StrategyConfig 对象。
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from .types import (
    StrategyConfig,
    FactorConfig,
    OperationConfig,
    CompositeConfig,
    BacktestConfig,
    ValidationConfig,
    DataConfig,
    OutputConfig,
    CoverageReport,
)


class ConfigLoader:
    """YAML配置加载器"""
    
    def __init__(self, working_dir: str = "."):
        self.working_dir = Path(working_dir)
        self._config: Optional[StrategyConfig] = None
    
    def load(self, path: str) -> StrategyConfig:
        """加载YAML配置文件
        
        Args:
            path: 配置文件路径
        
        Returns:
            StrategyConfig 对象
        """
        config_path = self.working_dir / path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        self._config = self._parse(data)
        return self._config
    
    def _parse(self, data: dict) -> StrategyConfig:
        """解析配置字典"""
        config = StrategyConfig(
            version=data.get("version", "1.0"),
            name=data.get("name", ""),
            description=data.get("description", ""),
        )
        
        # 数据源配置
        if "data" in data:
            d = data["data"]
            config.data = DataConfig(
                source=d.get("source", "csv"),
                path=d.get("path", ""),
                columns=d.get("columns", []),
                date_column=d.get("date_column", "date"),
                code_column=d.get("code_column", "code"),
            )
        
        # 因子定义
        config.factors = [
            FactorConfig(
                name=f["name"],
                expr=f.get("expr", ""),
                description=f.get("description", "")
            )
            for f in data.get("factors", [])
        ]
        
        # 运算配置
        config.operations = [
            OperationConfig(
                type=op["type"],
                name=op["name"],
                category=op["category"],
                inputs=op.get("inputs", []),
                params=op.get("params", {})
            )
            for op in data.get("operations", [])
        ]
        
        # 组合因子
        config.composite = [
            CompositeConfig(
                name=c["name"],
                formula=c.get("formula", ""),
                weights=c.get("weights"),
                normalize=c.get("normalize", False),
                winsorize=c.get("winsorize")
            )
            for c in data.get("composite", [])
        ]
        
        # 回测配置
        if "backtest" in data:
            bt = data["backtest"]
            config.backtest = BacktestConfig(
                start_date=bt.get("start_date", ""),
                end_date=bt.get("end_date", ""),
                initial_cash=bt.get("initial_cash", 1000000),
                commission=bt.get("commission", 0.001),
                slippage=bt.get("slippage", 0.001),
                universe=bt.get("universe", "A_stock"),
                signals=bt.get("signals", {}),
                positions=bt.get("positions", {}),
            )
        
        # 验证配置
        if "validation" in data:
            v = data["validation"]
            config.validation = ValidationConfig(
                run_tests=v.get("run_tests", True),
                test_files=v.get("test_files", []),
                metrics=v.get("metrics", {}),
                custom_operators=v.get("custom_operators", []),
            )
        
        # 输出配置
        if "output" in data:
            o = data["output"]
            config.output = OutputConfig(
                format=o.get("format", "parquet"),
                path=o.get("path", "outputs/result.parquet"),
                save_signals=o.get("save_signals", True),
                save_positions=o.get("save_positions", True),
                save_equity_curve=o.get("save_equity_curve", True),
            )
        
        return config
    
    def check_coverage(self, config: StrategyConfig) -> CoverageReport:
        """检查配置覆盖度
        
        Args:
            config: 策略配置
        
        Returns:
            CoverageReport 对象
        """
        covered = []
        unresolved = []
        
        # 检查因子定义
        for factor in config.factors:
            if factor.expr:
                covered.append(f"factor:{factor.name}")
            else:
                unresolved.append(f"factor:{factor.name}")
        
        # 检查算子
        for op in config.operations:
            category = op.category
            
            # 检查是否在注册表中
            if category in _OPERATOR_REGISTRY:
                covered.append(f"op:{category}")
            else:
                unresolved.append(f"op:{category}")
        
        return CoverageReport(covered=covered, unresolved=unresolved)
    
    def get_config(self) -> Optional[StrategyConfig]:
        """获取当前配置"""
        return self._config
    
    def to_yaml(self, config: StrategyConfig, path: str) -> None:
        """导出配置为YAML
        
        Args:
            config: 策略配置
            path: 输出路径
        """
        data = {
            "version": config.version,
            "name": config.name,
            "description": config.description,
        }
        
        if config.data:
            data["data"] = {
                "source": config.data.source,
                "path": config.data.path,
                "columns": config.data.columns,
                "date_column": config.data.date_column,
                "code_column": config.data.code_column,
            }
        
        data["factors"] = [
            {
                "name": f.name,
                "expr": f.expr,
                "description": f.description,
            }
            for f in config.factors
        ]
        
        data["operations"] = [
            {
                "type": op.type,
                "name": op.name,
                "category": op.category,
                "inputs": op.inputs,
                "params": op.params,
            }
            for op in config.operations
        ]
        
        data["composite"] = [
            {
                "name": c.name,
                "formula": c.formula,
                "weights": c.weights,
                "normalize": c.normalize,
                "winsorize": c.winsorize,
            }
            for c in config.composite
        ]
        
        if config.backtest:
            data["backtest"] = {
                "start_date": config.backtest.start_date,
                "end_date": config.backtest.end_date,
                "initial_cash": config.backtest.initial_cash,
                "commission": config.backtest.commission,
                "slippage": config.backtest.slippage,
            }
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)


# 简单算子注册表
_OPERATOR_REGISTRY = {
    # 时间序列算子
    "ts_mean": "time_series",
    "ts_std": "time_series",
    "ts_max": "time_series",
    "ts_min": "time_series",
    "ts_sum": "time_series",
    "ts_median": "time_series",
    "ts_corr": "time_series",
    "ts_cov": "time_series",
    "ts_rank": "time_series",
    "ts_delta": "time_series",
    "ts_pct_change": "time_series",
    "ts_lag": "time_series",
    # 截面算子
    "rank": "section",
    "zscore": "section",
    "winsorize": "section",
    "neutralize": "section",
    "scale": "section",
    "percentile": "section",
    # 算术算子
    "add": "math",
    "sub": "math",
    "mul": "math",
    "div": "math",
    "log": "math",
    "abs": "math",
    "pow": "math",
    # 组合算子
    "weighted_sum": "composite",
    "weighted_avg": "composite",
    "max": "composite",
    "min": "composite",
    "blend": "composite",
}


def load_config(path: str) -> StrategyConfig:
    """便捷加载函数
    
    Args:
        path: 配置文件路径
    
    Returns:
        StrategyConfig 对象
    """
    loader = ConfigLoader()
    return loader.load(path)
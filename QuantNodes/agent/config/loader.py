# coding=utf-8
"""
配置加载器

解析 YAML 配置文件为 StrategyConfig 对象。
"""

from __future__ import annotations

import re
import importlib.util
import warnings
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
from QuantNodes.factor_node.factor_functions import list_operators as _list_operators
from QuantNodes.factor_node.factor_functions import get_operator as _get_operator

# executor category → registry category 映射
_CATEGORY_MAP = {
    "time_series": "time",
    "section": "section",
    "math": "point",
    "composite": "point",
}


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
    
    def _preload_custom_operators(self, custom_operators: list) -> None:
        """预加载自定义算子到 registry
        
        在 check_coverage 之前调用，确保自定义算子被识别。
        
        Args:
            custom_operators: ValidationConfig.custom_operators 列表
        """
        if not custom_operators:
            return
        
        from QuantNodes.factor_node.factor_functions import register_operator
        
        for entry in custom_operators:
            if isinstance(entry, str):
                source_path = entry
                category = "point"
                functions = None
            else:
                source_path = entry.get("source", "")
                category = entry.get("category", "point")
                functions = entry.get("functions")
            
            if not source_path:
                continue
            
            try:
                spec = importlib.util.spec_from_file_location("custom_ops", source_path)
                if spec is None or spec.loader is None:
                    warnings.warn(f"无法加载自定义算子文件: {source_path}")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                warnings.warn(f"加载自定义算子文件失败 {source_path}: {e}")
                continue
            
            for name in dir(module):
                if name.startswith("_"):
                    continue
                if functions and name not in functions:
                    continue
                if not functions and not name.startswith("custom_"):
                    continue
                
                func = getattr(module, name)
                if not callable(func):
                    continue
                
                register_operator(_CATEGORY_MAP.get(category, "point"), name=name)(func)
    
    def check_coverage(self, config: StrategyConfig) -> CoverageReport:
        """检查配置覆盖度
        
        使用 factor_functions 的真实注册表检查算子是否存在。
        会先加载 custom_operators 以确保自定义算子被识别。
        
        Args:
            config: 策略配置
        
        Returns:
            CoverageReport 对象
        """
        # 预加载自定义算子
        self._preload_custom_operators(config.validation.custom_operators)
        
        covered = []
        unresolved = []
        
        # 获取所有已注册的算子名称
        all_operators = set(_list_operators())
        
        # 已定义的列名（factors + operations 的输出）
        defined_names = set()
        for factor in config.factors:
            defined_names.add(factor.name)
        for op in config.operations:
            defined_names.add(op.name)
        
        # 检查因子定义
        for factor in config.factors:
            if factor.expr:
                covered.append("factor:%s" % factor.name)
            else:
                unresolved.append("factor:%s" % factor.name)
        
        # 检查算子 - 使用 factor_functions 真实注册表
        for op in config.operations:
            category = op.category
            
            if category in all_operators:
                covered.append("op:%s" % category)
            else:
                unresolved.append("op:%s" % category)
        
        # 检查组合因子公式
        for comp in config.composite:
            if not comp.formula:
                unresolved.append("composite:%s" % comp.name)
                continue
            
            covered.append("composite:%s" % comp.name)
            
            # 提取公式中调用的函数名
            func_names = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', comp.formula)
            for fn in func_names:
                if fn not in all_operators and fn not in defined_names:
                    unresolved.append("composite:%s:unknown_func:%s" % (comp.name, fn))
            
            # 提取公式中引用的标识符（非函数调用的）
            # 移除函数调用部分后，提取剩余的标识符
            formula_no_funcs = re.sub(r'\b\w+\s*\([^)]*\)', '', comp.formula)
            ref_names = re.findall(r'\b([a-zA-Z_]\w*)\b', formula_no_funcs)
            for rn in ref_names:
                # 跳过 Python 关键字和数字
                if rn in ('true', 'false', 'null', 'and', 'or', 'not', 'None'):
                    continue
                if rn not in defined_names and rn not in all_operators:
                    unresolved.append("composite:%s:unknown_ref:%s" % (comp.name, rn))
        
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
            if config.backtest.universe != "A_stock":
                data["backtest"]["universe"] = config.backtest.universe
            if config.backtest.signals:
                data["backtest"]["signals"] = config.backtest.signals
            if config.backtest.positions:
                data["backtest"]["positions"] = config.backtest.positions

        if config.validation:
            v = config.validation
            data["validation"] = {
                "run_tests": v.run_tests,
            }
            if v.test_files:
                data["validation"]["test_files"] = v.test_files
            if v.metrics:
                data["validation"]["metrics"] = v.metrics
            if v.custom_operators:
                data["validation"]["custom_operators"] = v.custom_operators

        if config.output:
            data["output"] = {
                "format": config.output.format,
                "path": config.output.path,
                "save_signals": config.output.save_signals,
                "save_positions": config.output.save_positions,
                "save_equity_curve": config.output.save_equity_curve,
            }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

def load_config(path: str) -> StrategyConfig:
    """便捷加载函数
    
    Args:
        path: 配置文件路径
    
    Returns:
        StrategyConfig 对象
    """
    loader = ConfigLoader()
    return loader.load(path)
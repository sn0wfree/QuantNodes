# coding=utf-8
"""
配置类型定义

定义策略配置的数据类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import date, datetime


@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    expr: str = ""
    description: str = ""


@dataclass
class OperationConfig:
    """运算配置"""
    type: str  # time_series / section / math / composite
    name: str
    category: str  # 具体算子名称
    inputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeConfig:
    """组合因子配置"""
    name: str
    formula: str = ""
    weights: Optional[List[float]] = None
    normalize: bool = False
    winsorize: Optional[Dict[str, float]] = None


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_cash: float = 1000000
    commission: float = 0.001
    slippage: float = 0.001
    universe: str = "A_stock"
    signals: Dict[str, Any] = field(default_factory=dict)
    positions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """验证配置"""
    run_tests: bool = True
    test_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    custom_operators: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """数据源配置"""
    source: str = "csv"
    path: str = ""
    columns: List[str] = field(default_factory=list)
    date_column: str = "date"
    code_column: str = "code"


@dataclass
class OutputConfig:
    """输出配置"""
    format: str = "parquet"
    path: str = "outputs/result.parquet"
    save_signals: bool = True
    save_positions: bool = True
    save_equity_curve: bool = True


@dataclass
class StrategyConfig:
    """策略配置"""
    version: str = "1.0"
    name: str = ""
    description: str = ""
    data: Optional[DataConfig] = None
    factors: List[FactorConfig] = field(default_factory=list)
    operations: List[OperationConfig] = field(default_factory=list)
    composite: List[CompositeConfig] = field(default_factory=list)
    backtest: Optional[BacktestConfig] = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: Optional[OutputConfig] = None


@dataclass
class CoverageReport:
    """配置覆盖报告"""
    covered: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        return len(self.unresolved) == 0


@dataclass
class ExecutionResult:
    """执行结果"""
    status: str  # success / error
    factors: Dict[str, Any] = field(default_factory=dict)
    backtest: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Any = None  # 计算后的 LazyFrame
    
    @property
    def is_success(self) -> bool:
        return self.status == "success"
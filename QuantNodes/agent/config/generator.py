# coding=utf-8
"""
配置代码生成器

将 StrategyConfig 转换为可执行的 Python 代码。
"""

from __future__ import annotations

from typing import Optional
from .types import StrategyConfig


class ConfigCodeGenerator:
    """将 StrategyConfig 转换为可执行的 Python 代码
    
    生成的代码可以直接传递给 BacktestTool 执行。
    """
    
    def __init__(self):
        pass
    
    def generate(self, config: StrategyConfig) -> str:
        """生成完整的回测代码
        
        Args:
            config: 策略配置
        
        Returns:
            可执行的 Python 代码字符串
        """
        code_parts = []
        
        # 1. 生成 import 语句
        code_parts.append(self._generate_imports())
        
        # 2. 生成数据加载代码
        code_parts.append(self._generate_data_loading(config))
        
        # 3. 生成因子计算代码
        code_parts.append(self._generate_factor_calculation(config))
        
        # 4. 生成信号生成代码
        code_parts.append(self._generate_signal_generation(config))
        
        # 5. 生成策略节点定义
        code_parts.append(self._generate_strategy_class(config))
        
        # 6. 生成回测执行代码
        code_parts.append(self._generate_backtest_execution(config))
        
        return "\n\n".join(code_parts)
    
    def _generate_imports(self) -> str:
        """生成 import 语句"""
        return """import pandas as pd
import numpy as np
from QuantNodes.backtest.strategy_node import StrategyNode, Signal, OrdersResult
from QuantNodes.backtest.broker_node import SimulatedBrokerNode
from QuantNodes.backtest.risk_node import PositionLimitRiskNode"""
    
    def _generate_data_loading(self, config: StrategyConfig) -> str:
        """生成数据加载代码"""
        if config.data and config.data.path:
            path = config.data.path
            if path.endswith('.csv'):
                return f"""# 加载数据
quote_data = pd.read_csv("{path}")"""
            elif path.endswith('.parquet'):
                return f"""# 加载数据
quote_data = pd.read_parquet("{path}")"""
        
        return """# 加载数据 (请修改为实际数据路径)
# quote_data = pd.read_csv("data/stock_data.csv")"""
    
    def _generate_factor_calculation(self, config: StrategyConfig) -> str:
        """生成因子计算代码"""
        lines = ["# 计算因子"]
        
        for factor in config.factors:
            expr = factor.expr
            # 转换 Polars 风格的表达式到 Pandas 风格
            pandas_expr = self._convert_to_pandas_expr(expr)
            lines.append(f'quote_data["{factor.name}"] = {pandas_expr}')
        
        for op in config.operations:
            if op.type == "time_series":
                pandas_expr = self._convert_ts_operator(op)
                if pandas_expr:
                    lines.append(f'quote_data["{op.name}"] = {pandas_expr}')
            elif op.type == "section":
                pandas_expr = self._convert_section_operator(op)
                if pandas_expr:
                    lines.append(f'quote_data["{op.name}"] = {pandas_expr}')
        
        for comp in config.composite:
            pandas_expr = self._convert_to_pandas_expr(comp.formula)
            lines.append(f'quote_data["{comp.name}"] = {pandas_expr}')
        
        return "\n".join(lines)
    
    def _generate_signal_generation(self, config: StrategyConfig) -> str:
        """生成信号生成代码"""
        if not config.backtest:
            return "# 信号生成 (未配置回测参数)"
        
        bt = config.backtest
        buy_threshold = bt.signals.get("buy_threshold", 0.05)
        sell_threshold = bt.signals.get("sell_threshold", -0.03)
        
        # 确定信号列名
        signal_col = config.composite[-1].name if config.composite else \
                    config.operations[-1].name if config.operations else \
                    config.factors[-1].name if config.factors else None
        
        if not signal_col:
            return "# 无法确定信号列"
        
        lines = [
            "# 生成交易信号",
            f'quote_data["signal"] = 0',
            f'quote_data.loc[quote_data["{signal_col}"] > {buy_threshold}, "signal"] = 1',
            f'quote_data.loc[quote_data["{signal_col}"] < {sell_threshold}, "signal"] = -1',
        ]
        
        return "\n".join(lines)
    
    def _generate_strategy_class(self, config: StrategyConfig) -> str:
        """生成策略节点类"""
        return '''# 策略节点定义
class ConfigStrategy(StrategyNode):
    """基于配置生成的策略"""
    
    def _generate_signals(self, input_data, **kwargs):
        signals = []
        for _, row in input_data.iterrows():
            sig = row.get("signal", 0)
            if sig == 1:
                signals.append(Signal(
                    code=row.get("code", ""),
                    signal_type="buy",
                    strength=abs(row.get("alpha", 0.5)),
                    date=str(row.get("date", ""))
                ))
            elif sig == -1:
                signals.append(Signal(
                    code=row.get("code", ""),
                    signal_type="sell",
                    strength=abs(row.get("alpha", 0.5)),
                    date=str(row.get("date", ""))
                ))
        return signals

strategy = ConfigStrategy(config={})'''
    
    def _generate_backtest_execution(self, config: StrategyConfig) -> str:
        """生成回测执行代码"""
        lines = ["# 创建 broker"]
        
        cash = 1000000
        commission = 0.001
        
        if config.backtest:
            cash = config.backtest.initial_cash
            commission = config.backtest.commission
        
        lines.append(f'broker = SimulatedBrokerNode(config={{"cash": {cash}, "commission": {commission}}})')
        
        # 风控节点
        max_positions = 10
        if config.backtest and config.backtest.positions:
            max_positions = config.backtest.positions.get("max_positions", 10)
        
        lines.append(f'risk_nodes = [PositionLimitRiskNode(config={{"max_position": {max_positions}}})]')
        
        return "\n".join(lines)
    
    def _convert_to_pandas_expr(self, expr: str) -> str:
        """将 Polars 风格表达式转换为 Pandas 风格"""
        import re
        
        # 替换 Polars 函数调用为 Pandas 等价物
        result = expr
        
        # ts_lag(col, n) -> col.shift(n)
        result = re.sub(
            r'ts_lag\((\w+),\s*(\d+)\)',
            r'\1.shift(\2)',
            result
        )
        
        # ts_mean(col, n) -> col.rolling(n).mean()
        result = re.sub(
            r'ts_mean\((\w+),\s*(\d+)\)',
            r'\1.rolling(\2).mean()',
            result
        )
        
        # ts_std(col, n) -> col.rolling(n).std()
        result = re.sub(
            r'ts_std\((\w+),\s*(\d+)\)',
            r'\1.rolling(\2).std()',
            result
        )
        
        # ts_max(col, n) -> col.rolling(n).max()
        result = re.sub(
            r'ts_max\((\w+),\s*(\d+)\)',
            r'\1.rolling(\2).max()',
            result
        )
        
        # ts_min(col, n) -> col.rolling(n).min()
        result = re.sub(
            r'ts_min\((\w+),\s*(\d+)\)',
            r'\1.rolling(\2).min()',
            result
        )
        
        # ts_sum(col, n) -> col.rolling(n).sum()
        result = re.sub(
            r'ts_sum\((\w+),\s*(\d+)\)',
            r'\1.rolling(\2).sum()',
            result
        )
        
        # ts_rank(col, n) -> col.rolling(n).rank()
        result = re.sub(
            r'ts_rank\((\w+),\s*(\d+)\)',
            r'\1.rolling(\2).rank()',
            result
        )
        
        # ts_delta(col, n) -> col.diff(n)
        result = re.sub(
            r'ts_delta\((\w+),\s*(\d+)\)',
            r'\1.diff(\2)',
            result
        )
        
        # ts_pct_change(col, n) -> col.pct_change(n)
        result = re.sub(
            r'ts_pct_change\((\w+),\s*(\d+)\)',
            r'\1.pct_change(\2)',
            result
        )
        
        # rank(col) -> col.rank(pct=True)
        result = re.sub(
            r'rank\((\w+)\)',
            r'\1.rank(pct=True)',
            result
        )
        
        return result
    
    def _convert_ts_operator(self, op) -> Optional[str]:
        """转换时间序列算子为 Pandas 表达式"""
        if not op.inputs:
            return None
        
        col = op.inputs[0]
        category = op.category
        window = op.params.get("window", 20)
        periods = op.params.get("periods", 1)
        
        if category == "ts_mean":
            return f'{col}.rolling({window}).mean()'
        elif category == "ts_std":
            return f'{col}.rolling({window}).std()'
        elif category == "ts_max":
            return f'{col}.rolling({window}).max()'
        elif category == "ts_min":
            return f'{col}.rolling({window}).min()'
        elif category == "ts_sum":
            return f'{col}.rolling({window}).sum()'
        elif category == "ts_rank":
            return f'{col}.rolling({window}).rank()'
        elif category == "ts_delta":
            return f'{col}.diff({periods})'
        elif category == "ts_pct_change":
            return f'{col}.pct_change({periods})'
        elif category == "ts_lag":
            return f'{col}.shift({periods})'
        
        return None
    
    def _convert_section_operator(self, op) -> Optional[str]:
        """转换截面算子为 Pandas 表达式"""
        if not op.inputs:
            return None
        
        col = op.inputs[0]
        category = op.category
        
        if category == "rank":
            return f'{col}.rank(pct=True)'
        elif category == "zscore":
            return f'({col} - {col}.mean()) / {col}.std()'
        elif category == "scale":
            return f'{col} / {col}.abs().max()'
        
        return None

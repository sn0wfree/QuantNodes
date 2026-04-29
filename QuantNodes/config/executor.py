# coding=utf-8
"""
配置执行器

执行策略配置，生成 Polars 表达式并计算。
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import polars as pl

from .types import StrategyConfig, ExecutionResult
from ..operators import ts, sec, math, composite


class ConfigExecutor:
    """配置执行器"""
    
    def __init__(self):
        self._expressions: Dict[str, pl.Expr] = {}
        self._cache: Dict[str, Any] = {}
    
    def run(
        self,
        config: StrategyConfig,
        data: pl.LazyFrame
    ) -> ExecutionResult:
        """执行配置
        
        Args:
            config: 策略配置
            data: 数据 LazyFrame
        
        Returns:
            ExecutionResult 对象
        """
        result = ExecutionResult(status="success")
        
        try:
            # 1. 生成因子表达式
            for factor in config.factors:
                expr = self._parse_expr(factor.expr)
                self._expressions[factor.name] = expr
                result.factors[factor.name] = expr
            
            # 2. 执行运算
            for op in config.operations:
                expr = self._apply_operator(op)
                self._expressions[op.name] = expr
                result.factors[op.name] = expr
            
            # 3. 计算组合因子
            for comp in config.composite:
                expr = self._parse_expr(comp.formula)
                self._expressions[comp.name] = expr
                result.factors[comp.name] = expr
            
            # 4. 生成计算计划
            self._execute_plan(data, result)
            
        except Exception as e:
            result.status = "error"
            result.errors.append(str(e))
        
        return result
    
    def run_backtest(
        self,
        config: StrategyConfig,
        data: pl.LazyFrame
    ) -> ExecutionResult:
        """执行回测
        
        Args:
            config: 策略配置
            data: 数据 LazyFrame
        
        Returns:
            ExecutionResult 对象
        """
        result = self.run(config, data)
        
        if config.backtest is None:
            return result
        
        bt = config.backtest
        
        # 筛选日期
        start_parts = list(map(int, bt.start_date.split("-")))
        end_parts = list(map(int, bt.end_date.split("-")))
        data = data.filter(
            pl.col("date").str.to_date() >= pl.date(start_parts[0], start_parts[1], start_parts[2])
        ).filter(
            pl.col("date").str.to_date() <= pl.date(end_parts[0], end_parts[1], end_parts[2])
        )
        
        # 计算信号 (取最后一个因子作为信号)
        signal_name = config.composite[-1].name if config.composite else \
                   config.operations[-1].name if config.operations else \
                   config.factors[-1].name if config.factors else None
        
        if signal_name:
            expr = self._expressions.get(signal_name)
            if expr:
                # 生成交易信号
                data = data.with_columns([
                    pl.when(expr > 0.05).then(1)
                    .when(expr < -0.03).then(-1)
                    .otherwise(0).alias("signal")
                ])
                
                result.backtest = {
                    "signals": data.select("date", "code", "signal"),
                }
        
        return result
    
    def _parse_expr(self, expr_str: str) -> pl.Expr:
        """解析表达式字符串
        
        简化实现：��持基本列引用和方法链
        """
        # 处理列引用
        result = pl.col(expr_str)
        
        # 处理方法链 (简化)
        # TODO: 实现完整解析器
        return result
    
    def _apply_operator(self, op) -> pl.Expr:
        """应用算子"""
        op_type = op.type
        category = op.category
        inputs = op.inputs
        params = op.params
        
        # 获取输入表达式
        input_exprs = []
        for name in inputs:
            if name in self._expressions:
                input_exprs.append(self._expressions[name])
            else:
                input_exprs.append(pl.col(name))
        
        if not input_exprs:
            return pl.col(inputs[0]) if inputs else pl.lit(0)
        
        # 根据类型选择算子
        if op_type == "time_series":
            return self._apply_ts_operator(category, input_exprs[0], params)
        elif op_type == "section":
            return self._apply_sec_operator(category, input_exprs[0], params)
        elif op_type == "math":
            return self._apply_math_operator(category, input_exprs[0], params)
        elif op_type == "composite":
            return self._apply_composite_operator(category, input_exprs, params)
        
        return input_exprs[0]
    
    def _apply_ts_operator(
        self,
        category: str,
        expr: pl.Expr,
        params: Dict[str, Any]
    ) -> pl.Expr:
        """应用时间序列算子"""
        window = params.get("window", 20)
        
        if category == "ts_mean":
            return ts.ts_mean(expr, window)
        elif category == "ts_std":
            return ts.ts_std(expr, window)
        elif category == "ts_max":
            return ts.ts_max(expr, window)
        elif category == "ts_min":
            return ts.ts_min(expr, window)
        elif category == "ts_sum":
            return ts.ts_sum(expr, window)
        elif category == "ts_rank":
            return ts.ts_rank(expr, window)
        elif category == "ts_delta":
            return ts.ts_delta(expr, params.get("periods", 1))
        elif category == "ts_pct_change":
            return ts.ts_pct_change(expr, params.get("periods", 1))
        elif category == "ts_corr":
            return expr  # 需要第二个输入
        elif category == "ts_lag":
            return ts.ts_lag(expr, params.get("periods", 1))
        
        return expr
    
    def _apply_sec_operator(
        self,
        category: str,
        expr: pl.Expr,
        params: Dict[str, Any]
    ) -> pl.Expr:
        """应用截面算子"""
        if category == "rank":
            return sec.rank(expr)
        elif category == "zscore":
            return sec.zscore(expr)
        elif category == "winsorize":
            return sec.winsorize(
                expr,
                params.get("lower", 0.01),
                params.get("upper", 0.01)
            )
        elif category == "neutralize":
            return sec.neutralize_market(expr)
        elif category == "scale":
            return sec.scale(expr)
        elif category == "percentile":
            return sec.percentile(expr)
        
        return expr
    
    def _apply_math_operator(
        self,
        category: str,
        expr: pl.Expr,
        params: Dict[str, Any]
    ) -> pl.Expr:
        """应用数学算子"""
        value = params.get("value", 1.0)
        
        if category == "add":
            return math.add(expr, value)
        elif category == "sub":
            return math.sub(expr, value)
        elif category == "mul":
            return math.mul(expr, value)
        elif category == "div":
            return math.div(expr, value)
        elif category == "log":
            return math.log(expr)
        elif category == "abs":
            return math.abs(expr)
        elif category == "pow":
            return math.pow(expr, params.get("exponent", 2))
        
        return expr
    
    def _apply_composite_operator(
        self,
        category: str,
        exprs: List[pl.Expr],
        params: Dict[str, Any]
    ) -> pl.Expr:
        """应用组合算子"""
        if category == "weighted_sum":
            default_weights = [1.0 / len(exprs)] * len(exprs)
            weights = params.get("weights", default_weights)
            return composite.weighted_sum(exprs, weights)
        elif category == "weighted_avg":
            return composite.weighted_avg(exprs)
        elif category == "max":
            return composite.max(exprs)
        elif category == "min":
            return composite.min(exprs)
        elif category == "blend":
            alpha = params.get("alpha", 0.5)
            return composite.blend(exprs[0], exprs[1], alpha)
        
        return exprs[0] if exprs else pl.lit(0)
    
    def _execute_plan(
        self,
        data: pl.LazyFrame,
        result: ExecutionResult
    ) -> None:
        """执行计算计划"""
        # 收集需要计算的列
        select_cols = []
        for name, expr in self._expressions.items():
            select_cols.append(expr.alias(name))
        
        if select_cols:
            result.data = data.select(select_cols)
    
    def get_expressions(self) -> Dict[str, pl.Expr]:
        """获取生成的表达式"""
        return self._expressions
    
    def compile(
        self,
        config: StrategyConfig,
        data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """编译配置为 LazyFrame
        
        Args:
            config: 策略配置
            data: 数据 LazyFrame
        
        Returns:
            计算后的 LazyFrame
        """
        result = self.run(config, data)
        
        if result.is_success and hasattr(result, "data"):
            return result.data
        
        return data
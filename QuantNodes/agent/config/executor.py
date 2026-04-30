# coding=utf-8
"""
配置执行器

执行策略配置，生成 Polars 表达式并计算。
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import polars as pl

from .types import StrategyConfig, ExecutionResult
from QuantNodes.operators import ts, sec, math, composite


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
        
        try:
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
                    # 使用配置中的信号阈值，或使用默认值
                    long_threshold = bt.signals.get("long_threshold", 0.05)
                    short_threshold = bt.signals.get("short_threshold", -0.03)
                    
                    # 生成交易信号
                    data = data.with_columns([
                        pl.when(expr > long_threshold).then(1)
                        .when(expr < short_threshold).then(-1)
                        .otherwise(0).alias("signal")
                    ])
                    
                    result.backtest = {
                        "signals": data.select("date", "code", "signal"),
                        "config": {
                            "start_date": bt.start_date,
                            "end_date": bt.end_date,
                            "initial_cash": bt.initial_cash,
                            "commission": bt.commission,
                            "slippage": bt.slippage,
                        }
                    }
        except Exception as e:
            result.warnings.append(f"回测配置解析警告: {str(e)}")
        
        return result
    
    def _parse_expr(self, expr_str: str) -> pl.Expr:
        """解析表达式字符串
        
        支持格式:
        - 简单列引用: "close"
        - 函数调用: "rolling_mean(close, 20)"
        - 方法链: "close.rolling_mean(20)"
        - 组合: "rolling_mean(close, 20) + volume"
        """
        import re
        from QuantNodes.factor_node.factor_functions import get_operator
        
        expr_str = expr_str.strip()
        
        # 尝试作为简单列引用
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', expr_str):
            return pl.col(expr_str)
        
        # 尝试匹配函数调用模式: func_name(arg1, arg2, ...)
        func_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\((.+)\)$', expr_str, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            
            # 查找算子函数
            op_func = get_operator(func_name)
            if op_func is not None:
                # 解析参数
                args, kwargs = self._parse_func_args(args_str)
                # 将第一个参数作为表达式
                if args:
                    first_arg = self._parse_expr(args[0])
                    return op_func(first_arg, **kwargs)
        
        # 尝试匹配方法链模式: expr.method(args)
        method_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\((.+)\)$', expr_str, re.DOTALL)
        if method_match:
            col_name = method_match.group(1)
            method_name = method_match.group(2)
            args_str = method_match.group(3)
            
            op_func = get_operator(method_name)
            if op_func is not None:
                first_arg = pl.col(col_name)
                args, kwargs = self._parse_func_args(args_str)
                return op_func(first_arg, **kwargs)
        
        # 回退到简单列引用
        return pl.col(expr_str)
    
    def _parse_func_args(self, args_str: str):
        """解析函数参数字符串
        
        Returns:
            (positional_args, keyword_args)
        """
        import re
        
        positional = []
        keyword = {}
        
        if not args_str.strip():
            return positional, keyword
        
        # 简单分割（不处理嵌套括号）
        parts = []
        depth = 0
        current = ""
        for ch in args_str:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        
        for part in parts:
            # 检查是否是 keyword=value 格式
            kw_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', part)
            if kw_match:
                key = kw_match.group(1)
                value = self._parse_value(kw_match.group(2).strip())
                keyword[key] = value
            else:
                positional.append(self._parse_value(part))
        
        return positional, keyword
    
    def _parse_value(self, value_str: str):
        """解析单个值"""
        # 尝试解析为数字
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        # 去除引号
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        # 尝试作为列引用
        return pl.col(value_str)
    
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
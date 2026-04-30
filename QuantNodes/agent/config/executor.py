# coding=utf-8
"""
配置执行器

执行策略配置，生成 Polars 表达式并计算。
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import polars as pl

from .types import StrategyConfig, ExecutionResult
from QuantNodes.operators import ts, sec, math, composite


class ExprParser:
    """递归下降表达式解析器
    
    支持:
    - 简单列引用: "close"
    - 数字字面量: "20", "3.14"
    - 函数调用: "rolling_mean(close, 20)"
    - 方法链: "close.rolling_mean(20)"
    - 算术运算: "close / close.shift(20) - 1"
    - 一元运算: "-rank(close_ma_diff)"
    - 括号分组: "(close + volume) / 2"
    """
    
    def __init__(self, executor: 'ConfigExecutor'):
        self.executor = executor
        self._get_operator = None
    
    def _lazy_import(self):
        if self._get_operator is None:
            from QuantNodes.factor_node.factor_functions import get_operator
            self._get_operator = get_operator
    
    def parse(self, expr_str: str) -> pl.Expr:
        """解析表达式字符串为 Polars Expr"""
        self._pos = 0
        self._expr = expr_str.strip()
        result = self._parse_additive()
        return result
    
    def _current(self) -> str:
        self._skip_whitespace()
        if self._pos >= len(self._expr):
            return ''
        return self._expr[self._pos]
    
    def _skip_whitespace(self):
        while self._pos < len(self._expr) and self._expr[self._pos] in ' \t':
            self._pos += 1
    
    def _consume(self, ch: str):
        self._skip_whitespace()
        if self._pos < len(self._expr) and self._expr[self._pos] == ch:
            self._pos += 1
        else:
            raise ValueError(
                f"Expected '{ch}' at position {self._pos}, "
                f"got '{self._expr[self._pos] if self._pos < len(self._expr) else 'EOF'}'"
            )
    
    def _parse_additive(self) -> pl.Expr:
        """加减法: term (('+' | '-') term)*"""
        left = self._parse_multiplicative()
        
        while self._current() in ('+', '-'):
            op = self._current()
            self._pos += 1
            right = self._parse_multiplicative()
            
            if op == '+':
                left = left + right
            else:
                left = left - right
        
        return left
    
    def _parse_multiplicative(self) -> pl.Expr:
        """乘除法: unary (('*' | '/') unary)*"""
        left = self._parse_unary()
        
        while self._current() in ('*', '/'):
            op = self._current()
            self._pos += 1
            right = self._parse_unary()
            
            if op == '*':
                left = left * right
            else:
                left = left / right
        
        return left
    
    def _parse_unary(self) -> pl.Expr:
        """一元运算: ('-' | '+') primary | primary"""
        if self._current() == '-':
            self._pos += 1
            operand = self._parse_primary()
            return pl.lit(0) - operand
        elif self._current() == '+':
            self._pos += 1
            return self._parse_primary()
        return self._parse_primary()
    
    def _parse_primary(self) -> pl.Expr:
        """主项: number | column | function_call | method_chain | '(' expr ')'"""
        self._skip_whitespace()
        
        # 括号表达式
        if self._current() == '(':
            self._consume('(')
            expr = self._parse_additive()
            self._consume(')')
            return expr
        
        # 数字字面量
        if self._current().isdigit() or (self._current() == '.' and self._pos + 1 < len(self._expr) and self._expr[self._pos + 1].isdigit()):
            return self._parse_number()
        
        # 标识符 (列名/函数名/方法名)
        if self._current().isalpha() or self._current() == '_':
            return self._parse_identifier()
        
        raise ValueError(f"Unexpected character '{self._current()}' at position {self._pos}")
    
    def _parse_number(self) -> pl.Expr:
        """解析数字字面量"""
        start = self._pos
        while self._pos < len(self._expr) and (self._expr[self._pos].isdigit() or self._expr[self._pos] == '.'):
            self._pos += 1
        
        num_str = self._expr[start:self._pos]
        if '.' in num_str:
            return pl.lit(float(num_str))
        return pl.lit(int(num_str))
    
    def _parse_identifier(self) -> pl.Expr:
        """解析标识符 (可能是列名、函数调用或方法链)"""
        name = self._parse_name()
        
        self._skip_whitespace()
        
        # 函数调用: name(args)
        if self._current() == '(':
            return self._parse_func_call(name)
        
        # 方法链: name.method(args) 或 name.method
        if self._current() == '.':
            return self._parse_method_chain(name)
        
        # 简单列引用
        return pl.col(name)
    
    def _parse_name(self) -> str:
        """解析标识符名称"""
        start = self._pos
        while self._pos < len(self._expr) and (self._expr[self._pos].isalnum() or self._expr[self._pos] == '_'):
            self._pos += 1
        
        if self._pos == start:
            raise ValueError(f"Expected identifier at position {self._pos}")
        
        return self._expr[start:self._pos]
    
    def _parse_func_call(self, func_name: str) -> pl.Expr:
        """解析函数调用: func_name(arg1, arg2, ...)"""
        self._consume('(')
        
        # 读取原始参数字符串，使用 executor 的 _parse_func_args 解析
        # 这样数字参数会返回 Python int/float，列名会返回 pl.col()
        args_str = self._read_func_args()
        args, kwargs = self.executor._parse_func_args(args_str)
        
        # 查找算子函数
        self._lazy_import()
        op_func = self._get_operator(func_name) if self._get_operator else None
        
        if op_func is not None and args:
            # 将第一个参数转换为表达式（如果还不是的话）
            first_arg = args[0] if isinstance(args[0], pl.Expr) else pl.col(str(args[0]))
            rest_args = args[1:]
            return op_func(first_arg, *rest_args, **kwargs)
        
        # 如果没有找到算子，尝试作为 Polars 方法
        if args:
            first_arg = args[0] if isinstance(args[0], pl.Expr) else pl.col(str(args[0]))
            rest_args = args[1:]
            return getattr(first_arg, func_name)(*rest_args, **kwargs)
        
        return pl.col(func_name)
    
    def _read_func_args(self) -> str:
        """读取函数参数字符串（从当前位置到匹配的右括号，消耗 ')')
        
        注意: 此方法会消耗右括号 ')'，调用后不需要再 consume ')'
        """
        start = self._pos
        depth = 1
        while self._pos < len(self._expr) and depth > 0:
            if self._expr[self._pos] == '(':
                depth += 1
            elif self._expr[self._pos] == ')':
                depth -= 1
            self._pos += 1
        # 返回括号内的内容（不含两端括号）
        return self._expr[start:self._pos - 1]
    
    def _parse_method_chain(self, obj_name: str) -> pl.Expr:
        """解析方法链: obj.method(args) 或 obj.method"""
        self._consume('.')
        method_name = self._parse_name()
        
        self._skip_whitespace()
        
        if self._current() == '(':
            # 方法调用: obj.method(args)
            # 注意: _read_func_args 已经消耗了 ')'
            args_str = self._read_func_args()
            
            self._lazy_import()
            op_func = self._get_operator(method_name) if self._get_operator else None
            
            first_arg = pl.col(obj_name)
            args, kwargs = self.executor._parse_func_args(args_str)
            
            if op_func is not None:
                return op_func(first_arg, *args, **kwargs)
            
            # 回退到 Polars 原生方法
            return getattr(first_arg, method_name)(*args, **kwargs)
        
        # 属性访问: obj.method (不支持，回退到列引用)
        return pl.col(f"{obj_name}.{method_name}")


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
            if bt.start_date:
                start_parts = list(map(int, bt.start_date.split("-")))
                data = data.filter(
                    pl.col("date").str.to_date() >= pl.date(start_parts[0], start_parts[1], start_parts[2])
                )
            if bt.end_date:
                end_parts = list(map(int, bt.end_date.split("-")))
                data = data.filter(
                    pl.col("date").str.to_date() <= pl.date(end_parts[0], end_parts[1], end_parts[2])
                )
            
            # 计算信号 (取最后一个因子作为信号)
            signal_name = None
            if config.composite:
                signal_name = config.composite[-1].name
            elif config.operations:
                signal_name = config.operations[-1].name
            elif config.factors:
                signal_name = config.factors[-1].name
            
            if signal_name is not None:
                expr = self._expressions.get(signal_name)
                if expr is not None:
                    # 兼容两种阈值命名
                    buy_threshold = bt.signals.get("buy_threshold",
                                   bt.signals.get("long_threshold", 0.05))
                    sell_threshold = bt.signals.get("sell_threshold",
                                    bt.signals.get("short_threshold", -0.03))
                    
                    # 生成交易信号
                    data = data.with_columns([
                        pl.when(expr > buy_threshold).then(1)
                        .when(expr < sell_threshold).then(-1)
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
                        },
                        "buy_threshold": buy_threshold,
                        "sell_threshold": sell_threshold,
                    }
        except Exception as e:
            result.warnings.append(f"回测配置解析警告: {str(e)}")
        
        return result
    
    def _parse_expr(self, expr_str: str) -> pl.Expr:
        """解析表达式字符串
        
        使用递归下降解析器，支持:
        - 简单列引用: "close"
        - 函数调用: "rolling_mean(close, 20)"
        - 方法链: "close.rolling_mean(20)"
        - 算术运算: "close / close.shift(20) - 1"
        - 一元运算: "-rank(close_ma_diff)"
        - 括号分组: "(close + volume) / 2"
        """
        parser = ExprParser(self)
        return parser.parse(expr_str)
    
    def _parse_func_args(self, args_str: str) -> Tuple[List, Dict]:
        """解析函数参数字符串
        
        Returns:
            (positional_args, keyword_args)
        """
        import re
        
        positional = []
        keyword = {}
        
        if not args_str.strip():
            return positional, keyword
        
        # 处理嵌套括号的分割
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
        """解析单个值
        
        注意: 对于数字，返回 Python 原生类型（int/float），
        而不是 Polars 表达式。这是为了正确传递参数给算子函数。
        """
        value_str = value_str.strip()
        
        # 去除引号
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # 尝试解析为整数
        try:
            return int(value_str)
        except ValueError:
            pass
        
        # 尝试解析为浮点数
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # 纯数字字符串（可能是字符串格式的数字）
        if value_str.isdigit():
            return int(value_str)
        
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
            return expr
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
        """执行计算计划
        
        保留原始列 (date, code, close等) 并添加计算的因子列。
        """
        # 保留原始列
        original_col_names = data.collect_schema().names()
        original_cols = [pl.col(c) for c in original_col_names]
        
        # 添加计算的因子列
        computed_cols = []
        for name, expr in self._expressions.items():
            computed_cols.append(expr.alias(name))
        
        if computed_cols:
            result.data = data.select(original_cols + computed_cols)
        else:
            result.data = data
    
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

# coding=utf-8
"""
表达式系统 - 支持符号化表达计算逻辑

本模块提供表达式抽象和 DSL 构建器，支持：
1. 运算符重载构建表达式
2. AST 安全解析字符串表达式
3. 序列化/反序列化
4. 向后兼容 lambda 表达式
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


# ============================================================================
# 安全配置
# ============================================================================

ALLOWED_AST_NODES = {
    ast.Expression, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.Attribute, ast.Subscript, ast.Call, ast.Name,
    ast.Constant, ast.Load,
    ast.Gt, ast.Lt, ast.Eq, ast.GtE, ast.LtE, ast.NotEq,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub,
}

FORBIDDEN_METHODS = {
    'eval', 'exec', '__import__', 'compile', 'open', 'system',
    'subprocess', 'os', 'sys', 'builtins',
}


# ============================================================================
# 表达式基类
# ============================================================================

class Expression(ABC):
    """表达式基类，所有计算逻辑的抽象"""

    @abstractmethod
    def evaluate(self, input_data: Any) -> Any:
        """对输入数据执行表达式求值"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Expression':
        """从字典反序列化"""
        expr_type = data.get("type")
        if not expr_type:
            raise ValueError("Missing 'type' in expression data")
        
        # 首先检查基类本身
        if cls.__name__ == expr_type:
            return cls._from_dict_impl(data)
        
        # 然后检查子类
        for subclass in cls.__subclasses__():
            if subclass.__name__ == expr_type:
                return subclass._from_dict_impl(data)
        
        raise ValueError(f"Unknown expression type: {expr_type}")

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'Expression':
        """子类实现的反序列化逻辑"""
        raise NotImplementedError(f"{cls.__name__}._from_dict_impl not implemented")

    @classmethod
    def parse(cls, expr_str: str) -> 'Expression':
        """安全解析字符串表达式"""
        from QuantNodes.core.ast_parser import parse_expression
        return parse_expression(expr_str)

    def __add__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "+", _wrap_expr(other))

    def __radd__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "+", self)

    def __sub__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "-", _wrap_expr(other))

    def __rsub__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "-", self)

    def __mul__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "*", _wrap_expr(other))

    def __rmul__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "*", self)

    def __truediv__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "/", _wrap_expr(other))

    def __rtruediv__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "/", self)

    def __floordiv__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "//", _wrap_expr(other))

    def __rfloordiv__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "//", self)

    def __mod__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "%", _wrap_expr(other))

    def __rmod__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "%", self)

    def __pow__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(self, "**", _wrap_expr(other))

    def __rpow__(self, other: Any) -> 'BinaryOpExpr':
        return BinaryOpExpr(_wrap_expr(other), "**", self)

    def __neg__(self) -> 'UnaryOpExpr':
        return UnaryOpExpr("-", self)

    def __pos__(self) -> 'Expression':
        return self

    def __abs__(self) -> 'UnaryOpExpr':
        return UnaryOpExpr("abs", self)

    def __gt__(self, other: Any) -> 'ComparisonExpr':
        return ComparisonExpr(self, ">", _wrap_expr(other))

    def __ge__(self, other: Any) -> 'ComparisonExpr':
        return ComparisonExpr(self, ">=", _wrap_expr(other))

    def __lt__(self, other: Any) -> 'ComparisonExpr':
        return ComparisonExpr(self, "<", _wrap_expr(other))

    def __le__(self, other: Any) -> 'ComparisonExpr':
        return ComparisonExpr(self, "<=", _wrap_expr(other))

    def __eq__(self, other: Any) -> 'ComparisonExpr':  # type: ignore
        return ComparisonExpr(self, "==", _wrap_expr(other))

    def __ne__(self, other: Any) -> 'ComparisonExpr':  # type: ignore
        return ComparisonExpr(self, "!=", _wrap_expr(other))

    def __and__(self, other: Any) -> 'LogicalOpExpr':
        return LogicalOpExpr("and", self, _wrap_expr(other))

    def __rand__(self, other: Any) -> 'LogicalOpExpr':
        return LogicalOpExpr("and", _wrap_expr(other), self)

    def __or__(self, other: Any) -> 'LogicalOpExpr':
        return LogicalOpExpr("or", self, _wrap_expr(other))

    def __ror__(self, other: Any) -> 'LogicalOpExpr':
        return LogicalOpExpr("or", _wrap_expr(other), self)

    def __invert__(self) -> 'LogicalOpExpr':
        return LogicalOpExpr("not", self)

    def method(self, name: str, *args: Any, **kwargs: Any) -> 'MethodCallExpr':
        """方法调用"""
        wrapped_args = tuple(_wrap_expr(a) for a in args)
        wrapped_kwargs = {k: _wrap_expr(v) for k, v in kwargs.items()}
        return MethodCallExpr(self, name, wrapped_args, wrapped_kwargs)

    def __call__(self, input_data: Any) -> Any:
        """支持直接调用求值"""
        return self.evaluate(input_data)

    def __bool__(self) -> bool:
        """防止 Python 隐式布尔转换（用于 & / | 运算符）"""
        raise TypeError(
            "Expression 对象不能直接转换为布尔值。"
            "如果你在使用 & 或 | 运算符，请确保两边都是 Expression 对象。"
            "如果你想求值表达式，请使用 .evaluate(data) 方法。"
        )


def _wrap_expr(value: Any) -> Expression:
    """将值包装为表达式"""
    if isinstance(value, ExpressionBuilder):
        return value._expr
    if isinstance(value, Expression):
        return value
    return ConstantExpr(value)


# ============================================================================
# 原子表达式
# ============================================================================

class InputExpr(Expression):
    """输入数据本身 - 代表整个 input_data"""

    def evaluate(self, input_data: Any) -> Any:
        return input_data

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "InputExpr"}

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'InputExpr':
        return InputExpr()

    def __repr__(self) -> str:
        return "input"


class ConstantExpr(Expression):
    """常量值表达式"""

    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, input_data: Any) -> Any:
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "ConstantExpr", "value": self.value}

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'ConstantExpr':
        return ConstantExpr(data["value"])

    def __repr__(self) -> str:
        return repr(self.value)


class VariableExpr(Expression):
    """变量/列访问：input_data[name]"""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return input_data[self.name]
        return getattr(input_data, self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "VariableExpr", "name": self.name}

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'VariableExpr':
        return VariableExpr(data["name"])

    def __repr__(self) -> str:
        return f"Cond('{self.name}')"


class AttributeExpr(Expression):
    """属性访问：expr.attr"""

    def __init__(self, expr: Expression, attr: str):
        self.expr = expr
        self.attr = attr

    def evaluate(self, input_data: Any) -> Any:
        obj = self.expr.evaluate(input_data)
        return getattr(obj, self.attr)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "AttributeExpr",
            "expr": self.expr.to_dict(),
            "attr": self.attr,
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'AttributeExpr':
        return AttributeExpr(
            Expression.from_dict(data["expr"]),
            data["attr"],
        )

    def __repr__(self) -> str:
        return f"{self.expr}.{self.attr}"


class SubscriptExpr(Expression):
    """下标访问：expr[key]"""

    def __init__(self, expr: Expression, key: Any):
        self.expr = expr
        self.key = _wrap_expr(key)

    def evaluate(self, input_data: Any) -> Any:
        obj = self.expr.evaluate(input_data)
        key_val = self.key.evaluate(input_data)
        return obj[key_val]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SubscriptExpr",
            "expr": self.expr.to_dict(),
            "key": self.key.to_dict(),
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'SubscriptExpr':
        return SubscriptExpr(
            Expression.from_dict(data["expr"]),
            Expression.from_dict(data["key"]),
        )

    def __repr__(self) -> str:
        return f"{self.expr}[{self.key}]"




# ============================================================================
# 方法调用表达式
# ============================================================================

class MethodCallExpr(Expression):
    """方法调用表达式：expr.method(*args, **kwargs)"""

    def __init__(
        self,
        expr: Expression,
        method: str,
        args: Tuple[Expression, ...],
        kwargs: Dict[str, Expression],
    ):
        if method in FORBIDDEN_METHODS:
            raise ValueError(f"Forbidden method: {method}")
        self.expr = expr
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, input_data: Any) -> Any:
        obj = self.expr.evaluate(input_data)
        func = getattr(obj, self.method)
        args_val = tuple(a.evaluate(input_data) for a in self.args)
        kwargs_val = {k: v.evaluate(input_data) for k, v in self.kwargs.items()}
        return func(*args_val, **kwargs_val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "MethodCallExpr",
            "expr": self.expr.to_dict(),
            "method": self.method,
            "args": [a.to_dict() for a in self.args],
            "kwargs": {k: v.to_dict() for k, v in self.kwargs.items()},
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'MethodCallExpr':
        return MethodCallExpr(
            Expression.from_dict(data["expr"]),
            data["method"],
            tuple(Expression.from_dict(a) for a in data["args"]),
            {k: Expression.from_dict(v) for k, v in data["kwargs"].items()},
        )

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{self.expr}.{self.method}({all_args})"


# ============================================================================
# 运算表达式
# ============================================================================

class BinaryOpExpr(Expression):
    """二元运算表达式"""

    _OP_MAP = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "//": lambda a, b: a // b,
        "%": lambda a, b: a % b,
        "**": lambda a, b: a ** b,
    }

    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op
        self.right = right
        if op not in self._OP_MAP:
            raise ValueError(f"Unknown operator: {op}")

    def evaluate(self, input_data: Any) -> Any:
        left_val = self.left.evaluate(input_data)
        right_val = self.right.evaluate(input_data)
        return self._OP_MAP[self.op](left_val, right_val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "BinaryOpExpr",
            "left": self.left.to_dict(),
            "op": self.op,
            "right": self.right.to_dict(),
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'BinaryOpExpr':
        return BinaryOpExpr(
            Expression.from_dict(data["left"]),
            data["op"],
            Expression.from_dict(data["right"]),
        )

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class UnaryOpExpr(Expression):
    """一元运算表达式"""

    _OP_MAP = {
        "-": lambda x: -x,
        "+": lambda x: x,
        "abs": lambda x: abs(x),
    }

    def __init__(self, op: str, operand: Expression):
        self.op = op
        self.operand = operand
        if op not in self._OP_MAP:
            raise ValueError(f"Unknown operator: {op}")

    def evaluate(self, input_data: Any) -> Any:
        val = self.operand.evaluate(input_data)
        return self._OP_MAP[self.op](val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "UnaryOpExpr",
            "op": self.op,
            "operand": self.operand.to_dict(),
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'UnaryOpExpr':
        return UnaryOpExpr(
            data["op"],
            Expression.from_dict(data["operand"]),
        )

    def __repr__(self) -> str:
        return f"{self.op}({self.operand})"


class ComparisonExpr(Expression):
    """比较运算表达式"""

    _OP_MAP = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op
        self.right = right
        if op not in self._OP_MAP:
            raise ValueError(f"Unknown operator: {op}")

    def evaluate(self, input_data: Any) -> bool:
        left_val = self.left.evaluate(input_data)
        right_val = self.right.evaluate(input_data)
        return self._OP_MAP[self.op](left_val, right_val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ComparisonExpr",
            "left": self.left.to_dict(),
            "op": self.op,
            "right": self.right.to_dict(),
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'ComparisonExpr':
        return ComparisonExpr(
            Expression.from_dict(data["left"]),
            data["op"],
            Expression.from_dict(data["right"]),
        )

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class LogicalOpExpr(Expression):
    """逻辑运算表达式"""

    _OP_MAP = {
        "and": lambda *args: all(args),
        "or": lambda *args: any(args),
        "not": lambda x: not x,
    }

    def __init__(self, op: str, *operands: Expression):
        self.op = op
        self.operands = operands
        if op not in self._OP_MAP:
            raise ValueError(f"Unknown operator: {op}")

    def evaluate(self, input_data: Any) -> bool:
        vals = tuple(op.evaluate(input_data) for op in self.operands)
        return self._OP_MAP[self.op](*vals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LogicalOpExpr",
            "op": self.op,
            "operands": [op.to_dict() for op in self.operands],
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'LogicalOpExpr':
        return LogicalOpExpr(
            data["op"],
            *(Expression.from_dict(op) for op in data["operands"]),
        )

    def __repr__(self) -> str:
        if self.op == "not":
            return f"~{self.operands[0]}"
        return f"({(' ' + self.op + ' ').join(repr(op) for op in self.operands)})"


# ============================================================================
# Lambda 表达式（向后兼容）
# ============================================================================

class LambdaExpression(Expression):
    """包装 Callable，用于向后兼容"""

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def evaluate(self, input_data: Any) -> Any:
        return self.func(input_data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LambdaExpression",
            "name": self.func.__name__ if hasattr(self.func, "__name__") else str(self.func),
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'LambdaExpression':
        raise RuntimeError("LambdaExpression cannot be deserialized")

    def __repr__(self) -> str:
        if hasattr(self.func, "__name__"):
            return f"lambda({self.func.__name__})"
        return f"lambda({str(self.func)})"




# ============================================================================
# DSL 构建器
# ============================================================================

class ExpressionBuilder:
    """
    表达式构建器，提供链式 API

    使用方式：
        >>> Cond('close') > 50           # input_data['close'] > 50
        >>> Cond.attr('metrics').sharpe  # input_data.metrics.sharpe
        >>> Cond('value').mean()         # input_data['value'].mean()
    """

    def __init__(self, expr: Expression):
        self._expr = expr

    def attr(self, name: str) -> 'ExpressionBuilder':
        """按属性名访问"""
        return ExpressionBuilder(AttributeExpr(self._expr, name))

    def method(self, name: str, *args: Any, **kwargs: Any) -> 'ExpressionBuilder':
        """方法调用"""
        return ExpressionBuilder(self._expr.method(name, *args, **kwargs))

    def constant(self, value: Any) -> 'ExpressionBuilder':
        """常量值"""
        return ExpressionBuilder(ConstantExpr(value))

    def __getattr__(self, name: str) -> 'ExpressionBuilder':
        """支持 Cond.metrics 形式的属性访问"""
        return ExpressionBuilder(AttributeExpr(self._expr, name))

    def __getitem__(self, key: Any) -> 'ExpressionBuilder':
        """支持下标访问 Cond['close']"""
        return ExpressionBuilder(SubscriptExpr(self._expr, key))

    # ------------------------------------------------------------------------
    # 运算符转发
    # ------------------------------------------------------------------------

    def __add__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr + other)

    def __radd__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other + self._expr)

    def __sub__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr - other)

    def __rsub__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other - self._expr)

    def __mul__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr * other)

    def __rmul__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other * self._expr)

    def __truediv__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr / other)

    def __rtruediv__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other / self._expr)

    def __floordiv__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr // other)

    def __rfloordiv__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other // self._expr)

    def __mod__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr % other)

    def __rmod__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other % self._expr)

    def __pow__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr ** other)

    def __rpow__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other ** self._expr)

    def __neg__(self) -> 'ExpressionBuilder':
        return ExpressionBuilder(-self._expr)

    def __pos__(self) -> 'ExpressionBuilder':
        return ExpressionBuilder(+self._expr)

    def __abs__(self) -> 'ExpressionBuilder':
        return ExpressionBuilder(abs(self._expr))

    def __gt__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr > other)

    def __ge__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr >= other)

    def __lt__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr < other)

    def __le__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr <= other)

    def __eq__(self, other: Any) -> 'ExpressionBuilder':  # type: ignore
        return ExpressionBuilder(self._expr == other)

    def __ne__(self, other: Any) -> 'ExpressionBuilder':  # type: ignore
        return ExpressionBuilder(self._expr != other)

    def __and__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr & other)

    def __rand__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other & self._expr)

    def __or__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(self._expr | other)

    def __ror__(self, other: Any) -> 'ExpressionBuilder':
        return ExpressionBuilder(other | self._expr)

    def __invert__(self) -> 'ExpressionBuilder':
        return ExpressionBuilder(~self._expr)

    def __repr__(self) -> str:
        return repr(self._expr)

    def evaluate(self, input_data: Any) -> Any:
        return self._expr.evaluate(input_data)

    def to_dict(self) -> Dict[str, Any]:
        return self._expr.to_dict()

    def __call__(self, input_data: Any) -> Any:
        return self._expr.evaluate(input_data)

    def __bool__(self) -> bool:
        """防止 Python 隐式布尔转换"""
        return self._expr.__bool__()


# ============================================================================
# 表达式类注册表 - 用于反序列化
# ============================================================================

_EXPRESSION_CLASSES = {}


def _register_expression_class(cls):
    """注册表达式类用于反序列化"""
    _EXPRESSION_CLASSES[cls.__name__] = cls
    return cls


def _from_dict_redirect(data: Dict[str, Any]) -> Expression:
    """从字典反序列化表达式的实际实现"""
    expr_type = data.get("type")
    if not expr_type:
        raise ValueError("Missing 'type' in expression data")
    
    cls = _EXPRESSION_CLASSES.get(expr_type)
    if cls is None:
        raise ValueError(f"Unknown expression type: {expr_type}")
    
    return cls._from_dict_impl(data)


Expression.from_dict = staticmethod(_from_dict_redirect)

# 注册所有表达式类
_register_expression_class(InputExpr)
_register_expression_class(ConstantExpr)
_register_expression_class(VariableExpr)
_register_expression_class(AttributeExpr)
_register_expression_class(SubscriptExpr)
_register_expression_class(MethodCallExpr)
_register_expression_class(BinaryOpExpr)
_register_expression_class(UnaryOpExpr)
_register_expression_class(ComparisonExpr)
_register_expression_class(LogicalOpExpr)
_register_expression_class(LambdaExpression)

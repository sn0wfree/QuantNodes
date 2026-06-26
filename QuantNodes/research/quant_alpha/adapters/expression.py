# coding=utf-8
"""
adapters/expression.py - 极简 Expression AST（AlphaGen 兼容子集）

AlphaGen (KDD 2023) 使用 token 化的 Expression AST 表示因子公式。
本模块提供极简子集，足以让 PolarsAlphaCalculator 正常工作。

完整的 AlphaGen Expression 包含 ~22 个算子：
    Ref, Add, Sub, Mul, Div, Greater, Less, Neg, Abs, Log, Sign, Sqrt,
    Mean, Std, Var, Skew, Kurt, Max, Min, Sum, Med, Mad,
    Rank, Delta, Quantile, Condition, ...

本子集（11 个）覆盖 AlphaGen 80% 用例：
    - Feature: 引用基础字段
    - Ref: 时序滞后
    - BinaryOp (Add/Sub/Mul/Div/Greater/Less)
    - UnaryOp (Abs/Neg/Log/Sign/Sqrt)
    - RollingOp (Mean/Std/Sum/Max/Min/Skew/Kurt/Med/Rank/Delta)

对未覆盖的算子，PolarsAlphaCalculator.expression_to_formula() 会
fallback 到通用算子映射。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union


class Expression(ABC):
    """因子表达式抽象基类

    所有 Expression 必须实现：
    - to_string(): 转为 QuantNodes 公式字符串（给 OperatorVocab）
    - children: 子表达式列表（用于递归遍历）
    """

    @abstractmethod
    def to_string(self) -> str:
        """转为 QuantNodes 公式字符串

        Returns:
            公式字符串（如 "close - close.shift(5)"）
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def children(self) -> List["Expression"]:
        """子表达式列表"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Expr({self.to_string()})"


# ==============================================================================
# 基础表达式
# ==============================================================================


class Feature(Expression):
    """基础字段引用：$close, $open, $high, $low, $vwap, $volume

    Examples:
        Feature("close")  # → "$close" / "close"
    """
    def __init__(self, field: str):
        self.field = field

    def to_string(self) -> str:
        return self.field

    @property
    def children(self) -> List[Expression]:
        return []


class Literal(Expression):
    """字面量值

    Examples:
        Literal(1e-12)  # → "1e-12"
        Literal(0.5)     # → "0.5"
    """
    def __init__(self, value: Union[int, float]):
        self.value = value

    def to_string(self) -> str:
        # 整数直接显示，浮点保留精度
        if isinstance(self.value, int) or (
            isinstance(self.value, float) and self.value.is_integer()
        ):
            return str(int(self.value))
        return repr(self.value)

    @property
    def children(self) -> List[Expression]:
        return []


def _to_expr(x: Any) -> Expression:
    """自动把字面量转 Expression"""
    if isinstance(x, Expression):
        return x
    if isinstance(x, (int, float)):
        return Literal(x)
    raise TypeError(f"Cannot convert {type(x).__name__} to Expression")


class Ref(Expression):
    """时序滞后：Ref($close, 5) = 5 日前 close

    Examples:
        Ref(Feature("close"), 5)  # → "close.shift(5)"
    """
    def __init__(self, expr: Expression, lag: int):
        self.expr = expr
        self.lag = lag

    def to_string(self) -> str:
        return f"{self.expr.to_string()}.shift({self.lag})"

    @property
    def children(self) -> List[Expression]:
        return [self.expr]


# ==============================================================================
# 二元操作
# ==============================================================================


class BinaryOp(Expression):
    """二元操作：add, sub, mul, div, gt, lt, signedpower"""
    _OP_MAP = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "gt": ">",
        "lt": "<",
        "signedpower": "**",
    }

    def __init__(self, left, right, op: str):
        if op not in self._OP_MAP:
            raise ValueError(
                f"Unsupported op: {op}. Supported: {list(self._OP_MAP.keys())}"
            )
        self.left = _to_expr(left)
        self.right = _to_expr(right)
        self.op = op

    def to_string(self) -> str:
        symbol = self._OP_MAP[self.op]
        return f"({self.left.to_string()} {symbol} {self.right.to_string()})"

    @property
    def children(self) -> List[Expression]:
        return [self.left, self.right]


# 便捷构造（接受 Expression 或字面量）
def Add(left, right) -> BinaryOp:
    return BinaryOp(left, right, "add")


def Sub(left, right) -> BinaryOp:
    return BinaryOp(left, right, "sub")


def Mul(left, right) -> BinaryOp:
    return BinaryOp(left, right, "mul")


def Div(left, right) -> BinaryOp:
    return BinaryOp(left, right, "div")


def Greater(left, right) -> BinaryOp:
    return BinaryOp(left, right, "gt")


def Less(left, right) -> BinaryOp:
    return BinaryOp(left, right, "lt")


# ==============================================================================
# 一元操作
# ==============================================================================


class UnaryOp(Expression):
    """一元操作：abs, neg, log, sign, sqrt, rank, zscore, winsorize"""
    _OP_MAP = {
        "abs": "abs",
        "neg": "-",  # 一元负号特殊处理
        "log": "log",
        "sign": "sign",
        "sqrt": "sqrt",
        "rank": "rank",
        "zscore": "zscore",
        "winsorize": "winsorize",
    }

    def __init__(self, expr, op: str):
        if op not in self._OP_MAP:
            raise ValueError(
                f"Unsupported op: {op}. Supported: {list(self._OP_MAP.keys())}"
            )
        self.expr = _to_expr(expr)
        self.op = op

    def to_string(self) -> str:
        op_str = self._OP_MAP[self.op]
        if self.op == "neg":
            return f"(-{self.expr.to_string()})"
        return f"{op_str}({self.expr.to_string()})"

    @property
    def children(self) -> List[Expression]:
        return [self.expr]


# 便捷构造
def Abs(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "abs")


def Neg(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "neg")


def Log(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "log")


def Sign(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "sign")


def Sqrt(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "sqrt")


def Rank(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "rank")


def Zscore(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "zscore")


def Winsorize(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "winsorize")


def Rank(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "rank")


def Zscore(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "zscore")


def Winsorize(expr: Expression) -> UnaryOp:
    return UnaryOp(expr, "winsorize")


# ==============================================================================
# 滚动操作
# ==============================================================================


class RollingOp(Expression):
    """滚动操作：mean, std, var, max, min, sum, skew, kurt, median, rank, delta

    Examples:
        RollingOp(Feature("close"), window=20, op="mean")
        # → "ts_mean(close, 20)"
    """
    _OP_MAP = {
        "mean": "ts_mean",
        "std": "ts_std",
        "var": "ts_var",
        "max": "ts_max",
        "min": "ts_min",
        "sum": "ts_sum",
        "skew": "ts_skew",
        "kurt": "ts_kurt",
        "median": "ts_median",
        "rank": "ts_rank",
        "delta": "ts_delta",
    }

    def __init__(self, expr, window: int, op: str):
        if op not in self._OP_MAP:
            raise ValueError(
                f"Unsupported rolling op: {op}. "
                f"Supported: {list(self._OP_MAP.keys())}"
            )
        self.expr = _to_expr(expr)
        self.window = window
        self.op = op

    def to_string(self) -> str:
        func = self._OP_MAP[self.op]
        return f"{func}({self.expr.to_string()}, {self.window})"

    @property
    def children(self) -> List[Expression]:
        return [self.expr]


# 便捷构造
def Mean(expr: Expression, window: int) -> RollingOp:
    return RollingOp(expr, window, "mean")


def Std(expr: Expression, window: int) -> RollingOp:
    return RollingOp(expr, window, "std")


def Sum(expr: Expression, window: int) -> RollingOp:
    return RollingOp(expr, window, "sum")


def Max(expr: Expression, window: int) -> RollingOp:
    return RollingOp(expr, window, "max")


def Min(expr: Expression, window: int) -> RollingOp:
    return RollingOp(expr, window, "min")


def Delta(expr: Expression, window: int) -> RollingOp:
    return RollingOp(expr, window, "delta")


# ==============================================================================
# 工具函数
# ==============================================================================


def expression_to_formula(expr: Expression) -> str:
    """便利函数：Expression → 公式字符串

    Args:
        expr: Expression 对象

    Returns:
        QuantNodes 公式字符串
    """
    return expr.to_string()


def collect_feature_fields(expr: Expression) -> List[str]:
    """递归收集表达式中所有 Feature 字段"""
    fields: List[str] = []

    def _walk(e: Expression) -> None:
        if isinstance(e, Feature):
            if e.field not in fields:
                fields.append(e.field)
        for child in e.children:
            _walk(child)

    _walk(expr)
    return fields


def collect_rolling_windows(expr: Expression) -> List[int]:
    """递归收集表达式中所有 RollingOp 的窗口"""
    windows: List[int] = []

    def _walk(e: Expression) -> None:
        if isinstance(e, RollingOp):
            if e.window not in windows:
                windows.append(e.window)
        for child in e.children:
            _walk(child)

    _walk(expr)
    return windows

# coding=utf-8
"""
数学算子（代理层）

基于 factor_functions/math_ops.py 的实现，提供统一的类接口。

Available Operators:
    - add: 加法
    - sub: 减法
    - mul: 乘法
    - div: 除法
    - log: 对数
    - abs: 绝对值
    - pow: 幂运算
    - sqrt: 平方根
    - sign: 符号
    - clip: 裁剪
    - fill_null: 填充空值
    - nan_to_null: NaN 转 null

Usage:
    >>> math.add(pl.col("factor"), 1.0)
    >>> math.mul(pl.col("factor"), 0.5)
    >>> math.log(pl.col("factor"))
"""

from __future__ import annotations

from typing import Union, Optional, List, Any

import polars as pl
from polars import Expr

from QuantNodes.factor_node.factor_functions.math_ops import (
    add as _add,
    sub as _sub,
    mul as _mul,
    div as _div,
    log as _log,
    log1p as _log1p,
    abs as _abs,
    sqrt as _sqrt,
    sign as _sign,
    pow as _pow,
    clip as _clip,
    fill_null as _fill_null,
    fill_zero as _fill_zero,
    nan_to_null as _nan_to_null,
    ceil as _ceil,
    floor as _floor,
    sin as _sin,
    cos as _cos,
    tan as _tan,
    arcsin as _arcsin,
    arccos as _arccos,
    arctan as _arctan,
)


class MathOperators:
    """数学算子（代理层）"""

    @staticmethod
    def add(expr: Union[Expr, str], value: Union[float, Expr]) -> Expr:
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr + value

    @staticmethod
    def sub(expr: Union[Expr, str], value: Union[float, Expr]) -> Expr:
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr - value

    @staticmethod
    def mul(expr: Union[Expr, str], value: Union[float, Expr]) -> Expr:
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr * value

    @staticmethod
    def div(expr: Union[Expr, str], value: Union[float, Expr]) -> Expr:
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(value, (int, float)):
            return expr / (value + 1e-10)
        else:
            return expr / (value + 1e-10)

    @staticmethod
    def log(expr: Union[Expr, str], base: Optional[str] = None) -> Expr:
        return _log(expr, base=base)

    @staticmethod
    def log1p(expr: Union[Expr, str]) -> Expr:
        return _log1p(expr)

    @staticmethod
    def abs(expr: Union[Expr, str]) -> Expr:
        return _abs(expr)

    @staticmethod
    def sqrt(expr: Union[Expr, str]) -> Expr:
        return _sqrt(expr)

    @staticmethod
    def sign(expr: Union[Expr, str]) -> Expr:
        return _sign(expr)

    @staticmethod
    def pow(expr: Union[Expr, str], exponent: float = 2.0) -> Expr:
        return _pow(expr, exponent=exponent)

    @staticmethod
    def clip(expr: Union[Expr, str], lower: Optional[float] = None,
             upper: Optional[float] = None) -> Expr:
        return _clip(expr, lower=lower, upper=upper)

    @staticmethod
    def floor(expr: Union[Expr, str]) -> Expr:
        return _floor(expr)

    @staticmethod
    def ceil(expr: Union[Expr, str]) -> Expr:
        return _ceil(expr)

    @staticmethod
    def nan_to_null(expr: Union[Expr, str]) -> Expr:
        return _nan_to_null(expr)

    @staticmethod
    def fill_null(expr: Union[Expr, str], value: Union[float, str] = 0.0) -> Expr:
        return _fill_null(expr, value=value)

    @staticmethod
    def fill_zero(expr: Union[Expr, str]) -> Expr:
        return _fill_zero(expr)

    @staticmethod
    def round(expr: Union[Expr, str], decimals: int = 0) -> Expr:
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.round(decimals)

    @staticmethod
    def sin(expr: Union[Expr, str]) -> Expr:
        return _sin(expr)

    @staticmethod
    def cos(expr: Union[Expr, str]) -> Expr:
        return _cos(expr)

    @staticmethod
    def tan(expr: Union[Expr, str]) -> Expr:
        return _tan(expr)

    @staticmethod
    def arcsin(expr: Union[Expr, str]) -> Expr:
        return _arcsin(expr)

    @staticmethod
    def arccos(expr: Union[Expr, str]) -> Expr:
        return _arccos(expr)

    @staticmethod
    def arctan(expr: Union[Expr, str]) -> Expr:
        return _arctan(expr)

# coding=utf-8
"""
数学算子

基于 Polars 的基础数学运算。

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

Usage:
    >>> math.add(pl.col("factor"), 1.0)
    >>> math.mul(pl.col("factor"), 0.5)
    >>> math.log(pl.col("factor"))
"""

from __future__ import annotations

from typing import Union, Optional

import polars as pl
from polars import Expr


class MathOperators:
    """数学算子"""
    
    @staticmethod
    def add(
        expr: Union[Expr, str],
        value: Union[float, Expr]
    ) -> Expr:
        """
        加法
        
        Args:
            expr: 表达式或列名
            value: 加数 (常数或表达式)
        
        Example:
            >>> math.add("factor", 1.0)
            >>> math.add("factor", pl.col("weight"))
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(value, (int, float)):
            return expr + value
        else:
            return expr + value
    
    @staticmethod
    def sub(
        expr: Union[Expr, str],
        value: Union[float, Expr]
    ) -> Expr:
        """
        减法
        
        Args:
            expr: 表达式或列名
            value: 减数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(value, (int, float)):
            return expr - value
        else:
            return expr - value
    
    @staticmethod
    def mul(
        expr: Union[Expr, str],
        value: Union[float, Expr]
    ) -> Expr:
        """
        乘法
        
        Args:
            expr: 表达式或列名
            value: 乘数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(value, (int, float)):
            return expr * value
        else:
            return expr * value
    
    @staticmethod
    def div(
        expr: Union[Expr, str],
        value: Union[float, Expr],
        eps: float = 1e-8
    ) -> Expr:
        """
        除法
        
        Args:
            expr: 表达式或列名
            value: 除数
            eps: 防止除零的常数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(value, (int, float)):
            return expr / (value + eps)
        else:
            return expr / (value + eps)
    
    @staticmethod
    def log(
        expr: Union[Expr, str],
        base: str = "e"
    ) -> Expr:
        """
        对数
        
        Args:
            expr: 表达式或列名
            base: 底数 ("e"/"2"/"10")
        
        Example:
            >>> math.log("factor")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        if base == "e":
            return expr.log()
        elif base == "2":
            import math
            return expr.log() / math.log(2)
        elif base == "10":
            return expr.log10()
        return expr.log()
    
    @staticmethod
    def log1p(expr: Union[Expr, str]) -> Expr:
        """
        log(1 + x)
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.log1p()
    
    @staticmethod
    def abs(expr: Union[Expr, str]) -> Expr:
        """
        绝对值
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.abs()
    
    @staticmethod
    def pow(
        expr: Union[Expr, str],
        exponent: float
    ) -> Expr:
        """
        幂运算
        
        Args:
            expr: 表达式或列名
            exponent: 指数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.pow(exponent)
    
    @staticmethod
    def sqrt(
        expr: Union[Expr, str],
        eps: float = 1e-8
    ) -> Expr:
        """
        平方根
        
        Args:
            expr: 表达式或列名
            eps: 防止负数的常数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return (expr + eps).sqrt()
    
    @staticmethod
    def sign(expr: Union[Expr, str]) -> Expr:
        """
        符号
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.sign()
    
    @staticmethod
    def clip(
        expr: Union[Expr, str],
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> Expr:
        """
        裁剪
        
        Args:
            expr: 表达式或列名
            lower: 下界
            upper: 上界
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.clip(lower_bound=lower, upper_bound=upper)
    
    @staticmethod
    def floor(expr: Union[Expr, str]) -> Expr:
        """
        下取整
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.floor()
    
    @staticmethod
    def ceil(expr: Union[Expr, str]) -> Expr:
        """
        上取整
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.ceil()
    
    @staticmethod
    def round(
        expr: Union[Expr, str],
        decimals: int = 2
    ) -> Expr:
        """
        四舍五入
        
        Args:
            expr: 表达式或列名
            decimals: 小数位数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.round(decimals)
    
    @staticmethod
    def nan_to_null(expr: Union[Expr, str]) -> Expr:
        """
        NaN 转 null
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.fill_nan(None)
    
    @staticmethod
    def fill_null(
        expr: Union[Expr, str],
        value: Union[float, str] = 0.0
    ) -> Expr:
        """
        填充 null 值
        
        Args:
            expr: 表达式或列名
            value: 填充值 (float 或 "forward"/"backward")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        if isinstance(value, str):
            if value == "forward":
                return expr.fill_null(strategy="forward")
            elif value == "backward":
                return expr.fill_null(strategy="backward")
            else:
                return expr.fill_null(value=0.0)
        return expr.fill_null(value=value)
    
    @staticmethod
    def fill_zero(expr: Union[Expr, str]) -> Expr:
        """
        填充 0
        
        Args:
            expr: 表达式或列名
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.fill_null(value=0.0)
    
    # ========== 三角函数 ==========
    
    @staticmethod
    def sin(expr: Union[Expr, str]) -> Expr:
        """正弦"""
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.sin()
    
    @staticmethod
    def cos(expr: Union[Expr, str]) -> Expr:
        """余弦"""
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.cos()
    
    @staticmethod
    def tan(expr: Union[Expr, str]) -> Expr:
        """正切"""
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.tan()
    
    @staticmethod
    def arcsin(expr: Union[Expr, str]) -> Expr:
        """反正弦"""
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.arcsin()
    
    @staticmethod
    def arccos(expr: Union[Expr, str]) -> Expr:
        """反余弦"""
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.arccos()
    
    @staticmethod
    def arctan(expr: Union[Expr, str]) -> Expr:
        """反正切"""
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.arctan()
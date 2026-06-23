# coding=utf-8
"""
数学算子

本模块包含所有数学（point-wise）相关的因子运算算子。
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import polars as pl
from polars import Expr

from QuantNodes.factor_node.factor_functions._helpers import (
    OperatorCategory,
    register_operator,
    _ensure_expr,
    _make_nan_wrapper,
)


# ==============================================================================
# Point 算子 - 数学运算
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def abs(f: Union[Expr, str], **kwargs) -> Expr:
    """绝对值"""
    return _ensure_expr(f).abs()


@register_operator(OperatorCategory.POINT)
def log(f: Union[Expr, str], base: Optional[str] = None, **kwargs) -> Expr:
    """对数

    base: 对数底数，"e"/"2"/"10" 或 None（自然对数）
    """
    e = _ensure_expr(f)
    col_name = f if isinstance(f, str) else None

    if base is None or base == "e":
        result = e.log()
    elif base == "2":
        result = e.log() / pl.lit(2).log()
    elif base == "10":
        result = e.log10()
    else:
        result = e.log()

    if col_name:
        return result.alias(col_name)
    return result


@register_operator(OperatorCategory.POINT)
def sign(f: Union[Expr, str], **kwargs) -> Expr:
    """符号"""
    return _ensure_expr(f).sign()


@register_operator(OperatorCategory.POINT, "signedpower")
def signedpower(f: Union[Expr, str], exponent: float = 2.0, **kwargs) -> Expr:
    """保留符号的幂运算（Alpha 101 关键算子）

    signedpower(x, a) = sign(x) * abs(x) ** a

    Examples:
        - signedpower(close, 2)  → 保留正负号的 close²
        - signedpower(returns, 0.5)  → 保留正负号的 sqrt(|returns|)
    """
    e = _ensure_expr(f)
    return e.sign() * e.abs() ** exponent


@register_operator(OperatorCategory.POINT)
def sqrt(f: Union[Expr, str], **kwargs) -> Expr:
    """平方根"""
    return _ensure_expr(f).sqrt()


@register_operator(OperatorCategory.POINT)
def square(f: Union[Expr, str], **kwargs) -> Expr:
    """平方"""
    return _ensure_expr(f) ** 2


@register_operator(OperatorCategory.POINT)
def pow(f: Union[Expr, str], exponent: float = 2.0, **kwargs) -> Expr:
    """幂运算"""
    return _ensure_expr(f) ** exponent


@register_operator(OperatorCategory.POINT)
def clip(f: Union[Expr, str], lower: Optional[float] = None,
         upper: Optional[float] = None, **kwargs) -> Expr:
    """裁剪"""
    e = _ensure_expr(f)
    if lower is not None and upper is not None:
        return e.clip(lower, upper)
    elif lower is not None:
        return e.clip(lower_bound=lower)
    elif upper is not None:
        return e.clip(upper_bound=upper)
    return e


@register_operator(OperatorCategory.POINT)
def fill_null(f: Union[Expr, str], value: float = 0.0, **kwargs) -> Expr:
    """填充 null

    value: 填充值，或 "forward"/"backward" 策略
    """
    e = _ensure_expr(f)
    col_name = f if isinstance(f, str) else None

    if isinstance(value, str):
        if value == "forward":
            result = e.fill_null(strategy="forward")
        elif value == "backward":
            result = e.fill_null(strategy="backward")
        else:
            result = e.fill_null(0)
    else:
        result = e.fill_null(value)

    if col_name:
        return result.alias(col_name)
    return result


@register_operator(OperatorCategory.POINT)
def fill_null_by_strategy(f: Union[Expr, str], strategy: str = "mean", **kwargs) -> Expr:
    """按策略填充 null

    strategy: mean / median / max / min / zero / one
    """
    f = _ensure_expr(f)
    strategy_map = {
        "mean": f.fill_null(f.mean()),
        "median": f.fill_null(f.median()),
        "max": f.fill_null(f.max()),
        "min": f.fill_null(f.min()),
        "zero": f.fill_null(0),
        "one": f.fill_null(1),
    }
    return strategy_map.get(strategy, f.fill_null(0))


@register_operator(OperatorCategory.POINT)
def fill_zero(f: Union[Expr, str], **kwargs) -> Expr:
    """填充 0"""
    return _ensure_expr(f).fill_null(0)


@register_operator(OperatorCategory.POINT)
def nan_to_null(f: Union[Expr, str], **kwargs) -> Expr:
    """NaN 转 null"""
    col_name = f if isinstance(f, str) else None
    e = _ensure_expr(f)
    result = pl.when(e.is_nan()).then(pl.lit(None).cast(pl.Float64)).otherwise(e)
    if col_name:
        return result.alias(col_name)
    return result


@register_operator(OperatorCategory.POINT)
def isnull(f: Union[Expr, str], **kwargs) -> Expr:
    """判断空值"""
    return _ensure_expr(f).is_null()


@register_operator(OperatorCategory.POINT)
def notnull(f: Union[Expr, str], **kwargs) -> Expr:
    """判断非空"""
    return _ensure_expr(f).is_not_null()


# ==============================================================================
# Point 算子 (补充)
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def ceil(f: Union[Expr, str], **kwargs) -> Expr:
    """向上取整"""
    return _ensure_expr(f).ceil()


@register_operator(OperatorCategory.POINT)
def floor(f: Union[Expr, str], **kwargs) -> Expr:
    """向下取整"""
    return _ensure_expr(f).floor()


@register_operator(OperatorCategory.POINT)
def fix(f: Union[Expr, str], **kwargs) -> Expr:
    """向零取整"""
    e = _ensure_expr(f)
    return pl.when(e < 0).then(e.ceil()).otherwise(e.floor())


# ==============================================================================
# Point 算子 - 三角函数
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def sin(f: Union[Expr, str], **kwargs) -> Expr:
    """正弦"""
    return _ensure_expr(f).sin()


@register_operator(OperatorCategory.POINT)
def cos(f: Union[Expr, str], **kwargs) -> Expr:
    """余弦"""
    return _ensure_expr(f).cos()


@register_operator(OperatorCategory.POINT)
def tan(f: Union[Expr, str], **kwargs) -> Expr:
    """正切"""
    return _ensure_expr(f).tan()


@register_operator(OperatorCategory.POINT)
def arcsin(f: Union[Expr, str], **kwargs) -> Expr:
    """反正弦"""
    return _ensure_expr(f).arcsin()


@register_operator(OperatorCategory.POINT)
def arccos(f: Union[Expr, str], **kwargs) -> Expr:
    """反余弦"""
    return _ensure_expr(f).arccos()


@register_operator(OperatorCategory.POINT)
def arctan(f: Union[Expr, str], **kwargs) -> Expr:
    """反正切"""
    return _ensure_expr(f).arctan()


# ==============================================================================
# Point 算子 - 补充
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def log1p(f: Union[Expr, str], **kwargs) -> Expr:
    """log(1+x)"""
    return (_ensure_expr(f) + 1).log()


# ==============================================================================
# 双因子算子
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def add(f1: Union[Expr, str], f2: Union[Expr, str], **kwargs) -> Expr:
    """加法"""
    return _ensure_expr(f1) + _ensure_expr(f2)


@register_operator(OperatorCategory.POINT)
def sub(f1: Union[Expr, str], f2: Union[Expr, str], **kwargs) -> Expr:
    """减法"""
    return _ensure_expr(f1) - _ensure_expr(f2)


@register_operator(OperatorCategory.POINT)
def mul(f1: Union[Expr, str], f2: Union[Expr, str], **kwargs) -> Expr:
    """乘法"""
    return _ensure_expr(f1) * _ensure_expr(f2)


@register_operator(OperatorCategory.POINT)
def div(f1: Union[Expr, str], f2: Union[Expr, str], **kwargs) -> Expr:
    """除法"""
    return _ensure_expr(f1) / _ensure_expr(f2)


# ==============================================================================
# NaN 聚合
# ==============================================================================

_make_nan = lambda name, doc: _make_nan_wrapper(name, name, doc)

_make_nan("nanmax", "NaN 忽略的最大值")
_make_nan("nanmin", "NaN 忽略的最小值")
_make_nan("nanmean", "NaN 忽略的均值")
_make_nan("nansum", "NaN 忽略的求和")
_make_nan("nanstd", "NaN 忽略的标准差")
_make_nan("nanvar", "NaN 忽略的方差")
_make_nan("nanargmax", "NaN 忽略的最大值位置")
_make_nan("nanargmin", "NaN 忽略的最小值位置")
_make_nan("nanmedian", "NaN 忽略的中位数")
_make_nan("nancount", "NaN 忽略的计数")
_make_nan("nanprod", "NaN 忽略的乘积")


@register_operator(OperatorCategory.POINT)
def nanquantile(f: Union[Expr, str], quantile: float = 0.5,
                interpolation: str = "nearest", **kwargs) -> Expr:
    """NaN 忽略的分位数"""
    return _ensure_expr(f).quantile(quantile, interpolation=interpolation)


# ==============================================================================
# 其他
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def applymap(f: Union[Expr, str], func: Callable, **kwargs) -> Expr:
    """应用函数到每个元素"""
    return _ensure_expr(f).map_elements(func, return_dtype=pl.Float64)


@register_operator(OperatorCategory.POINT)
def astype(f: Union[Expr, str], dtype: Any = "float64", **kwargs) -> Expr:
    """类型转换"""
    if isinstance(dtype, str):
        dtype = getattr(pl, dtype, None) or getattr(pl.datatypes, dtype, None)
        if dtype is None:
            dtype = pl.Float64
    return _ensure_expr(f).cast(dtype)


@register_operator(OperatorCategory.POINT)
def replace(f: Union[Expr, str], old: Any = None, new: Any = None, **kwargs) -> Expr:
    """替换值"""
    if old is None or new is None:
        return _ensure_expr(f)
    return _ensure_expr(f).replace(old, new)


@register_operator(OperatorCategory.POINT)
def fetch(f: Union[Expr, str], index: int = 0, **kwargs) -> Expr:
    """取第 n 行"""
    e = _ensure_expr(f)
    if index == 0:
        return e.first()
    elif index == -1:
        return e.last()
    return e.nth(index)


@register_operator(OperatorCategory.POINT)
def where(condition: Union[Expr, str], true_val: Any = None,
          false_val: Any = None, **kwargs) -> Expr:
    """条件选择"""
    c = _ensure_expr(condition)
    t = _ensure_expr(true_val) if true_val is not None else pl.lit(None)
    f = _ensure_expr(false_val) if false_val is not None else pl.lit(None)
    return pl.when(c).then(t).otherwise(f)


@register_operator(OperatorCategory.POINT)
def fillna(f: Union[Expr, str], value: Any = None,
           method: str = "value", limit: int = 0, **kwargs) -> Expr:
    """填充空值"""
    e = _ensure_expr(f)
    if value is not None:
        return e.fill_null(value)
    elif method == "ffill":
        return e.forward_fill(limit=limit if limit else None)
    elif method == "bfill":
        return e.backward_fill(limit=limit if limit else None)
    return e


# ==============================================================================
# 组合算子
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def weighted_sum(factors: List[Union[Expr, str]], weights: Optional[List[float]] = None,
                 **kwargs) -> Expr:
    """加权求和"""
    exprs = [_ensure_expr(f) for f in factors]
    if weights is None:
        weights = [1.0] * len(exprs)
    weights_arr = pl.Series(weights)
    weights_arr = weights_arr / weights_arr.sum()
    return sum(e * w for e, w in zip(exprs, weights_arr))


@register_operator(OperatorCategory.POINT)
def combine(f1: Union[Expr, str], f2: Union[Expr, str],
            method: str = "add", **kwargs) -> Expr:
    """组合两个因子"""
    e1, e2 = _ensure_expr(f1), _ensure_expr(f2)
    if method == "add":
        return e1 + e2
    elif method == "sub":
        return e1 - e2
    elif method == "mul":
        return e1 * e2
    elif method == "div":
        return e1 / (e2 + 1e-10)
    elif method == "max":
        return pl.max_horizontal(e1, e2)
    elif method == "min":
        return pl.min_horizontal(e1, e2)
    return e1


@register_operator(OperatorCategory.POINT)
def if_then_else(condition: Union[Expr, str], then: Union[Expr, str],
                 else_: Union[Expr, str], **kwargs) -> Expr:
    """条件选择"""
    c = _ensure_expr(condition)
    t = _ensure_expr(then)
    e = _ensure_expr(else_)
    return pl.when(c).then(t).otherwise(e)


@register_operator(OperatorCategory.POINT)
def market_cap(price: Union[Expr, str], shares: Union[Expr, str],
               **kwargs) -> Expr:
    """市值"""
    return _ensure_expr(price) * _ensure_expr(shares)


@register_operator(OperatorCategory.POINT)
def book_to_market(book_value: Union[Expr, str], market_cap: Union[Expr, str],
                   **kwargs) -> Expr:
    """账面市值比"""
    return _ensure_expr(book_value) / (_ensure_expr(market_cap) + 1e-10)


@register_operator(OperatorCategory.POINT)
def earnings_to_market(earnings: Union[Expr, str], market_cap: Union[Expr, str],
                       **kwargs) -> Expr:
    """盈利市值比 (E/P)"""
    return _ensure_expr(earnings) / (_ensure_expr(market_cap) + 1e-10)

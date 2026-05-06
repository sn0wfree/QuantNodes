# coding=utf-8
"""
辅助函数模块

此模块包含所有算子实现所需的辅助函数，独立于 factor_functions 包，
以避免循环导入问题。
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import polars as pl
from polars import Expr


# ==============================================================================
# 注册表（全局单例）
# ==============================================================================

_OPERATOR_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "point": {},
    "time": {},
    "section": {},
    "multi_section": {},
    "talib": {},
}


class OperatorCategory:
    POINT = "point"
    TIME = "time"
    SECTION = "section"
    MULTI_SECTION = "multi_section"
    TALIB = "talib"


# ==============================================================================
# 注册函数
# ==============================================================================

def register_operator(category: str, name: Optional[str] = None):
    """装饰器：自动注册算子到注册表"""
    def decorator(func: Callable):
        op_name = name or func.__name__
        sig = inspect.signature(func)

        _OPERATOR_REGISTRY[category][op_name] = {
            "name": op_name,
            "category": category,
            "func": func,
            "doc": inspect.getdoc(func) or "",
            "signature": str(sig),
            "parameters": list(sig.parameters.keys()),
        }
        return func
    return decorator


# ==============================================================================
# 辅助函数
# ==============================================================================

def _ensure_expr(f: Any) -> Expr:
    """确保是表达式"""
    if isinstance(f, pl.Expr):
        return f
    if isinstance(f, str):
        return pl.col(f)
    return pl.lit(f)

def _combo_add(a: Expr, b: Expr) -> Expr:
    return a + b


def _combo_mul(a: Expr, b: Expr) -> Expr:
    return a * b


def _combo_max(a: Expr, b: Expr) -> Expr:
    return pl.max_horizontal(a, b)


def _combo_min(a: Expr, b: Expr) -> Expr:
    return pl.min_horizontal(a, b)


_COMBO_METHODS = {
    "add": _combo_add,
    "sum": _combo_add,
    "mul": _combo_mul,
    "max": _combo_max,
    "min": _combo_min,
}


def _inject(name: str, func: Callable):
    """将函数注入调用者的全局作用域"""
    import sys
    frame = sys._getframe(2)
    frame.f_globals[name] = func


def _expanding_var_expr(f: Union[Expr, str]) -> Expr:
    """扩展窗口方差核心公式"""
    e = _ensure_expr(f)
    n = pl.int_range(0, pl.len()) + 1
    mean = e.cum_sum() / n
    mean_sq = (e ** 2).cum_sum() / n
    return mean_sq - mean ** 2


def _apply_weights(f: Union[Expr, str], weights) -> Expr:
    """通用加权移动算子"""
    expr = _ensure_expr(f)
    result = expr * weights[0]
    for i in range(1, len(weights)):
        result = result + expr.shift(i) * weights[i]
    return result


def _cum_single_median(window):
    window.sort()
    n = len(window)
    mid = n // 2
    if n % 2 == 1:
        return window[mid]
    return (window[mid - 1] + window[mid]) / 2


def _cum_single_kurt(window):
    if len(window) < 4:
        return None
    arr = np.array(window, dtype=np.float64)
    m = arr.mean()
    s2 = arr.std(ddof=1)
    if s2 < 1e-15:
        return None
    return float(np.mean(((arr - m) / s2) ** 4) - 3)


def _cum_single_skew(window):
    if len(window) < 3:
        return None
    arr = np.array(window, dtype=np.float64)
    m = arr.mean()
    s2 = arr.std(ddof=1)
    if s2 < 1e-15:
        return None
    n = len(arr)
    return float(n / ((n - 1) * (n - 2)) * np.sum(((arr - m) / s2) ** 3))


def _cum_single_quantile(window, quantile=0.5):
    window.sort()
    n = len(window)
    idx = quantile * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return window[lo] * (1 - frac) + window[hi] * frac


_CUM_SINGLE_FUNCS = {
    "median": _cum_single_median,
    "kurt": _cum_single_kurt,
    "skew": _cum_single_skew,
}


def _cumulative_map_batches_single(e: Expr, func_name: str, return_dtype=pl.Float64) -> Expr:
    """单因子扩展窗口 map_batches 通用包装"""
    def _cum_stat(s: pl.Series) -> pl.Series:
        vals = s.to_list()
        result = []
        for i in range(len(vals)):
            window = [v for v in vals[:i + 1] if v is not None]
            if len(window) == 0:
                result.append(None)
            else:
                result.append(_CUM_SINGLE_FUNCS[func_name](window))
        return pl.Series(values=result)
    return e.map_batches(_cum_stat, return_dtype=return_dtype)


def _cum_dual_corr(arr1, arr2):
    c = np.corrcoef(arr1, arr2)[0, 1]
    return float(c) if np.isfinite(c) else None


def _cum_dual_cov(arr1, arr2):
    c = np.cov(arr1, arr2)[0, 1]
    return float(c) if np.isfinite(c) else None


_CUM_DUAL_FUNCS = {
    "corr": _cum_dual_corr,
    "cov": _cum_dual_cov,
}


def _cumulative_map_batches_dual(e1: Expr, e2: Expr, func_name: str,
                                 return_dtype=pl.Float64) -> Expr:
    """双因子扩展窗口 map_batches 通用包装"""
    def _cum_stat(args: list) -> pl.Series:
        s1, s2 = args[0], args[1]
        v1 = s1.to_list()
        v2 = s2.to_list()
        result = []
        for i in range(len(v1)):
            pairs = [(a, b) for a, b in zip(v1[:i + 1], v2[:i + 1])
                     if a is not None and b is not None]
            if len(pairs) < 2:
                result.append(None)
            else:
                arr1, arr2 = zip(*pairs)
                result.append(_CUM_DUAL_FUNCS[func_name](arr1, arr2))
        return pl.Series(values=result)
    return pl.map_batches([e1, e2], _cum_stat, return_dtype=return_dtype)


# ==============================================================================
# 工厂函数
# ==============================================================================

def _make_rolling_ts_wrapper(name: str, polars_method: str, doc: str):
    """创建使用 polars 滚动方法的包装器"""
    def _wrapper(f, window=20, min_periods=None, **kwargs):
        e = _ensure_expr(f)
        mp = min_periods or max(1, window // 2)
        return getattr(e, polars_method)(window, min_samples=mp)

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_expanding_wrapper(name: str, polars_method: str, doc: str):
    """创建使用 polars expanding 方法的包装器"""
    def _wrapper(f, min_periods=None, **kwargs):
        return _ensure_expr(f).__getattribute__(polars_method)()

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


_POLARS_NAN_METHOD_MAP = {
    "nanmax": "nan_max",
    "nanmin": "nan_min",
    "nanargmax": "arg_max",
    "nanargmin": "arg_min",
    "nanmedian": "median",
    "nancount": "count",
    "nanprod": "product",
    "nanmean": "mean",
    "nansum": "sum",
    "nanstd": "std",
    "nanvar": "var",
}


def _make_nan_wrapper(name: str, polars_method: str, doc: str, extra_params=None):
    """创建 NaN 跨截面聚合包装器"""
    actual_method = _POLARS_NAN_METHOD_MAP.get(polars_method, polars_method)

    def _wrapper(f, **kwargs):
        return getattr(_ensure_expr(f), actual_method)(**(extra_params or {}))

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.POINT, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_dual_rolling_wrapper(name: str, doc: str, is_corr: bool = True):
    """创建双因子滚动相关/协方差包装器（使用纯 Polars）"""
    def _wrapper(f1, f2, window=20, min_periods=None, **kwargs):
        e1 = _ensure_expr(f1)
        e2 = _ensure_expr(f2)
        mp = min_periods or max(1, window // 2)

        if is_corr:
            mean1 = e1.rolling_mean(window, min_samples=mp)
            mean2 = e2.rolling_mean(window, min_samples=mp)
            mean12 = (e1 * e2).rolling_mean(window, min_samples=mp)
            cov = mean12 - mean1 * mean2
            std1 = e1.rolling_std(window, min_samples=mp, ddof=0)
            std2 = e2.rolling_std(window, min_samples=mp, ddof=0)
            return cov / (std1 * std2 + 1e-8)
        else:
            m1 = e1.rolling_mean(window)
            m2 = e2.rolling_mean(window)
            return ((e1 - m1) * (e2 - m2)).rolling_mean(window)

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_alias(name: str, target_func: Callable, doc: str, category: str = None):
    """创建算子别名"""
    @functools.wraps(target_func)
    def _alias(*args, **kwargs):
        return target_func(*args, **kwargs)

    _alias.__name__ = name
    _alias.__doc__ = doc
    _alias.__qualname__ = name
    cat = category or OperatorCategory.TIME
    register_operator(cat, name)(_alias)
    _inject(name, _alias)
    return _alias


def _make_diff_wrapper(name: str, polars_method: str, doc: str, neg_periods: bool = False):
    """创建差分/变化包装器（使用纯 Polars）"""
    def _wrapper(f, periods=1, **kwargs):
        e = _ensure_expr(f)
        if neg_periods:
            return getattr(e, polars_method)(-periods)
        return getattr(e, polars_method)(periods)

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_aggr_wrapper(method: str, doc: str):
    """创建聚合算子包装器"""
    from QuantNodes.factor_node.factor_functions.composite_ops import aggregate

    def _wrapper(f, group_by, **kwargs):
        return aggregate(f, group_by, method)

    _wrapper.__name__ = f"aggr_{method}"
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = f"aggr_{method}"
    register_operator(OperatorCategory.MULTI_SECTION, f"aggr_{method}")(_wrapper)
    _inject(f"aggr_{method}", _wrapper)
    return _wrapper

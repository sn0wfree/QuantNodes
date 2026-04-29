# coding=utf-8
"""
因子函数 - Polars 版本

重构说明:
- 内部使用 QuantNodes.operators 中的 Polars 算子
- 装饰器注册表系统，支持动态发现、文档生成
- 纯 Polars 向量化，无 multiprocessing
- 工厂函数消除重复代码

Usage:
    from QuantNodes.factor_node import factor_functions as ff

    result = ff.rolling_mean("close", 20)
    result = ff.ts_corr("close", "volume", 20)
    result = ff.rank("factor")

    # 注册表 API
    ff.list_operators()
    ff.get_operator("rolling_mean")
    ff.operator_info("rolling_mean")
"""

from __future__ import annotations

import inspect
import functools
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import polars as pl
from polars import Expr

from QuantNodes.operators.time_series import TimeSeriesOperators
from QuantNodes.operators.section import SectionOperators
from QuantNodes.operators.math import MathOperators
from QuantNodes.operators.composite import CompositeOperators


# ==============================================================================
# 算子分类常量
# ==============================================================================

class OperatorCategory:
    """算子分类常量"""
    POINT = "point"
    TIME = "time"
    SECTION = "section"
    MULTI_SECTION = "multi_section"


# ==============================================================================
# 装饰器注册表
# ==============================================================================

_OPERATOR_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    OperatorCategory.POINT: {},
    OperatorCategory.TIME: {},
    OperatorCategory.SECTION: {},
    OperatorCategory.MULTI_SECTION: {},
}


def register_operator(category: str, name: Optional[str] = None):
    """装饰器：自动注册算子到注册表

    Args:
        category: 算子分类 (point/time/section/multi_section)
        name: 算子名称，默认使用函数名
    """
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
# 注册表查询 API
# ==============================================================================

def list_operators(category: Optional[str] = None) -> List[str]:
    """列出所有算子名称

    Args:
        category: 算子分类，可选值: point, time, section, multi_section
    """
    if category:
        return list(_OPERATOR_REGISTRY.get(category, {}).keys())
    return [name for cat in _OPERATOR_REGISTRY for name in _OPERATOR_REGISTRY[cat]]


def get_operator(name: str, category: Optional[str] = None) -> Optional[Callable]:
    """根据名称获取算子函数

    Args:
        name: 算子名称
        category: 算子分类，可选
    """
    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op["func"] if op else None

    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]["func"]
    return None


def operator_info(name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """获取算子详细信息"""
    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op if op else None

    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]
    return None


def generate_documentation(output_format: str = "markdown", category: Optional[str] = None) -> str:
    """生成算子文档"""
    if category:
        ops = {category: _OPERATOR_REGISTRY.get(category, {})}
    else:
        ops = _OPERATOR_REGISTRY

    if output_format == "json":
        import json
        serializable = {}
        for cat, cat_ops in ops.items():
            serializable[cat] = {}
            for name, info in cat_ops.items():
                serializable[cat][name] = {k: v for k, v in info.items() if k != "func"}
        return json.dumps(serializable, indent=2, ensure_ascii=False)

    lines = []
    for cat, cat_ops in ops.items():
        if not cat_ops:
            continue
        lines.append(f"\n## {cat.upper()}")
        lines.append(f"共 {len(cat_ops)} 个算子\n")
        for name, info in sorted(cat_ops.items()):
            lines.append(f"### {name}")
            if info.get("doc"):
                lines.append(f"{info['doc']}")
            lines.append(f"- 参数: {info.get('parameters', [])}")
            lines.append(f"- 签名: {info.get('signature', '')}")
            lines.append("")

    return "\n".join(lines)


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


# ==============================================================================
# 工厂函数 - 消除重复代码
# ==============================================================================

def _inject(name: str, func: Callable):
    """将函数注入模块全局作用域"""
    globals()[name] = func


def _make_rolling_ts_wrapper(name: str, ts_method: str, doc: str):
    """创建委托 TimeSeriesOperators 的滚动窗口包装器"""
    _ts_method = getattr(TimeSeriesOperators, ts_method)

    def _wrapper(f: Union[Expr, str], window: int = 20,
                 min_periods: Optional[int] = None, **kwargs) -> Expr:
        return _ts_method(f, window, min_periods)

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_expanding_wrapper(name: str, polars_method: str, doc: str):
    """创建使用 polars expanding 方法的包装器"""
    def _wrapper(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
        return _ensure_expr(f).__getattribute__(polars_method)()

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_nan_wrapper(name: str, polars_method: str, doc: str, extra_params: Optional[Dict[str, Any]] = None):
    """创建 NaN 跨截面聚合包装器"""
    def _wrapper(f: Union[Expr, str], **kwargs) -> Expr:
        return getattr(_ensure_expr(f), polars_method)(**(extra_params or {}))

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.POINT, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


def _make_aggr_wrapper(method: str, doc: str):
    """创建聚合算子包装器"""
    def _wrapper(f: Union[Expr, str], group_by: str, **kwargs) -> Expr:
        return aggregate(f, group_by, method)

    _wrapper.__name__ = f"aggr_{method}"
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = f"aggr_{method}"
    register_operator(OperatorCategory.MULTI_SECTION, f"aggr_{method}")(_wrapper)
    _inject(f"aggr_{method}", _wrapper)
    return _wrapper


def _expanding_var_expr(f: Union[Expr, str]) -> Expr:
    """扩展窗口方差核心公式（expanding_var 和 expanding_std 共用）"""
    e = _ensure_expr(f)
    n = pl.int_range(0, pl.len()) + 1
    mean = e.cum_sum() / n
    mean_sq = (e ** 2).cum_sum() / n
    return mean_sq - mean ** 2


def _apply_weights(f: Union[Expr, str], weights) -> Expr:
    """通用加权移动算子（decay_linear / decay_exp 共用）"""
    expr = _ensure_expr(f)
    result = expr * weights[0]
    for i in range(1, len(weights)):
        result = result + expr.shift(i) * weights[i]
    return result


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


# ==============================================================================
# 时间序列算子 - 滚动窗口 (工厂生成)
# ==============================================================================

_ROLLING_TS_DOCS = {
    "rolling_mean": "滚动窗口均值\n\n    Args:\n        f: 表达式或列名\n        window: 窗口大小\n        min_periods: 最小观测数",
    "rolling_max": "滚动窗口最大值",
    "rolling_min": "滚动窗口最小值",
    "rolling_sum": "滚动窗口求和",
    "rolling_median": "滚动窗口中位数",
}

_ROLLING_TS_METHODS = {
    "rolling_mean": "ts_mean",
    "rolling_max": "ts_max",
    "rolling_min": "ts_min",
    "rolling_sum": "ts_sum",
    "rolling_median": "ts_median",
}

for _name, _method in _ROLLING_TS_METHODS.items():
    _make_rolling_ts_wrapper(_name, _method, _ROLLING_TS_DOCS[_name])

del _name, _method, _ROLLING_TS_METHODS, _ROLLING_TS_DOCS


@register_operator(OperatorCategory.TIME)
def rolling_std(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    **kwargs
) -> Expr:
    """滚动窗口标准差"""
    return TimeSeriesOperators.ts_std(f, window, min_periods, ddof)


@register_operator(OperatorCategory.TIME)
def rolling_var(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口方差"""
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return e.rolling_var(window, min_samples=mp)


@register_operator(OperatorCategory.TIME)
def rolling_prod(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口求积"""
    return TimeSeriesOperators.ts_prod(f, window, min_periods)


@register_operator(OperatorCategory.TIME)
def rolling_skew(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口偏度"""
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    mean = e.rolling_mean(window, min_samples=mp)
    std = e.rolling_std(window, min_samples=mp)
    m3 = ((e - mean) ** 3).rolling_mean(window, min_samples=mp)
    return m3 / (std ** 3 + 1e-10)


@register_operator(OperatorCategory.TIME)
def rolling_kurt(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口峰度"""
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    mean = e.rolling_mean(window, min_samples=mp)
    std = e.rolling_std(window, min_samples=mp)
    m4 = ((e - mean) ** 4).rolling_mean(window, min_samples=mp)
    return m4 / (std ** 4 + 1e-10) - 3


@register_operator(OperatorCategory.TIME)
def rolling_count(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口计数"""
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return e.is_not_null().cast(pl.Int64).rolling_sum(window, min_samples=mp)


@register_operator(OperatorCategory.TIME)
def rolling_quantile(
    f: Union[Expr, str],
    window: int = 20,
    quantile: float = 0.5,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口分位数"""
    e = _ensure_expr(f)
    mp = min_periods or window
    return e.rolling_quantile(quantile, window_size=window, interpolation="nearest", min_samples=mp)


@register_operator(OperatorCategory.TIME)
def rolling_rank(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口排名（归一化到 0-1）"""
    e = _ensure_expr(f)
    mp = min_periods or window
    return e.rolling_rank(window, min_samples=mp)


# ==============================================================================
# 时间序列算子 - 双因子滚动 (工厂生成)
# ==============================================================================

def _rolling_arg_op(f: Union[Expr, str], window: int, op: str,
                    min_periods: Optional[int] = None) -> Expr:
    """用 shift 比较链实现 rolling argmax/argmin"""
    e = _ensure_expr(f)
    max_window = min(window, 30)
    is_max = op == "max"

    best_val = e.shift(0)
    best_idx = pl.lit(0, dtype=pl.Int32)

    for i in range(1, max_window):
        shifted = e.shift(i)
        is_better = (shifted >= best_val) if is_max else (shifted <= best_val)
        best_val = pl.when(is_better).then(shifted).otherwise(best_val)
        best_idx = pl.when(is_better).then(pl.lit(i, dtype=pl.Int32)).otherwise(best_idx)

    return best_idx


@register_operator(OperatorCategory.TIME)
def rolling_argmax(f: Union[Expr, str], window: int = 20,
                   min_periods: Optional[int] = None, **kwargs) -> Expr:
    """滚动窗口最大值索引（0=当前行，1=前一行...）"""
    return _rolling_arg_op(f, window, "max", min_periods)


@register_operator(OperatorCategory.TIME)
def rolling_argmin(f: Union[Expr, str], window: int = 20,
                   min_periods: Optional[int] = None, **kwargs) -> Expr:
    """滚动窗口最小值索引（0=当前行，1=前一行...）"""
    return _rolling_arg_op(f, window, "min", min_periods)


# ==============================================================================
# 时间序列算子 - 相关系数/协方差 (工厂生成)
# ==============================================================================

def _make_dual_rolling_wrapper(name: str, ts_method: str, doc: str):
    """创建双因子滚动相关/协方差包装器"""
    _ts_func = getattr(TimeSeriesOperators, ts_method)

    def _wrapper(f1: Union[Expr, str], f2: Union[Expr, str],
                 window: int = 20, min_periods: Optional[int] = None, **kwargs) -> Expr:
        min_periods = min_periods or max(1, window // 2)
        return _ts_func(f1, f2, window, min_periods)

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


_make_dual_rolling_wrapper(
    "rolling_corr", "ts_corr",
    "滚动窗口相关系数（双因子）\n\n    Args:\n        f1: 第一个表达式\n        f2: 第二个表达式\n        window: 窗口大小\n        min_periods: 最小观测数"
)
_make_dual_rolling_wrapper(
    "rolling_cov", "ts_cov",
    "滚动窗口协方差（双因子）\n\n    Args:\n        f1: 第一个表达式\n        f2: 第二个表达式\n        window: 窗口大小\n        min_periods: 最小观测数"
)


# ==============================================================================
# 时间序列算子 - 别名
# ==============================================================================

def _make_alias(name: str, target_func: Callable, doc: str,
                 category: str = OperatorCategory.TIME):
    """创建算子别名"""
    @functools.wraps(target_func)
    def _alias(*args, **kwargs):
        return target_func(*args, **kwargs)

    _alias.__name__ = name
    _alias.__doc__ = doc
    _alias.__qualname__ = name
    register_operator(category, name)(_alias)
    _inject(name, _alias)
    return _alias


_make_alias("ts_corr", get_operator("rolling_corr"),
            "滚动相关系数 (rolling_corr 别名)")
_make_alias("ts_cov", get_operator("rolling_cov"),
            "滚动协方差 (rolling_cov 别名)")

_make_alias("correlation", get_operator("ts_corr"),
            "相关系数 (ts_corr 别名)")
_make_alias("covariance", get_operator("ts_cov"),
            "协方差 (ts_cov 别名)")

# ts_* 别名
_make_alias("ts_mean", get_operator("rolling_mean"),
            "时间序列均值 (滚动均值别名)")
_make_alias("ts_std", get_operator("rolling_std"),
            "时间序列标准差")
_make_alias("ts_max", get_operator("rolling_max"),
            "时间序列最大值")
_make_alias("ts_min", get_operator("rolling_min"),
            "时间序列最小值")
_make_alias("ts_sum", get_operator("rolling_sum"),
            "时间序列求和")
_make_alias("ts_median", get_operator("rolling_median"),
            "时间序列中位数")


@register_operator(OperatorCategory.TIME)
def ts_rank(f: Union[Expr, str], window: int = 20,
            min_periods: Optional[int] = None, **kwargs) -> Expr:
    """滚动排名 (0-1 归一化)"""
    return TimeSeriesOperators.ts_rank(f, window, min_periods)


_make_alias("ts_argmax", get_operator("rolling_argmax"),
            "滚动最大值的位置（相对于当前行的偏移量）")
_make_alias("ts_argmin", get_operator("rolling_argmin"),
            "滚动最小值的位置（相对于当前行的偏移量）")


# ==============================================================================
# 时间序列算子 - 差分与变化
# ==============================================================================

def _make_diff_wrapper(name: str, ts_method: str, doc: str):
    """创建差分/变化包装器"""
    _ts_func = getattr(TimeSeriesOperators, ts_method)

    def _wrapper(f: Union[Expr, str], periods: int = 1, **kwargs) -> Expr:
        return _ts_func(f, periods)

    _wrapper.__name__ = name
    _wrapper.__doc__ = doc
    _wrapper.__qualname__ = name
    register_operator(OperatorCategory.TIME, name)(_wrapper)
    _inject(name, _wrapper)
    return _wrapper


_make_diff_wrapper("ts_delta", "ts_delta", "差分")
_make_diff_wrapper("ts_pct_change", "ts_pct_change", "百分比变化")
_make_diff_wrapper("diff", "ts_delta", "差分")
_make_diff_wrapper("lag", "ts_lag", "滞后算子")

_make_alias("delta", get_operator("ts_delta"), "差分 (ts_delta 别名)")
_make_alias("pct_change", get_operator("ts_pct_change"), "百分比变化 (ts_pct_change 别名)")


@register_operator(OperatorCategory.TIME)
def ts_lag(f: Union[Expr, str], periods: int = 1, **kwargs) -> Expr:
    """滞后 (向后移动)"""
    return TimeSeriesOperators.ts_lag(f, periods)


@register_operator(OperatorCategory.TIME)
def ts_lead(f: Union[Expr, str], periods: int = 1, **kwargs) -> Expr:
    """前向移动 (向前移动)"""
    return TimeSeriesOperators.ts_lead(f, periods)


_make_alias("delay", get_operator("ts_lag"), "滞后 (ts_lag 别名)")
_make_alias("ref", get_operator("delay"), "引用历史值 (delay 别名)")
_make_alias("shift", get_operator("ts_lag"), "移动 (shift 别名)")


# ==============================================================================
# 时间序列算子 - 扩展窗口
# ==============================================================================

_make_expanding_wrapper("expanding_sum", "cum_sum",
                        "扩展窗口求和")
_make_expanding_wrapper("expanding_max", "cum_max",
                        "扩展窗口最大值")
_make_expanding_wrapper("expanding_min", "cum_min",
                        "扩展窗口最小值")


@register_operator(OperatorCategory.TIME)
def expanding_mean(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口均值 (累计至当前)"""
    e = _ensure_expr(f)
    return e.cum_sum() / (pl.int_range(0, pl.len()) + 1)


@register_operator(OperatorCategory.TIME)
def expanding_var(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口方差"""
    return _expanding_var_expr(f)


@register_operator(OperatorCategory.TIME)
def expanding_std(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口标准差"""
    return (_expanding_var_expr(f) + 1e-10).sqrt()


@register_operator(OperatorCategory.TIME)
def expanding_count(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口计数（非空值累计数量）"""
    return _ensure_expr(f).is_not_null().cast(pl.Int64).cum_sum()


@register_operator(OperatorCategory.TIME)
def expanding_median(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口中位数"""
    return _cumulative_map_batches_single(_ensure_expr(f), "median")


@register_operator(OperatorCategory.TIME)
def expanding_kurt(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口峰度"""
    return _cumulative_map_batches_single(_ensure_expr(f), "kurt")


@register_operator(OperatorCategory.TIME)
def expanding_skew(f: Union[Expr, str], min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口偏度"""
    return _cumulative_map_batches_single(_ensure_expr(f), "skew")


@register_operator(OperatorCategory.TIME)
def expanding_quantile(f: Union[Expr, str], quantile: float = 0.5,
                       min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口分位数"""

    def _cum_quantile(s: pl.Series) -> pl.Series:
        vals = s.to_list()
        result = []
        for i in range(len(vals)):
            window = [v for v in vals[:i + 1] if v is not None]
            if len(window) == 0:
                result.append(None)
            else:
                result.append(_cum_single_quantile(window, quantile))
        return pl.Series(values=result)

    return _ensure_expr(f).map_batches(_cum_quantile, return_dtype=pl.Float64)


@register_operator(OperatorCategory.TIME)
def expanding_corr(f1: Union[Expr, str], f2: Union[Expr, str],
                   min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口相关系数（双因子）"""
    return _cumulative_map_batches_dual(_ensure_expr(f1), _ensure_expr(f2), "corr")


@register_operator(OperatorCategory.TIME)
def expanding_cov(f1: Union[Expr, str], f2: Union[Expr, str],
                  min_periods: Optional[int] = None, **kwargs) -> Expr:
    """扩展窗口协方差（双因子）"""
    return _cumulative_map_batches_dual(_ensure_expr(f1), _ensure_expr(f2), "cov")


# ==============================================================================
# 时间序列算子 - 指数加权
# ==============================================================================

@register_operator(OperatorCategory.TIME)
def ewm_mean(f: Union[Expr, str], alpha: float = 0.5,
             adjust: bool = True, **kwargs) -> Expr:
    """指数加权移动平均"""
    return TimeSeriesOperators.ewm_mean(f, alpha, adjust=adjust)


@register_operator(OperatorCategory.TIME)
def ewm_std(f: Union[Expr, str], alpha: float = 0.5,
            adjust: bool = True, **kwargs) -> Expr:
    """指数加权移动标准差"""
    return TimeSeriesOperators.ewm_std(f, alpha, adjust=adjust)


@register_operator(OperatorCategory.TIME)
def ewm_corr(f1: Union[Expr, str], f2: Union[Expr, str],
             alpha: float = 0.5, **kwargs) -> Expr:
    """指数加权相关系数"""
    return TimeSeriesOperators.ewm_corr(f1, f2, alpha=alpha)


@register_operator(OperatorCategory.TIME)
def ewm_var(f: Union[Expr, str], alpha: float = 0.5,
            adjust: bool = True, **kwargs) -> Expr:
    """指数加权移动方差"""
    e = _ensure_expr(f)
    mean = e.ewm_mean(alpha=alpha, adjust=adjust)
    mean_sq = (e ** 2).ewm_mean(alpha=alpha, adjust=adjust)
    return mean_sq - mean ** 2


@register_operator(OperatorCategory.TIME)
def ewm_cov(f1: Union[Expr, str], f2: Union[Expr, str],
            alpha: float = 0.5, adjust: bool = True, **kwargs) -> Expr:
    """指数加权移动协方差（双因子）"""
    e1, e2 = _ensure_expr(f1), _ensure_expr(f2)
    mean1 = e1.ewm_mean(alpha=alpha, adjust=adjust)
    mean2 = e2.ewm_mean(alpha=alpha, adjust=adjust)
    mean12 = (e1 * e2).ewm_mean(alpha=alpha, adjust=adjust)
    return mean12 - mean1 * mean2


# ==============================================================================
# 截面算子
# ==============================================================================

@register_operator(OperatorCategory.SECTION)
def standardizeZScore(f: Union[Expr, str], eps: float = 1e-8, **kwargs) -> Expr:
    """Z-score 标准化"""
    return SectionOperators.zscore(f, eps)


_make_alias("zscore", standardizeZScore, "Z-score 标准化 (standardizeZScore 别名)",
            category=OperatorCategory.SECTION)


@register_operator(OperatorCategory.SECTION)
def rank(f: Union[Expr, str], method: str = "dense", **kwargs) -> Expr:
    """截面排名 (归一化到 0-1)"""
    return SectionOperators.rank(f, method)


@register_operator(OperatorCategory.SECTION)
def winsorize(f: Union[Expr, str], lower: float = 0.01,
              upper: float = 0.01, method: str = "quantile", **kwargs) -> Expr:
    """去极值"""
    return SectionOperators.winsorize(f, lower, upper, method)


@register_operator(OperatorCategory.SECTION)
def neutralize(f: Union[Expr, str],
               group: Optional[Union[Expr, str]] = None, **kwargs) -> Expr:
    """行业中性的 (减去行业均值)"""
    if group:
        return SectionOperators.neutralize(f, group)
    return SectionOperators.neutralize_market(f)


@register_operator(OperatorCategory.SECTION)
def neutralize_market(f: Union[Expr, str], **kwargs) -> Expr:
    """市场中性 (减去市场均值)"""
    return SectionOperators.neutralize_market(f)


@register_operator(OperatorCategory.SECTION)
def scale(f: Union[Expr, str], method: str = "zscore", **kwargs) -> Expr:
    """归一化"""
    return SectionOperators.scale(f, method)



@register_operator(OperatorCategory.SECTION)
def ic(f: Union[Expr, str], target: Union[Expr, str], **kwargs) -> Expr:
    """IC (Pearson 相关系数)"""
    return SectionOperators.ic(f, target)


@register_operator(OperatorCategory.SECTION)
def rank_ic(f: Union[Expr, str], target: Union[Expr, str], **kwargs) -> Expr:
    """Rank IC (Spearman 相关系数)"""
    return SectionOperators.rank_ic(f, target)


@register_operator(OperatorCategory.SECTION)
def group_norm(f: Union[Expr, str], group: Union[Expr, str],
               method: str = "zscore", **kwargs) -> Expr:
    """分组标准化"""
    return SectionOperators.group_norm(f, group, method)


@register_operator(OperatorCategory.SECTION)
def group_winsorize(f: Union[Expr, str], group: Union[Expr, str],
                    lower: float = 0.01, upper: float = 0.01, **kwargs) -> Expr:
    """分组去极值"""
    return SectionOperators.group_winsorize(f, group, lower, upper)


@register_operator(OperatorCategory.SECTION)
def orthogonalize(f: Union[Expr, str], reference: Union[Expr, str], **kwargs) -> Expr:
    """正交化：从因子 f 中剔除 reference 的影响"""
    f = _ensure_expr(f)
    reference = _ensure_expr(reference)
    cov = (f * reference).mean() - f.mean() * reference.mean()
    var_ref = (reference ** 2).mean() - reference.mean() ** 2
    beta = cov / (var_ref + 1e-10)
    return f - beta * reference


@register_operator(OperatorCategory.SECTION)
def fillNaNByFun(f: Union[Expr, str], value: Any = 0, **kwargs) -> Expr:
    """按值填充 NaN"""
    f = _ensure_expr(f)
    if callable(value):
        return f.map_elements(value)
    return f.fill_null(value)


@register_operator(OperatorCategory.SECTION)
def fillNaNByRegress(f: Union[Expr, str], reference: Union[Expr, str], **kwargs) -> Expr:
    """按回归值填充 NaN"""
    f = _ensure_expr(f)
    reference = _ensure_expr(reference)
    cov = (f * reference).mean() - f.mean() * reference.mean()
    var_ref = (reference ** 2).mean() - reference.mean() ** 2
    beta = cov / (var_ref + 1e-10)
    alpha = f.mean() - beta * reference.mean()
    predicted = alpha + beta * reference
    return f.fill_null(predicted)


# ==============================================================================
# NaN 跨截面聚合算子 (工厂生成)
# ==============================================================================

_NAN_DOCS = {
    "nanmax": "跨截面忽略空值求最大值",
    "nanmin": "跨截面忽略空值求最小值",
    "nanmean": "跨截面忽略空值求均值",
    "nansum": "跨截面忽略空值求和",
    "nanstd": "跨截面忽略空值求标准差",
    "nanvar": "跨截面忽略空值求方差",
}

for _name, _method in [("nanmax", "max"), ("nanmin", "min"), ("nanmean", "mean"),
                        ("nansum", "sum"), ("nanstd", "std"), ("nanvar", "var")]:
    _make_nan_wrapper(_name, _method, _NAN_DOCS[_name])

del _name, _method, _NAN_DOCS


# ==============================================================================
# Point 算子 - 数学运算
# ==============================================================================

@register_operator(OperatorCategory.POINT)
def abs(f: Union[Expr, str], **kwargs) -> Expr:
    """绝对值"""
    return MathOperators.abs(f)


@register_operator(OperatorCategory.POINT)
def log(f: Union[Expr, str], **kwargs) -> Expr:
    """对数"""
    return MathOperators.log(f)


@register_operator(OperatorCategory.POINT)
def sign(f: Union[Expr, str], **kwargs) -> Expr:
    """符号"""
    return MathOperators.sign(f)


@register_operator(OperatorCategory.POINT)
def sqrt(f: Union[Expr, str], **kwargs) -> Expr:
    """平方根"""
    return MathOperators.sqrt(f)


@register_operator(OperatorCategory.POINT)
def square(f: Union[Expr, str], **kwargs) -> Expr:
    """平方"""
    return _ensure_expr(f) ** 2


@register_operator(OperatorCategory.POINT)
def pow(f: Union[Expr, str], exponent: float = 2.0, **kwargs) -> Expr:
    """幂运算"""
    return MathOperators.pow(f, exponent)


@register_operator(OperatorCategory.POINT)
def clip(f: Union[Expr, str], lower: Optional[float] = None,
         upper: Optional[float] = None, **kwargs) -> Expr:
    """裁剪"""
    return MathOperators.clip(f, lower, upper)


@register_operator(OperatorCategory.POINT)
def fill_null(f: Union[Expr, str], value: float = 0.0, **kwargs) -> Expr:
    """填充 null"""
    return MathOperators.fill_null(f, value)


@register_operator(OperatorCategory.POINT)
def fill_zero(f: Union[Expr, str], **kwargs) -> Expr:
    """填充 0"""
    return MathOperators.fill_zero(f)


@register_operator(OperatorCategory.POINT)
def nan_to_null(f: Union[Expr, str], **kwargs) -> Expr:
    """NaN 转 null"""
    return MathOperators.nan_to_null(f)


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


@register_operator(OperatorCategory.POINT)
def applymap(f: Union[Expr, str], func: Callable, **kwargs) -> Expr:
    """应用自定义函数"""
    return _ensure_expr(f).map_elements(func)


@register_operator(OperatorCategory.POINT)
def nanargmax(f: Union[Expr, str], **kwargs) -> Expr:
    """忽略空值求最大值索引"""
    return _ensure_expr(f).arg_max()


@register_operator(OperatorCategory.POINT)
def nanargmin(f: Union[Expr, str], **kwargs) -> Expr:
    """忽略空值求最小值索引"""
    return _ensure_expr(f).arg_min()


@register_operator(OperatorCategory.POINT)
def nanmedian(f: Union[Expr, str], **kwargs) -> Expr:
    """忽略空值求中位数"""
    return _ensure_expr(f).median()


@register_operator(OperatorCategory.POINT)
def nanquantile(f: Union[Expr, str], quantile: float = 0.5,
                interpolation: str = "nearest", **kwargs) -> Expr:
    """忽略空值求分位数"""
    return _ensure_expr(f).quantile(quantile, interpolation=interpolation)


@register_operator(OperatorCategory.POINT)
def nancount(f: Union[Expr, str], **kwargs) -> Expr:
    """统计非空值数量"""
    return _ensure_expr(f).count()


@register_operator(OperatorCategory.POINT)
def nanprod(f: Union[Expr, str], **kwargs) -> Expr:
    """忽略空值求积"""
    return _ensure_expr(f).product()


@register_operator(OperatorCategory.POINT)
def astype(f: Union[Expr, str], dtype: str = "float64", **kwargs) -> Expr:
    """类型转换"""
    type_map = {
        "float64": pl.Float64, "float32": pl.Float32,
        "int64": pl.Int64, "int32": pl.Int32, "int8": pl.Int8,
        "utf8": pl.Utf8, "bool": pl.Boolean,
    }
    return _ensure_expr(f).cast(type_map.get(dtype, pl.Float64))


@register_operator(OperatorCategory.POINT)
def replace(f: Union[Expr, str], old: Any, new: Any, **kwargs) -> Expr:
    """值替换"""
    return _ensure_expr(f).replace(old, new)


@register_operator(OperatorCategory.POINT)
def fetch(f: Union[Expr, str], index: int = 0, **kwargs) -> Expr:
    """获取指定位置数据"""
    return _ensure_expr(f).list.get(index)


@register_operator(OperatorCategory.POINT)
def where(condition: Union[Expr, str], true_val: Union[Expr, str, Any],
          false_val: Union[Expr, str, Any] = None, **kwargs) -> Expr:
    """条件选择"""
    cond = _ensure_expr(condition)
    t = pl.lit(true_val) if isinstance(true_val, (int, float, str)) else _ensure_expr(true_val)
    if false_val is None:
        fv = pl.lit(None)
    elif isinstance(false_val, (int, float, str)):
        fv = pl.lit(false_val)
    else:
        fv = _ensure_expr(false_val)
    return pl.when(cond).then(t).otherwise(fv)


@register_operator(OperatorCategory.POINT)
def fillna(f: Union[Expr, str], value: Any = None, method: str = "ffill",
           limit: int = 0, **kwargs) -> Expr:
    """填充空值"""
    e = _ensure_expr(f)
    if value is not None:
        return e.fill_null(value)
    if method == "ffill":
        return e.forward_fill(limit) if limit > 0 else e.forward_fill()
    elif method == "bfill":
        return e.backward_fill(limit) if limit > 0 else e.backward_fill()
    return e


# ==============================================================================
# 组合算子
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


@register_operator(OperatorCategory.POINT)
def weighted_sum(factors: List[Union[Expr, str]],
                 weights: Optional[List[float]] = None, **kwargs) -> Expr:
    """加权求和"""
    return CompositeOperators.weighted_sum(factors, weights)


@register_operator(OperatorCategory.POINT)
def combine(f1: Union[Expr, str], f2: Union[Expr, str],
            method: str = "add", **kwargs) -> Expr:
    """组合因子"""
    a, b = _ensure_expr(f1), _ensure_expr(f2)
    return _COMBO_METHODS.get(method, _COMBO_METHODS["add"])(a, b)


@register_operator(OperatorCategory.POINT)
def if_then_else(condition: Union[Expr, str], then: Union[Expr, str],
                 else_: Union[Expr, str], **kwargs) -> Expr:
    """条件表达式"""
    return pl.when(_ensure_expr(condition)).then(_ensure_expr(then)).otherwise(_ensure_expr(else_))


# ==============================================================================
# 高级算子
# ==============================================================================

@register_operator(OperatorCategory.TIME)
def regress(y: Union[Expr, str], x: Union[Expr, str],
            window: int = 20, **kwargs) -> Expr:
    """滑动窗口线性回归的残差"""
    ey, ex = _ensure_expr(y), _ensure_expr(x)
    y_mean = ey.rolling_mean(window)
    x_mean = ex.rolling_mean(window)
    x_var = ex.rolling_var(window)
    xy_cov = (ey * ex).rolling_mean(window) - y_mean * x_mean
    beta = xy_cov / (x_var + 1e-8)
    return ey - y_mean - beta * (ex - x_mean)


@register_operator(OperatorCategory.TIME)
def zscored(f: Union[Expr, str], window: int = 20, **kwargs) -> Expr:
    """滚动 Z-score"""
    e = _ensure_expr(f)
    return (e - e.rolling_mean(window)) / (e.rolling_std(window) + 1e-8)


@register_operator(OperatorCategory.TIME)
def decay_linear(f: Union[Expr, str], window: int = 20, **kwargs) -> Expr:
    """线性衰减加权"""
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    return _apply_weights(f, weights)


@register_operator(OperatorCategory.TIME)
def decay_exp(f: Union[Expr, str], halflife: int = 10, **kwargs) -> Expr:
    """指数衰减加权"""
    alpha = 0.5 ** (1 / halflife)
    weights = alpha ** np.arange(halflife)
    weights = weights / weights.sum()
    return _apply_weights(f, weights)


@register_operator(OperatorCategory.TIME)
def vwap(price: Union[Expr, str], volume: Union[Expr, str],
         window: int = 20, **kwargs) -> Expr:
    """成交量加权平均价"""
    ep, ev = _ensure_expr(price), _ensure_expr(volume)
    return (ep * ev).rolling_sum(window) / (ev.rolling_sum(window) + 1e-8)


@register_operator(OperatorCategory.POINT)
def market_cap(price: Union[Expr, str], shares: Union[Expr, str], **kwargs) -> Expr:
    """市值"""
    return _ensure_expr(price) * _ensure_expr(shares)


@register_operator(OperatorCategory.POINT)
def book_to_market(book_value: Union[Expr, str], market_cap: Union[Expr, str], **kwargs) -> Expr:
    """市净率"""
    return _ensure_expr(book_value) / _ensure_expr(market_cap)


@register_operator(OperatorCategory.POINT)
def earnings_to_market(earnings: Union[Expr, str], market_cap: Union[Expr, str], **kwargs) -> Expr:
    """盈利市率"""
    return _ensure_expr(earnings) / _ensure_expr(market_cap)


# ==============================================================================
# Multi-Section 算子
# ==============================================================================

@register_operator(OperatorCategory.MULTI_SECTION)
def aggregate(f: Union[Expr, str], group_by: str, method: str = "mean", **kwargs) -> Expr:
    """按组聚合"""
    f = _ensure_expr(f)
    method_map = {
        "mean": f.mean().over(group_by),
        "sum": f.sum().over(group_by),
        "std": f.std().over(group_by),
        "var": f.var().over(group_by),
        "median": f.median().over(group_by),
        "min": f.min().over(group_by),
        "max": f.max().over(group_by),
        "first": f.first().over(group_by),
        "last": f.last().over(group_by),
        "count": f.count().over(group_by),
    }
    return method_map.get(method, f.mean().over(group_by))


@register_operator(OperatorCategory.MULTI_SECTION)
def disaggregate(f: Union[Expr, str], group_by: str, **kwargs) -> Expr:
    """解聚合 (将聚合值展开到组内每个成员)"""
    return _ensure_expr(f).over(group_by)


# Aggr 系列工厂
_MAKE_AGGR_DOCS = {
    "sum": "聚合求和",
    "mean": "聚合均值",
    "max": "聚合最大值",
    "min": "聚合最小值",
    "std": "聚合标准差",
    "var": "聚合方差",
    "median": "聚合中位数",
    "count": "聚合计数",
}

for _method, _doc in _MAKE_AGGR_DOCS.items():
    _make_aggr_wrapper(_method, _doc)

del _method, _doc, _MAKE_AGGR_DOCS


@register_operator(OperatorCategory.MULTI_SECTION)
def aggr_prod(f: Union[Expr, str], group_by: str, **kwargs) -> Expr:
    """聚合求积"""
    return _ensure_expr(f).log().sum().over(group_by).exp()


@register_operator(OperatorCategory.MULTI_SECTION)
def aggr_quantile(f: Union[Expr, str], group_by: str,
                  quantile: float = 0.5, **kwargs) -> Expr:
    """聚合分位数"""
    return _ensure_expr(f).quantile(quantile).over(group_by)


@register_operator(OperatorCategory.MULTI_SECTION)
def merge(factors: List[Union[Expr, str]], weights: Optional[List[float]] = None,
          method: str = "add", **kwargs) -> Expr:
    """合并多个因子"""
    if weights is None:
        weights = [1.0 / len(factors)] * len(factors)
    factors = [_ensure_expr(f) for f in factors]
    weights = list(weights)

    if method == "add":
        result = factors[0] * weights[0]
        for i in range(1, len(factors)):
            result = result + factors[i] * weights[i]
        return result
    elif method == "wavg":
        weighted = sum(f * w for f, w in zip(factors, weights))
        return weighted / sum(weights)
    elif method == "rank":
        ranked = [f.rank() for f in factors]
        result = ranked[0] * weights[0]
        for i in range(1, len(ranked)):
            result = result + ranked[i] * weights[i]
        return result
    elif method == "mul":
        result = factors[0] ** weights[0]
        for i in range(1, len(factors)):
            result = result * factors[i] ** weights[i]
        return result
    return factors[0]


@register_operator(OperatorCategory.MULTI_SECTION)
def chg_ids(f: Union[Expr, str], id_map: Dict[str, str], **kwargs) -> Expr:
    """ID转换"""
    f = _ensure_expr(f)
    return f.replace(list(id_map.keys()), list(id_map.values()))


# ==============================================================================
# 别名: standardizeRank, weightStandardize 需要用 SectionOperators.rank
# ==============================================================================

@register_operator(OperatorCategory.SECTION)
def standardizeRank(f: Union[Expr, str], **kwargs) -> Expr:
    """标准化排名"""
    return SectionOperators.rank(f, method="average")


@register_operator(OperatorCategory.SECTION)
def weightStandardize(f: Union[Expr, str], **kwargs) -> Expr:
    """加权标准化"""
    return standardizeZScore(f, **kwargs)


# ==============================================================================
# 导出
# ==============================================================================

__all__ = [
    # 注册表 API
    "OperatorCategory", "register_operator",
    "list_operators", "get_operator", "operator_info", "generate_documentation",

    # Point 算子
    "abs", "log", "sign", "sqrt", "square", "pow", "clip",
    "ceil", "floor", "fix", "applymap",
    "nanargmax", "nanargmin", "nanmedian", "nanquantile", "nancount", "nanprod",
    "astype", "replace", "fetch",
    "fill_null", "fill_zero", "nan_to_null", "isnull", "notnull",
    "nanmax", "nanmin", "nanmean", "nansum", "nanstd", "nanvar",
    "where", "fillna",

    # Time 算子 - 滚动窗口
    "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
    "rolling_sum", "rolling_median", "rolling_var",
    "rolling_prod", "rolling_skew", "rolling_kurt", "rolling_count",
    "rolling_corr", "rolling_cov", "rolling_quantile", "rolling_rank",
    "rolling_argmax", "rolling_argmin",

    # Time 算子 - 时间序列
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_median",
    "ts_corr", "ts_cov", "ts_rank", "ts_argmax", "ts_argmin",
    "ts_delta", "ts_pct_change", "ts_lag", "ts_lead",

    # Time 算子 - 扩展窗口
    "expanding_mean", "expanding_std", "expanding_sum",
    "expanding_max", "expanding_min", "expanding_median", "expanding_count",
    "expanding_var", "expanding_kurt", "expanding_skew", "expanding_quantile",
    "expanding_corr", "expanding_cov",

    # Time 算子 - 指数加权
    "ewm_mean", "ewm_std", "ewm_var", "ewm_corr", "ewm_cov",

    # Section 算子
    "standardizeZScore", "zscore", "rank", "winsorize",
    "neutralize", "neutralize_market", "scale",
    "standardizeRank", "weightStandardize",
    "ic", "rank_ic", "group_norm", "group_winsorize",
    "orthogonalize", "fillNaNByFun", "fillNaNByRegress",

    # Multi-Section 算子
    "aggregate", "disaggregate",
    "aggr_sum", "aggr_prod", "aggr_max", "aggr_min", "aggr_mean",
    "aggr_std", "aggr_var", "aggr_median", "aggr_quantile", "aggr_count",
    "merge", "chg_ids",

    # 组合算子
    "add", "sub", "mul", "div",
    "weighted_sum", "combine", "if_then_else",

    # 高级算子
    "regress", "zscored", "decay_linear", "decay_exp",
    "vwap", "market_cap", "book_to_market", "earnings_to_market",

    # 别名
    "correlation", "covariance", "delta", "pct_change",
    "delay", "ref", "shift", "diff", "lag",
]

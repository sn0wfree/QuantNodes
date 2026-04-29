# coding=utf-8
"""
因子函数 - Polars 版本

重构说明:
- 内部使用 QuantNodes.operators 中的 Polars 算子
- 装饰器注册表系统，支持动态发现、文档生成
- 纯 Polars 向量化，无 multiprocessing

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
from typing import Any, Dict, List, Optional, Union, Callable

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
    
    Usage:
        @register_operator(OperatorCategory.TIME, "rolling_mean")
        def rolling_mean(f, window=20):
            ...
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
    
    Returns:
        算子名称列表
    """
    if category:
        return list(_OPERATOR_REGISTRY.get(category, {}).keys())
    return [name for cat in _OPERATOR_REGISTRY for name in _OPERATOR_REGISTRY[cat]]


def get_operator(name: str, category: Optional[str] = None) -> Optional[Callable]:
    """根据名称获取算子函数
    
    Args:
        name: 算子名称
        category: 算子分类，可选
    
    Returns:
        算子函数或 None
    """
    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op["func"] if op else None
    
    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]["func"]
    return None


def operator_info(name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """获取算子详细信息
    
    Args:
        name: 算子名称
        category: 算子分类，可选
    
    Returns:
        包含 name, category, doc, signature, parameters 的字典
    """
    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op if op else None
    
    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]
    return None


def generate_documentation(output_format: str = "markdown", category: Optional[str] = None) -> str:
    """生成算子文档
    
    Args:
        output_format: 输出格式 (markdown/text/json)
        category: 算子分类筛选
    
    Returns:
        格式化的文档字符串
    """
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

def _to_expr(f: Union[Expr, str]) -> Expr:
    """转换为 Polars 表达式"""
    if isinstance(f, str):
        return pl.col(f)
    return f


def _ensure_expr(f: Any) -> Expr:
    """确保是表达式"""
    if isinstance(f, pl.Expr):
        return f
    if isinstance(f, str):
        return pl.col(f)
    return pl.lit(f)


# ==============================================================================
# 时间序列算子 - 滚动窗口
# ==============================================================================

def rolling_mean(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口均值
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_mean(f, window, min_periods)


def rolling_std(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    **kwargs
) -> Expr:
    """滚动窗口标准差"""
    return TimeSeriesOperators.ts_std(f, window, min_periods, ddof)


def rolling_max(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口最大值"""
    return TimeSeriesOperators.ts_max(f, window, min_periods)


def rolling_min(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口最小值"""
    return TimeSeriesOperators.ts_min(f, window, min_periods)


def rolling_sum(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口求和"""
    return TimeSeriesOperators.ts_sum(f, window, min_periods)


def rolling_median(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口中位数"""
    return TimeSeriesOperators.ts_median(f, window, min_periods)


def rolling_var(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口方差"""
    expr = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return expr.rolling_var(window, min_samples=mp)


def ts_mean(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """时间序列均值 (滚动均值别名)"""
    return rolling_mean(f, window, **kwargs)


def ts_std(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """时间序列标准差"""
    return rolling_std(f, window, **kwargs)


def ts_max(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """时间序列最大值"""
    return rolling_max(f, window, **kwargs)


def ts_min(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """时间序列最小值"""
    return rolling_min(f, window, **kwargs)


def ts_sum(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """时间序列求和"""
    return rolling_sum(f, window, **kwargs)


def ts_median(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """时间序列中位数"""
    return rolling_median(f, window, **kwargs)


# ==============================================================================
# 时间序列算子 - 相关系数
# ==============================================================================

def ts_corr(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动相关系数
    
    Args:
        f1: 第一个表达式或列名
        f2: 第二个表达式或列名
        window: 窗口大小
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_corr(f1, f2, window, min_periods)


def ts_cov(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动协方差"""
    return TimeSeriesOperators.ts_cov(f1, f2, window, min_periods)


def correlation(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """相关系数 (ts_corr 别名)"""
    return ts_corr(f1, f2, window, **kwargs)


def covariance(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """协方差 (ts_cov 别名)"""
    return ts_cov(f1, f2, window, **kwargs)


# ==============================================================================
# 时间序列算子 - 排名
# ==============================================================================

def ts_rank(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动排名 (0-1 归一化)
    
    Args:
        f: 表达式或列名
        window: 窗口大小
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_rank(f, window, min_periods)


def ts_argmax(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动最大值的位置（相对于当前行的偏移量）"""
    return rolling_argmax(f, window, min_periods, **kwargs)


def ts_argmin(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动最小值的位置（相对于当前行的偏移量）"""
    return rolling_argmin(f, window, min_periods, **kwargs)


# ==============================================================================
# 时间序列算子 - 差分与变化
# ==============================================================================

def ts_delta(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """差分
    
    Args:
        f: 表达式或列名
        periods: 差分阶数
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_delta(f, periods)


def ts_pct_change(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """百分比变化"""
    return TimeSeriesOperators.ts_pct_change(f, periods)


def delta(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """差分 (ts_delta 别名)"""
    return ts_delta(f, periods, **kwargs)


def pct_change(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """百分比变化 (ts_pct_change 别名)"""
    return ts_pct_change(f, periods, **kwargs)


# ==============================================================================
# 时间序列算子 - 滞后与前向
# ==============================================================================

def ts_lag(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """滞后 (向后移动)
    
    Args:
        f: 表达式或列名
        periods: 滞后阶数
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_lag(f, periods)


def ts_lead(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """前向移动 (向前移动)"""
    return TimeSeriesOperators.ts_lead(f, periods)


def delay(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """滞后 (ts_lag 别名)"""
    return ts_lag(f, periods, **kwargs)


def ref(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """引用历史值 (delay 别名)"""
    return delay(f, periods, **kwargs)


def shift(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """移动 (shift 别名)"""
    return ts_lag(f, periods, **kwargs)


# ==============================================================================
# 时间序列算子 - 扩展窗口
# ==============================================================================

def expanding_mean(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口均值 (累计至当前)
    
    Args:
        f: 表达式或列名
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    expr = _to_expr(f)
    return expr.cum_sum() / (pl.arange(0, pl.col("x").count()) + 1)


def expanding_std(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口标准差"""
    e = _to_expr(f)
    n = pl.int_range(0, pl.len()) + 1
    mean = e.cum_sum() / n
    mean_sq = (e ** 2).cum_sum() / n
    var = mean_sq - mean ** 2
    return (var + 1e-10).sqrt()


def expanding_sum(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口求和"""
    expr = _to_expr(f)
    return expr.cum_sum()


def expanding_max(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口最大值"""
    e = _ensure_expr(f)
    return e.cum_max()


def expanding_min(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口最小值"""
    e = _ensure_expr(f)
    return e.cum_min()


def expanding_median(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口中位数

    用 sort + gather 实现：对每个位置取前 n 个元素的中位数。
    Polars 没有 cum_median，通过 map_batches 逐步计算。
    """
    e = _ensure_expr(f)

    def _cum_median(s: pl.Series) -> pl.Series:
        vals = s.to_list()
        result = []
        for i in range(len(vals)):
            window = [v for v in vals[:i + 1] if v is not None]
            if len(window) == 0:
                result.append(None)
            else:
                window.sort()
                n = len(window)
                mid = n // 2
                if n % 2 == 1:
                    result.append(window[mid])
                else:
                    result.append((window[mid - 1] + window[mid]) / 2)
        return pl.Series(values=result)

    return e.map_batches(_cum_median, return_dtype=pl.Float64)


def expanding_count(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口计数（非空值累计数量）"""
    e = _ensure_expr(f)
    return e.is_not_null().cast(pl.Int64).cum_sum()


def expanding_var(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口方差"""
    e = _ensure_expr(f)
    n = pl.int_range(0, pl.len()) + 1
    mean = e.cum_sum() / n
    mean_sq = (e ** 2).cum_sum() / n
    return mean_sq - mean ** 2


def expanding_kurt(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口峰度

    用 map_batches 逐步计算每个扩展窗口的峰度。
    """
    e = _ensure_expr(f)

    def _cum_kurt(s: pl.Series) -> pl.Series:
        import numpy as np
        vals = s.to_list()
        result = []
        for i in range(len(vals)):
            window = [v for v in vals[:i + 1] if v is not None]
            if len(window) < 4:
                result.append(None)
            else:
                arr = np.array(window, dtype=np.float64)
                m = arr.mean()
                s2 = arr.std(ddof=1)
                if s2 < 1e-15:
                    result.append(None)
                else:
                    result.append(float(np.mean(((arr - m) / s2) ** 4) - 3))
        return pl.Series(values=result)

    return e.map_batches(_cum_kurt, return_dtype=pl.Float64)


def expanding_skew(
    f: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口偏度

    用 map_batches 逐步计算每个扩展窗口的偏度。
    """
    e = _ensure_expr(f)

    def _cum_skew(s: pl.Series) -> pl.Series:
        import numpy as np
        vals = s.to_list()
        result = []
        for i in range(len(vals)):
            window = [v for v in vals[:i + 1] if v is not None]
            if len(window) < 3:
                result.append(None)
            else:
                arr = np.array(window, dtype=np.float64)
                m = arr.mean()
                s2 = arr.std(ddof=1)
                if s2 < 1e-15:
                    result.append(None)
                else:
                    n = len(arr)
                    result.append(float(n / ((n - 1) * (n - 2)) * np.sum(((arr - m) / s2) ** 3)))
        return pl.Series(values=result)

    return e.map_batches(_cum_skew, return_dtype=pl.Float64)


def expanding_quantile(
    f: Union[Expr, str],
    quantile: float = 0.5,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口分位数

    用 map_batches 逐步计算每个扩展窗口的分位数。
    """
    e = _ensure_expr(f)

    def _cum_quantile(s: pl.Series) -> pl.Series:
        vals = s.to_list()
        result = []
        for i in range(len(vals)):
            window = [v for v in vals[:i + 1] if v is not None]
            if len(window) == 0:
                result.append(None)
            else:
                window.sort()
                n = len(window)
                idx = quantile * (n - 1)
                lo = int(idx)
                hi = min(lo + 1, n - 1)
                frac = idx - lo
                result.append(window[lo] * (1 - frac) + window[hi] * frac)
        return pl.Series(values=result)

    return e.map_batches(_cum_quantile, return_dtype=pl.Float64)


def expanding_corr(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口相关系数（双因子）

    用 map_batches 逐步计算每个扩展窗口的 Pearson 相关系数。
    """
    e1 = _ensure_expr(f1)
    e2 = _ensure_expr(f2)

    def _cum_corr(args: list) -> pl.Series:
        import numpy as np
        s1, s2 = args[0], args[1]
        v1 = s1.to_list()
        v2 = s2.to_list()
        result = []
        for i in range(len(v1)):
            pairs = [(a, b) for a, b in zip(v1[:i + 1], v2[:i + 1]) if a is not None and b is not None]
            if len(pairs) < 2:
                result.append(None)
            else:
                arr1, arr2 = zip(*pairs)
                c = np.corrcoef(arr1, arr2)[0, 1]
                result.append(float(c) if np.isfinite(c) else None)
        return pl.Series(values=result)

    return pl.map_batches([e1, e2], _cum_corr, return_dtype=pl.Float64)


def expanding_cov(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """扩展窗口协方差（双因子）

    用 map_batches 逐步计算每个扩展窗口的协方差。
    """
    e1 = _ensure_expr(f1)
    e2 = _ensure_expr(f2)

    def _cum_cov(args: list) -> pl.Series:
        import numpy as np
        s1, s2 = args[0], args[1]
        v1 = s1.to_list()
        v2 = s2.to_list()
        result = []
        for i in range(len(v1)):
            pairs = [(a, b) for a, b in zip(v1[:i + 1], v2[:i + 1]) if a is not None and b is not None]
            if len(pairs) < 2:
                result.append(None)
            else:
                arr1, arr2 = zip(*pairs)
                c = np.cov(arr1, arr2)[0, 1]
                result.append(float(c) if np.isfinite(c) else None)
        return pl.Series(values=result)

    return pl.map_batches([e1, e2], _cum_cov, return_dtype=pl.Float64)


# ==============================================================================
# 时间序列算子 - 指数加权
# ==============================================================================

def ewm_mean(
    f: Union[Expr, str],
    alpha: float = 0.5,
    adjust: bool = True,
    **kwargs
) -> Expr:
    """指数加权移动平均
    
    Args:
        f: 表达式或列名
        alpha: 平滑因子
        adjust: 是否调整
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ewm_mean(f, alpha, adjust=adjust)


def ewm_std(
    f: Union[Expr, str],
    alpha: float = 0.5,
    adjust: bool = True,
    **kwargs
) -> Expr:
    """指数加权移动标准差"""
    return TimeSeriesOperators.ewm_std(f, alpha, adjust=adjust)


def ewm_corr(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    alpha: float = 0.5,
    **kwargs
) -> Expr:
    """指数加权相关系数"""
    return TimeSeriesOperators.ewm_corr(f1, f2, alpha=alpha)


def ewm_var(
    f: Union[Expr, str],
    alpha: float = 0.5,
    adjust: bool = True,
    **kwargs
) -> Expr:
    """指数加权移动方差
    
    Args:
        f: 表达式或列名
        alpha: 平滑因子
        adjust: 是否调整
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    mean = e.ewm_mean(alpha=alpha, adjust=adjust)
    mean_sq = (e ** 2).ewm_mean(alpha=alpha, adjust=adjust)
    return mean_sq - mean ** 2


def ewm_cov(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    alpha: float = 0.5,
    adjust: bool = True,
    **kwargs
) -> Expr:
    """指数加权移动协方差（双因子）

    Args:
        f1: 第一个表达式或列名
        f2: 第二个表达式或列名
        alpha: 平滑因子
        adjust: 是否调整
    """
    e1 = _ensure_expr(f1)
    e2 = _ensure_expr(f2)
    mean1 = e1.ewm_mean(alpha=alpha, adjust=adjust)
    mean2 = e2.ewm_mean(alpha=alpha, adjust=adjust)
    mean12 = (e1 * e2).ewm_mean(alpha=alpha, adjust=adjust)
    return mean12 - mean1 * mean2


def rolling_prod(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口求积
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_prod(f, window, min_periods)


def rolling_skew(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口偏度
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    min_periods = min_periods or max(1, window // 2)
    
    mean = e.rolling_mean(window, min_samples=min_periods)
    std = e.rolling_std(window, min_samples=min_periods)
    n = window
    
    m3 = ((e - mean) ** 3).rolling_mean(window, min_samples=min_periods)
    return m3 / (std ** 3 + 1e-10)


def rolling_kurt(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口峰度
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    min_periods = min_periods or max(1, window // 2)
    
    mean = e.rolling_mean(window, min_samples=min_periods)
    std = e.rolling_std(window, min_samples=min_periods)
    
    m4 = ((e - mean) ** 4).rolling_mean(window, min_samples=min_periods)
    return m4 / (std ** 4 + 1e-10) - 3


def rolling_count(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口计数
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    min_periods = min_periods or max(1, window // 2)
    return e.is_not_null().cast(pl.Int64).rolling_sum(window, min_samples=min_periods)


def rolling_corr(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口相关系数（双因子）

    Args:
        f1: 第一个表达式或列名
        f2: 第二个表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    """
    e1 = _ensure_expr(f1)
    e2 = _ensure_expr(f2)
    min_periods = min_periods or max(1, window // 2)
    return TimeSeriesOperators.ts_corr(e1, e2, window, min_periods)


def rolling_cov(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口协方差（双因子）

    Args:
        f1: 第一个表达式或列名
        f2: 第二个表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    """
    e1 = _ensure_expr(f1)
    e2 = _ensure_expr(f2)
    min_periods = min_periods or max(1, window // 2)
    return TimeSeriesOperators.ts_cov(e1, e2, window, min_periods)


def rolling_quantile(
    f: Union[Expr, str],
    window: int = 20,
    quantile: float = 0.5,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口分位数

    Args:
        f: 表达式或列名
        window: 窗口大小
        quantile: 分位数 (0-1)
        min_periods: 最小观测数
    """
    e = _ensure_expr(f)
    return e.rolling_quantile(quantile, window_size=window, interpolation="nearest")


def rolling_rank(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口排名

    返回当前值在窗口内的排名（归一化到 0-1）。

    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    """
    e = _ensure_expr(f)
    return e.rolling_rank(window)


def _rolling_arg_op(f: Union[Expr, str], window: int, op: str, min_periods: Optional[int] = None) -> Expr:
    """内部函数：用 shift 比较链实现 rolling argmax/argmin
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        op: "max" 或 "min"
        min_periods: 最小观测数
        
    Returns:
        最大值/最小值的相对偏移量（0=当前行，1=前一行...）
    """
    e = _ensure_expr(f)
    min_periods = min_periods or max(1, window // 2)
    
    max_window = min(window, 30)  # 限制最大窗口，避免表达式过大
    
    cmp = lambda a, b: a >= b if op == "max" else a <= b
    result = pl.lit(None, dtype=pl.Int32)
    
    for i in range(max_window):
        shifted_i = e.shift(i)
        is_best = pl.lit(True)
        for j in range(max_window):
            if i != j:
                shifted_j = e.shift(j)
                is_best = is_best & cmp(shifted_i, shifted_j)
        result = pl.when(is_best).then(pl.lit(i, dtype=pl.Int32)).otherwise(result)
    
    return result


def rolling_argmax(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口最大值索引（相对于当前行的偏移量）
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        最大值的相对偏移量（0=当前行，1=前一行...）
    """
    return _rolling_arg_op(f, window, "max", min_periods)


def rolling_argmin(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动窗口最小值索引（相对于当前行的偏移量）
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        最小值的相对偏移量（0=当前行，1=前一行...）
    """
    return _rolling_arg_op(f, window, "min", min_periods)


def diff(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """差分
    
    Args:
        f: 表达式或列名
        periods: 差分阶数
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_delta(f, periods)


def lag(
    f: Union[Expr, str],
    periods: int = 1,
    **kwargs
) -> Expr:
    """滞后算子
    
    Args:
        f: 表达式或列名
        periods: 滞后阶数
    
    Returns:
        Polars 表达式
    """
    return TimeSeriesOperators.ts_lag(f, periods)


# ==============================================================================
# 截面算子
# ==============================================================================

def standardizeZScore(
    f: Union[Expr, str],
    eps: float = 1e-8,
    **kwargs
) -> Expr:
    """Z-score 标准化
    
    Args:
        f: 表达式或列名
        eps: 防止除零的常数
    
    Returns:
        Polars 表达式
    """
    return SectionOperators.zscore(f, eps)


def zscore(
    f: Union[Expr, str],
    eps: float = 1e-8,
    **kwargs
) -> Expr:
    """Z-score 标准化 (standardizeZScore 别名)"""
    return standardizeZScore(f, eps, **kwargs)


def rank(
    f: Union[Expr, str],
    method: str = "dense",
    **kwargs
) -> Expr:
    """截面排名 (归一化到 0-1)
    
    Args:
        f: 表达式或列名
        method: 排名方法 (dense/ordinal/average/min/max)
    
    Returns:
        Polars 表达式
    """
    return SectionOperators.rank(f, method)


def winsorize(
    f: Union[Expr, str],
    lower: float = 0.01,
    upper: float = 0.01,
    method: str = "quantile",
    **kwargs
) -> Expr:
    """去极值
    
    Args:
        f: 表达式或列名
        lower: 下界比例
        upper: 上界比例
        method: 方法
    
    Returns:
        Polars 表达式
    """
    return SectionOperators.winsorize(f, lower, upper, method)


def neutralize(
    f: Union[Expr, str],
    group: Optional[Union[Expr, str]] = None,
    **kwargs
) -> Expr:
    """行业中性的 (减去行业均值)
    
    Args:
        f: 表达式或列名
        group: 分组表达式或列名
    
    Returns:
        Polars 表达式
    """
    if group:
        return SectionOperators.neutralize(f, group)
    return SectionOperators.neutralize_market(f)


def neutralize_market(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """市场中性 (减去市场均值)"""
    return SectionOperators.neutralize_market(f)


def scale(
    f: Union[Expr, str],
    method: str = "zscore",
    **kwargs
) -> Expr:
    """归一化
    
    Args:
        f: 表达式或列名
        method: 方法 (zscore/minmax/abs)
    
    Returns:
        Polars 表达式
    """
    return SectionOperators.scale(f, method)


def standardizeRank(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """标准化排名"""
    return SectionOperators.rank(f, method="average")


def weightStandardize(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """加权标准化"""
    return standardizeZScore(f, **kwargs)


def ic(
    f: Union[Expr, str],
    target: Union[Expr, str],
    **kwargs
) -> Expr:
    """IC (Pearson 相关系数)
    
    Args:
        f: 因子表达式
        target: 目标表达式
    
    Returns:
        Polars 表达式
    """
    return SectionOperators.ic(f, target)


def rank_ic(
    f: Union[Expr, str],
    target: Union[Expr, str],
    **kwargs
) -> Expr:
    """Rank IC (Spearman 相关系数)"""
    return SectionOperators.rank_ic(f, target)


def group_norm(
    f: Union[Expr, str],
    group: Union[Expr, str],
    method: str = "zscore",
    **kwargs
) -> Expr:
    """分组标准化
    
    Args:
        f: 表达式或列名
        group: 分组列名
        method: 方法
    
    Returns:
        Polars 表达式
    """
    return SectionOperators.group_norm(f, group, method)


def group_winsorize(
    f: Union[Expr, str],
    group: Union[Expr, str],
    lower: float = 0.01,
    upper: float = 0.01,
    **kwargs
) -> Expr:
    """分组去极值"""
    return SectionOperators.group_winsorize(f, group, lower, upper)


def orthogonalize(
    f: Union[Expr, str],
    reference: Union[Expr, str],
    **kwargs
) -> Expr:
    """正交化：从因子 f 中剔除 reference 的影响
    
    Args:
        f: 目标因子
        reference: 参考因子
    
    Returns:
        正交化后的表达式
    """
    f = _ensure_expr(f)
    reference = _ensure_expr(reference)
    
    cov = (f * reference).mean() - f.mean() * reference.mean()
    var_ref = (reference ** 2).mean() - reference.mean() ** 2
    beta = cov / (var_ref + 1e-10)
    return f - beta * reference


def fillNaNByFun(
    f: Union[Expr, str],
    value: Any = 0,
    **kwargs
) -> Expr:
    """按值填充 NaN
    
    Args:
        f: 表达式或列名
        value: 填充值
    
    Returns:
        填充后的表达式
    """
    f = _ensure_expr(f)
    if callable(value):
        return f.map_elements(value)
    return f.fill_null(value)


def fillNaNByRegress(
    f: Union[Expr, str],
    reference: Union[Expr, str],
    **kwargs
) -> Expr:
    """按回归值填充 NaN（用 reference 回归预测缺失值）
    
    Args:
        f: 目标因子（包含NaN）
        reference: 参考因子
    
    Returns:
        填充后的表达式
    """
    f = _ensure_expr(f)
    reference = _ensure_expr(reference)
    
    cov = (f * reference).mean() - f.mean() * reference.mean()
    var_ref = (reference ** 2).mean() - reference.mean() ** 2
    beta = cov / (var_ref + 1e-10)
    alpha = f.mean() - beta * reference.mean()
    predicted = alpha + beta * reference
    return f.fill_null(predicted)


def where(
    condition: Union[Expr, str],
    true_val: Union[Expr, str, Any],
    false_val: Union[Expr, str, Any] = None,
    **kwargs
) -> Expr:
    """条件选择

    Args:
        condition: 条件表达式或列名
        true_val: 条件为真时的值
        false_val: 条件为假时的值
    """
    cond = _ensure_expr(condition)
    if isinstance(true_val, (int, float, str)):
        t = pl.lit(true_val)
    else:
        t = _ensure_expr(true_val)
    if false_val is None:
        f = pl.lit(None)
    elif isinstance(false_val, (int, float, str)):
        f = pl.lit(false_val)
    else:
        f = _ensure_expr(false_val)
    return pl.when(cond).then(t).otherwise(f)


def fillna(
    f: Union[Expr, str],
    value: Any = None,
    method: str = "ffill",
    limit: int = 0,
    **kwargs
) -> Expr:
    """填充空值

    Args:
        f: 表达式或列名
        value: 填充值（与 method 二选一）
        method: 填充方法 (ffill/bfill)
        limit: 前向/后向填充的最大连续数量
    """
    e = _ensure_expr(f)
    if value is not None:
        return e.fill_null(value)
    if method == "ffill":
        return e.forward_fill(limit) if limit > 0 else e.forward_fill()
    elif method == "bfill":
        return e.backward_fill(limit) if limit > 0 else e.backward_fill()
    return e


# ==============================================================================
# 数学算子
# ==============================================================================

def abs(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """绝对值
    
    Args:
        f: 表达式或列名
    
    Returns:
        Polars 表达式
    """
    return MathOperators.abs(f)


def log(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """对数
    
    Args:
        f: 表达式或列名
    
    Returns:
        Polars 表达式
    """
    return MathOperators.log(f)


def sign(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """符号
    
    Args:
        f: 表达式或列名
    
    Returns:
        Polars 表达式
    """
    return MathOperators.sign(f)


def sqrt(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """平方根
    
    Args:
        f: 表达式或列名
    
    Returns:
        Polars 表达式
    """
    return MathOperators.sqrt(f)


def square(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """平方
    
    Args:
        f: 表达式或列名
    
    Returns:
        Polars 表达式
    """
    return _ensure_expr(f) ** 2


def pow(
    f: Union[Expr, str],
    exponent: float = 2.0,
    **kwargs
) -> Expr:
    """幂运算
    
    Args:
        f: 表达式或列名
        exponent: 指数
    
    Returns:
        Polars 表达式
    """
    return MathOperators.pow(f, exponent)


def clip(
    f: Union[Expr, str],
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    **kwargs
) -> Expr:
    """裁剪
    
    Args:
        f: 表达式或列名
        lower: 下界
        upper: 上界
    
    Returns:
        Polars 表达式
    """
    return MathOperators.clip(f, lower, upper)


def fill_null(
    f: Union[Expr, str],
    value: float = 0.0,
    **kwargs
) -> Expr:
    """填充 null
    
    Args:
        f: 表达式或列名
        value: 填充值
    
    Returns:
        Polars 表达式
    """
    return MathOperators.fill_null(f, value)


def fill_zero(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """填充 0"""
    return MathOperators.fill_zero(f)


def nan_to_null(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """NaN 转 null"""
    return MathOperators.nan_to_null(f)


def isnull(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """判断空值
    
    Args:
        f: 表达式或列名
    
    Returns:
        Polars 表达式 (布尔)
    """
    return _ensure_expr(f).is_null()


def notnull(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """判断非空"""
    return _ensure_expr(f).is_not_null()


# ==============================================================================
# Point 算子 (补充)
# ==============================================================================

def ceil(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """向上取整
    
    Args:
        f: 表达式或列名
    
    Returns:
        向上取整表达式
    """
    return _ensure_expr(f).ceil()


def floor(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """向下取整
    
    Args:
        f: 表达式或列名
    
    Returns:
        向下取整表达式
    """
    return _ensure_expr(f).floor()


def fix(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """向零取整
    
    Args:
        f: 表达式或列名
    
    Returns:
        向零取整表达式
    """
    e = _ensure_expr(f)
    return pl.when(e < 0).then(e.ceil()).otherwise(e.floor())


def applymap(
    f: Union[Expr, str],
    func: Callable,
    **kwargs
) -> Expr:
    """应用自定义函数
    
    Args:
        f: 表达式或列名
        func: 自定义函数
    
    Returns:
        转换后的表达式
    """
    return _ensure_expr(f).map_elements(func)


def nanargmax(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """忽略空值求最大值索引
    
    Args:
        f: 表达式或列名
    
    Returns:
        最大值索引表达式
    """
    e = _ensure_expr(f)
    return e.arg_max()


def nanargmin(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """忽略空值求最小值索引
    
    Args:
        f: 表达式或列名
    
    Returns:
        最小值索引表达式
    """
    e = _ensure_expr(f)
    return e.arg_min()


def nanmedian(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """忽略空值求中位数
    
    Args:
        f: 表达式或列名
    
    Returns:
        中位数表达式
    """
    return _ensure_expr(f).median()


def nanquantile(
    f: Union[Expr, str],
    quantile: float = 0.5,
    interpolation: str = "nearest",
    **kwargs
) -> Expr:
    """忽略空值求分位数
    
    Args:
        f: 表达式或列名
        quantile: 分位数 (0-1)
        interpolation: 插值方法 (nearest/higher/lower/midpoint/linear)
    
    Returns:
        分位数表达式
    """
    return _ensure_expr(f).quantile(quantile, interpolation=interpolation)


def nancount(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """统计非空值数量
    
    Args:
        f: 表达式或列名
    
    Returns:
        非空值数量表达式
    """
    return _ensure_expr(f).count()


def nanprod(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """忽略空值求积
    
    Args:
        f: 表达式或列名
    
    Returns:
        求积表达式
    """
    return _ensure_expr(f).product()


def astype(
    f: Union[Expr, str],
    dtype: str = "float64",
    **kwargs
) -> Expr:
    """类型转换
    
    Args:
        f: 表达式或列名
        dtype: 目标类型 (float64/int64/utf8/bool)
    
    Returns:
        类型转换后的表达式
    """
    type_map = {
        "float64": pl.Float64,
        "float32": pl.Float32,
        "int64": pl.Int64,
        "int32": pl.Int32,
        "int8": pl.Int8,
        "utf8": pl.Utf8,
        "bool": pl.Boolean,
    }
    return _ensure_expr(f).cast(type_map.get(dtype, pl.Float64))


def replace(
    f: Union[Expr, str],
    old: Any,
    new: Any,
    **kwargs
) -> Expr:
    """值替换
    
    Args:
        f: 表达式或列名
        old: 旧值
        new: 新值
    
    Returns:
        值替换后的表达式
    """
    return _ensure_expr(f).replace(old, new)


def fetch(
    f: Union[Expr, str],
    index: int = 0,
    **kwargs
) -> Expr:
    """获取指定位置数据
    
    Args:
        f: 表达式或列名
        index: 位置索引
    
    Returns:
        指定位置数据表达式
    """
    return _ensure_expr(f).list.get(index)


# ==============================================================================
# NaN 跨截面聚合算子
# ==============================================================================

def nanmax(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """跨截面忽略空值求最大值

    Args:
        f: 表达式或列名
    """
    e = _ensure_expr(f)
    return e.max()


def nanmin(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """跨截面忽略空值求最小值

    Args:
        f: 表达式或列名
    """
    e = _ensure_expr(f)
    return e.min()


def nanmean(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """跨截面忽略空值求均值

    Args:
        f: 表达式或列名
    """
    e = _ensure_expr(f)
    return e.mean()


def nansum(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """跨截面忽略空值求和

    Args:
        f: 表达式或列名
    """
    e = _ensure_expr(f)
    return e.sum()


def nanstd(
    f: Union[Expr, str],
    ddof: int = 1,
    **kwargs
) -> Expr:
    """跨截面忽略空值求标准差

    Args:
        f: 表达式或列名
        ddof: 自由度调整
    """
    e = _ensure_expr(f)
    return e.std(ddof=ddof)


def nanvar(
    f: Union[Expr, str],
    ddof: int = 1,
    **kwargs
) -> Expr:
    """跨截面忽略空值求方差

    Args:
        f: 表达式或列名
        ddof: 自由度调整
    """
    e = _ensure_expr(f)
    return e.var(ddof=ddof)


# ==============================================================================
# 组合算子
# ==============================================================================

def add(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """加法
    
    Args:
        f1: 第一个表达式或列名
        f2: 第二个表达式或列名
    
    Returns:
        Polars 表达式
    """
    return _ensure_expr(f1) + _ensure_expr(f2)


def sub(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """减法"""
    return _ensure_expr(f1) - _ensure_expr(f2)


def mul(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """乘法"""
    return _ensure_expr(f1) * _ensure_expr(f2)


def div(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """除法"""
    return _ensure_expr(f1) / _ensure_expr(f2)


def weighted_sum(
    factors: List[Union[Expr, str]],
    weights: Optional[List[float]] = None,
    **kwargs
) -> Expr:
    """加权求和
    
    Args:
        factors: 因子列表
        weights: 权重列表
    
    Returns:
        Polars 表达式
    """
    return CompositeOperators.weighted_sum(factors, weights)


def combine(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    method: str = "add",
    **kwargs
) -> Expr:
    """组合因子
    
    Args:
        f1: 第一个因子
        f2: 第二个因子
        method: 组合方法 add/mul/max/min
    """
    method_map = {
        "add": lambda a, b: a + b,
        "sum": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "max": lambda a, b: pl.max_horizontal(a, b),
        "min": lambda a, b: pl.min_horizontal(a, b),
    }
    a = _ensure_expr(f1)
    b = _ensure_expr(f2)
    return method_map.get(method, method_map["add"])(a, b)


def if_then_else(
    condition: Union[Expr, str],
    then: Union[Expr, str],
    else_: Union[Expr, str],
    **kwargs
) -> Expr:
    """条件表达式
    
    Args:
        condition: 条件表达式
        then: 条件为真时的值
        else_: 条件为假时的值
    """
    return pl.when(_ensure_expr(condition)).then(_ensure_expr(then)).otherwise(_ensure_expr(else_))


# ==============================================================================
# 高级算子
# ==============================================================================

def regress(
    y: Union[Expr, str],
    x: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """滑动窗口线性回归的残差
    
    Args:
        y: 因变量
        x: 自变量
        window: 窗口大小
    
    Returns:
        残差表达式
    """
    ey = _to_expr(y)
    ex = _to_expr(x)
    
    y_mean = ey.rolling_mean(window)
    x_mean = ex.rolling_mean(window)
    x_var = ex.rolling_var(window)
    
    # 滚动协方差: cov = mean(xy) - mean(x)mean(y)
    xy_cov = (ey * ex).rolling_mean(window) - y_mean * x_mean
    
    beta = xy_cov / (x_var + 1e-8)
    return ey - y_mean - beta * (ex - x_mean)


def zscored(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """滚动 Z-score
    
    Args:
        f: 表达式或列名
        window: 窗口大小
    """
    e = _ensure_expr(f)
    return (e - e.rolling_mean(window)) / (e.rolling_std(window) + 1e-8)


def decay_linear(
    f: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """线性衰减加权
    
    Args:
        f: 表达式或列名
        window: 窗口大小
    """
    import numpy as np
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    expr = _to_expr(f)
    result = expr * weights[0]
    for i in range(1, window):
        result = result + expr.shift(i) * weights[i]
    return result


def decay_exp(
    f: Union[Expr, str],
    halflife: int = 10,
    **kwargs
) -> Expr:
    """指数衰减加权
    
    Args:
        f: 表达式或列名
        halflife: 半衰期
    """
    import numpy as np
    alpha = 0.5 ** (1 / halflife)
    weights = alpha ** np.arange(halflife)
    weights = weights / weights.sum()
    
    expr = _to_expr(f)
    result = expr * weights[0]
    for i in range(1, len(weights)):
        result = result + expr.shift(i) * weights[i]
    return result


def vwap(
    price: Union[Expr, str],
    volume: Union[Expr, str],
    window: int = 20,
    **kwargs
) -> Expr:
    """成交量加权平均价
    
    Args:
        price: 价格表达式或列名
        volume: 成交量表达式或列名
        window: 窗口大小
    """
    expr_price = _to_expr(price)
    expr_volume = _to_expr(volume)
    return (expr_price * expr_volume).rolling_sum(window) / (expr_volume.rolling_sum(window) + 1e-8)


def market_cap(
    price: Union[Expr, str],
    shares: Union[Expr, str],
    **kwargs
) -> Expr:
    """市值
    
    Args:
        price: 价格
        shares: 股份数
    """
    return _to_expr(price) * _to_expr(shares)


def book_to_market(
    book_value: Union[Expr, str],
    market_cap: Union[Expr, str],
    **kwargs
) -> Expr:
    """市净率"""
    return _to_expr(book_value) / _to_expr(market_cap)


def earnings_to_market(
    earnings: Union[Expr, str],
    market_cap: Union[Expr, str],
    **kwargs
) -> Expr:
    """盈利市率"""
    return _to_expr(earnings) / _to_expr(market_cap)


# ==============================================================================
# 导出
# ==============================================================================

__all__ = [
    # 滚动窗口
    "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
    "rolling_sum", "rolling_median", "rolling_var",
    
    # 时间序列
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_median",
    "ts_corr", "ts_cov", "ts_rank", "ts_argmax", "ts_argmin",
    "ts_delta", "ts_pct_change", "ts_lag", "ts_lead",
    
    # 扩展窗口
    "expanding_mean", "expanding_std", "expanding_sum",
    
    # 指数加权
    "ewm_mean", "ewm_std", "ewm_corr",
    
    # 截面
    "standardizeZScore", "zscore", "rank", "winsorize",
    "neutralize", "neutralize_market", "scale",
    "standardizeRank", "weightStandardize",
    "ic", "rank_ic", "group_norm", "group_winsorize",
    
    # 数学
    "abs", "log", "sign", "sqrt", "square", "pow",
    "clip", "fill_null", "fill_zero", "nan_to_null",
    "isnull", "notnull",
    
    # 组合
    "add", "sub", "mul", "div",
    "weighted_sum", "combine", "if_then_else",
    
    # 高级
    "regress", "zscored", "decay_linear", "decay_exp",
    "vwap", "market_cap", "book_to_market", "earnings_to_market",
    
    # 别名
    "correlation", "covariance", "delta", "pct_change",
    "delay", "ref", "shift",
]


# ==============================================================================
# Multi-Section 算子
# ==============================================================================

def aggregate(
    f: Union[Expr, str],
    group_by: str,
    method: str = "mean",
    **kwargs
) -> Expr:
    """按组聚合
    
    Args:
        f: 表达式或列名
        group_by: "group by" column
        method: 聚合方法 (mean/sum/std/var/median/min/max)
    
    Returns:
        聚合后的表达式
    """
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


def disaggregate(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """解聚合 (将聚合值展开到组内每个成员)
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        解聚合表达式
    """
    f = _ensure_expr(f)
    return f.over(group_by)


def aggr_sum(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合求和
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合求和表达式
    """
    return aggregate(f, group_by, "sum")


def aggr_prod(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合求积
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合求积表达式
    """
    import numpy as np
    f = _ensure_expr(f)
    return f.log().sum().over(group_by).exp()


def aggr_max(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合最大值
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合最大值表达式
    """
    return aggregate(f, group_by, "max")


def aggr_min(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合最小值
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合最小值表达式
    """
    return aggregate(f, group_by, "min")


def aggr_mean(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合均值
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合均值表达式
    """
    return aggregate(f, group_by, "mean")


def aggr_std(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合标准差
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合标准差表达式
    """
    return aggregate(f, group_by, "std")


def aggr_var(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合方差
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合方差表达式
    """
    return aggregate(f, group_by, "var")


def aggr_median(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合中位数
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合中位数表达式
    """
    return aggregate(f, group_by, "median")


def aggr_quantile(
    f: Union[Expr, str],
    group_by: str,
    quantile: float = 0.5,
    **kwargs
) -> Expr:
    """聚合分位数
    
    Args:
        f: 表达式或列名
        group_by: 分组列
        quantile: 分位数
    
    Returns:
        聚合分位数表达式
    """
    f = _ensure_expr(f)
    return f.quantile(quantile).over(group_by)


def aggr_count(
    f: Union[Expr, str],
    group_by: str,
    **kwargs
) -> Expr:
    """聚合计数
    
    Args:
        f: 表达式或列名
        group_by: 分组列
    
    Returns:
        聚合计数表达式
    """
    return aggregate(f, group_by, "count")


def merge(
    factors: List[Union[Expr, str]],
    weights: Optional[List[float]] = None,
    method: str = "add",
    **kwargs
) -> Expr:
    """合并多个因子
    
    Args:
        factors: 因子列表
        weights: 权重列表
        method: 合并方法 (add/wavg/rank/mul)
    
    Returns:
        合并后的表达式
    """
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
        total_weight = sum(weights)
        return weighted / total_weight
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


def chg_ids(
    f: Union[Expr, str],
    id_map: Dict[str, str],
    **kwargs
) -> Expr:
    """ID转换
    
    Args:
        f: 表达式或列名
        id_map: ID映射字典
    
    Returns:
        ID转换后的表达式
    """
    f = _ensure_expr(f)
    old_keys = list(id_map.keys())
    new_keys = list(id_map.values())
    return f.replace(old_keys, new_keys)


# ==============================================================================
# 自动注册：扫描模块中的所有函数并注册到注册表
# ==============================================================================

import sys

def _auto_register():
    """自动注册所有模块级函数到注册表"""
    current_module = sys.modules[__name__]
    
    category_map = {
        # Point 算子
        "abs": OperatorCategory.POINT, "log": OperatorCategory.POINT,
        "sign": OperatorCategory.POINT, "sqrt": OperatorCategory.POINT,
        "square": OperatorCategory.POINT, "pow": OperatorCategory.POINT,
        "clip": OperatorCategory.POINT, "ceil": OperatorCategory.POINT,
        "floor": OperatorCategory.POINT, "fix": OperatorCategory.POINT,
        "applymap": OperatorCategory.POINT, "nanargmax": OperatorCategory.POINT,
        "nanargmin": OperatorCategory.POINT, "nanmedian": OperatorCategory.POINT,
        "nanquantile": OperatorCategory.POINT, "nancount": OperatorCategory.POINT,
        "nanprod": OperatorCategory.POINT, "nan_to_null": OperatorCategory.POINT,
        "astype": OperatorCategory.POINT, "replace": OperatorCategory.POINT,
        "fetch": OperatorCategory.POINT, "fill_null": OperatorCategory.POINT,
        "fill_zero": OperatorCategory.POINT, "isnull": OperatorCategory.POINT,
        "notnull": OperatorCategory.POINT,
        "nanmax": OperatorCategory.POINT, "nanmin": OperatorCategory.POINT,
        "nanmean": OperatorCategory.POINT, "nansum": OperatorCategory.POINT,
        "nanstd": OperatorCategory.POINT, "nanvar": OperatorCategory.POINT,
        "where": OperatorCategory.POINT, "fillna": OperatorCategory.POINT,
        
        # Time 算子
        "rolling_mean": OperatorCategory.TIME, "rolling_std": OperatorCategory.TIME,
        "rolling_max": OperatorCategory.TIME, "rolling_min": OperatorCategory.TIME,
        "rolling_sum": OperatorCategory.TIME, "rolling_median": OperatorCategory.TIME,
        "rolling_var": OperatorCategory.TIME, "rolling_prod": OperatorCategory.TIME,
        "rolling_skew": OperatorCategory.TIME, "rolling_kurt": OperatorCategory.TIME,
        "rolling_count": OperatorCategory.TIME, "rolling_argmax": OperatorCategory.TIME,
        "rolling_argmin": OperatorCategory.TIME,
        "rolling_corr": OperatorCategory.TIME, "rolling_cov": OperatorCategory.TIME,
        "rolling_quantile": OperatorCategory.TIME, "rolling_rank": OperatorCategory.TIME,
        "ts_mean": OperatorCategory.TIME, "ts_std": OperatorCategory.TIME,
        "ts_max": OperatorCategory.TIME, "ts_min": OperatorCategory.TIME,
        "ts_sum": OperatorCategory.TIME, "ts_median": OperatorCategory.TIME,
        "ts_corr": OperatorCategory.TIME, "ts_cov": OperatorCategory.TIME,
        "ts_rank": OperatorCategory.TIME, "ts_argmax": OperatorCategory.TIME,
        "ts_argmin": OperatorCategory.TIME, "ts_delta": OperatorCategory.TIME,
        "ts_pct_change": OperatorCategory.TIME, "ts_lag": OperatorCategory.TIME,
        "ts_lead": OperatorCategory.TIME,
        "correlation": OperatorCategory.TIME, "covariance": OperatorCategory.TIME,
        "delta": OperatorCategory.TIME, "pct_change": OperatorCategory.TIME,
        "diff": OperatorCategory.TIME, "lag": OperatorCategory.TIME,
        "delay": OperatorCategory.TIME, "ref": OperatorCategory.TIME,
        "shift": OperatorCategory.TIME,
        "expanding_mean": OperatorCategory.TIME, "expanding_std": OperatorCategory.TIME,
        "expanding_sum": OperatorCategory.TIME,
        "expanding_max": OperatorCategory.TIME, "expanding_min": OperatorCategory.TIME,
        "expanding_median": OperatorCategory.TIME, "expanding_count": OperatorCategory.TIME,
        "expanding_var": OperatorCategory.TIME, "expanding_kurt": OperatorCategory.TIME,
        "expanding_skew": OperatorCategory.TIME, "expanding_quantile": OperatorCategory.TIME,
        "expanding_corr": OperatorCategory.TIME, "expanding_cov": OperatorCategory.TIME,
        "ewm_mean": OperatorCategory.TIME, "ewm_std": OperatorCategory.TIME,
        "ewm_var": OperatorCategory.TIME, "ewm_corr": OperatorCategory.TIME,
        "ewm_cov": OperatorCategory.TIME,
        
        # Section 算子
        "standardizeZScore": OperatorCategory.SECTION, "zscore": OperatorCategory.SECTION,
        "rank": OperatorCategory.SECTION, "winsorize": OperatorCategory.SECTION,
        "neutralize": OperatorCategory.SECTION, "neutralize_market": OperatorCategory.SECTION,
        "scale": OperatorCategory.SECTION, "orthogonalize": OperatorCategory.SECTION,
        "fillNaNByFun": OperatorCategory.SECTION, "fillNaNByRegress": OperatorCategory.SECTION,
        "ic": OperatorCategory.SECTION, "rank_ic": OperatorCategory.SECTION,
        "group_norm": OperatorCategory.SECTION, "group_winsorize": OperatorCategory.SECTION,
        "standardizeRank": OperatorCategory.SECTION, "weightStandardize": OperatorCategory.SECTION,
        
        # Multi-Section 算子
        "aggregate": OperatorCategory.MULTI_SECTION,
        "disaggregate": OperatorCategory.MULTI_SECTION,
        "aggr_sum": OperatorCategory.MULTI_SECTION,
        "aggr_prod": OperatorCategory.MULTI_SECTION,
        "aggr_max": OperatorCategory.MULTI_SECTION,
        "aggr_min": OperatorCategory.MULTI_SECTION,
        "aggr_mean": OperatorCategory.MULTI_SECTION,
        "aggr_std": OperatorCategory.MULTI_SECTION,
        "aggr_var": OperatorCategory.MULTI_SECTION,
        "aggr_median": OperatorCategory.MULTI_SECTION,
        "aggr_quantile": OperatorCategory.MULTI_SECTION,
        "aggr_count": OperatorCategory.MULTI_SECTION,
        "merge": OperatorCategory.MULTI_SECTION,
        "chg_ids": OperatorCategory.MULTI_SECTION,
        
        # 组合算子
        "add": OperatorCategory.POINT, "sub": OperatorCategory.POINT,
        "mul": OperatorCategory.POINT, "div": OperatorCategory.POINT,
        "weighted_sum": OperatorCategory.POINT, "combine": OperatorCategory.POINT,
        "if_then_else": OperatorCategory.POINT,
        "regress": OperatorCategory.TIME, "zscored": OperatorCategory.TIME,
        "decay_linear": OperatorCategory.TIME, "decay_exp": OperatorCategory.TIME,
        "vwap": OperatorCategory.TIME,
    }
    
    for name, category in category_map.items():
        func = getattr(current_module, name, None)
        if func is not None and callable(func) and name not in _OPERATOR_REGISTRY[category]:
            sig = inspect.signature(func)
            _OPERATOR_REGISTRY[category][name] = {
                "name": name,
                "category": category,
                "func": func,
                "doc": inspect.getdoc(func) or "",
                "signature": str(sig),
                "parameters": list(sig.parameters.keys()),
            }

_auto_register()
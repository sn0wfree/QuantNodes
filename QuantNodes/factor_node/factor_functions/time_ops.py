# coding=utf-8
"""
时间序列算子

本模块包含所有时间序列相关的因子运算算子。
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
from polars import Expr

from QuantNodes.factor_node.factor_functions._helpers import (
    OperatorCategory,
    register_operator,
    _ensure_expr,
    _expanding_var_expr,
    _apply_weights,
    _cumulative_map_batches_single,
    _cumulative_map_batches_dual,
    _cum_single_quantile,
    _make_rolling_ts_wrapper,
    _make_expanding_wrapper,
    _make_dual_rolling_wrapper,
    _make_diff_wrapper,
    _make_alias,
)

# ==============================================================================
# 滚动窗口算子 - 工厂生成
# ==============================================================================

_ROLLING_TS_DOCS = {
    "rolling_mean": "滚动窗口均值\n\n    Args:\n        f: 表达式或列名\n        window: 窗口大小\n        min_periods: 最小观测数",
    "rolling_max": "滚动窗口最大值",
    "rolling_min": "滚动窗口最小值",
    "rolling_sum": "滚动窗口求和",
    "rolling_median": "滚动窗口中位数",
}

_ROLLING_TS_METHODS = {
    "rolling_mean": "rolling_mean",
    "rolling_max": "rolling_max",
    "rolling_min": "rolling_min",
    "rolling_sum": "rolling_sum",
    "rolling_median": "rolling_median",
}

for _name, _method in _ROLLING_TS_METHODS.items():
    _make_rolling_ts_wrapper(_name, _method, _ROLLING_TS_DOCS[_name])

del _name, _method, _ROLLING_TS_METHODS, _ROLLING_TS_DOCS


# ==============================================================================
# 滚动窗口算子 - 显式定义
# ==============================================================================

@register_operator(OperatorCategory.TIME)
def rolling_std(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    **kwargs
) -> Expr:
    """滚动窗口标准差"""
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return e.rolling_std(window, min_samples=mp, ddof=ddof)


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
    """滚动窗口求积（log-sum-exp 方法）"""
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return e.log().rolling_sum(window, min_samples=mp).exp()


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
# Argmax/Argmin 算子
# ==============================================================================

def _rolling_arg_op(f: Union[Expr, str], window: int, op: str,
                    min_periods: Optional[int] = None) -> Expr:
    """滚动窗口最大值/最小值索引，用 map_batches + numpy 实现 O(n·window)"""
    e = _ensure_expr(f)
    window = min(window, 30)
    is_max = op == "max"

    def _arg_numpy(s: pl.Series) -> pl.Series:
        arr = s.to_numpy()
        n = len(arr)
        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            start = max(0, i - window + 1)
            idx = np.argmax(arr[start:i + 1]) if is_max else np.argmin(arr[start:i + 1])
            result[i] = idx
        return pl.Series(result)

    return e.map_batches(_arg_numpy, return_dtype=pl.Int32)


@register_operator(OperatorCategory.TIME)
def rolling_argmax(f: Union[Expr, str], window: int = 20,
                   min_periods: Optional[int] = None, **kwargs) -> Expr:
    """滚动窗口最大值索引"""
    return _rolling_arg_op(f, window, "max", min_periods)


@register_operator(OperatorCategory.TIME)
def rolling_argmin(f: Union[Expr, str], window: int = 20,
                   min_periods: Optional[int] = None, **kwargs) -> Expr:
    """滚动窗口最小值索引"""
    return _rolling_arg_op(f, window, "min", min_periods)


# ==============================================================================
# 双因子滚动相关/协方差
# ==============================================================================

_make_dual_rolling_wrapper(
    "rolling_corr", True,
    "滚动窗口相关系数（双因子）\n\n    Args:\n        f1: 第一个表达式\n        f2: 第二个表达式\n        window: 窗口大小\n        min_periods: 最小观测数"
)
_make_dual_rolling_wrapper(
    "rolling_cov", False,
    "滚动窗口协方差（双因子）\n\n    Args:\n        f1: 第一个表达式\n        f2: 第二个表达式\n        window: 窗口大小\n        min_periods: 最小观测数"
)
_make_dual_rolling_wrapper(
    "rolling_cov", "ts_cov",
    "滚动窗口协方差（双因子）\n\n    Args:\n        f1: 第一个表达式\n        f2: 第二个表达式\n        window: 窗口大小\n        min_periods: 最小观测数"
)


# ==============================================================================
# 别名
# ==============================================================================

_make_alias("ts_corr", rolling_corr,  # noqa: F821
            "滚动相关系数 (rolling_corr 别名)")
_make_alias("ts_cov", rolling_cov,  # noqa: F821
            "滚动协方差 (rolling_cov 别名)")

_make_alias("correlation", rolling_corr,  # noqa: F821
            "相关系数 (rolling_corr 别名)")
_make_alias("covariance", rolling_cov,  # noqa: F821
            "滚动协方差 (rolling_cov 别名)")

_make_alias("ts_mean", rolling_mean,  # noqa: F821
            "时间序列均值 (滚动均值别名)")
_make_alias("ts_std", rolling_std,
            "时间序列标准差")
_make_alias("ts_max", rolling_max,  # noqa: F821
            "时间序列最大值")
_make_alias("ts_min", rolling_min,  # noqa: F821
            "时间序列最小值")
_make_alias("ts_sum", rolling_sum,  # noqa: F821
            "时间序列求和")
_make_alias("ts_median", rolling_median,  # noqa: F821
            "时间序列中位数")


@register_operator(OperatorCategory.TIME)
def ts_rank(f: Union[Expr, str], window: int = 20,
            min_periods: Optional[int] = None, **kwargs) -> Expr:
    """滚动排名 (0-1 归一化)"""
    e = _ensure_expr(f)
    return e.rolling_rank(window)


_make_alias("ts_argmax", rolling_argmax,
            "滚动最大值的位置")
_make_alias("ts_argmin", rolling_argmin,
            "滚动最小值的位置")


# ==============================================================================
# 差分与变化
# ==============================================================================

_make_diff_wrapper("ts_delta", "diff", "差分")
_make_diff_wrapper("ts_pct_change", "pct_change", "百分比变化")
_make_diff_wrapper("diff", "diff", "差分")
_make_diff_wrapper("lag", "shift", "滞后算子")

_make_alias("delta", ts_delta, "差分 (ts_delta 别名)")  # noqa: F821
_make_alias("pct_change", ts_pct_change, "百分比变化 (ts_pct_change 别名)")  # noqa: F821


@register_operator(OperatorCategory.TIME)
def ts_lag(f: Union[Expr, str], periods: int = 1, **kwargs) -> Expr:
    """滞后 (向后移动)"""
    e = _ensure_expr(f)
    return e.shift(periods)


@register_operator(OperatorCategory.TIME)
def ts_lead(f: Union[Expr, str], periods: int = 1, **kwargs) -> Expr:
    """前向移动 (向前移动)"""
    e = _ensure_expr(f)
    return e.shift(-periods)


_make_alias("delay", ts_lag, "滞后 (ts_lag 别名)")
_make_alias("ref", delay, "引用历史值 (delay 别名)")  # noqa: F821
_make_alias("shift", ts_lag, "移动 (shift 别名)")
_make_alias("ts_shift", ts_lag, "移动 (ts_lag 别名)")
_make_alias("ts_prod", rolling_prod, "滚动求积 (rolling_prod 别名)")


# ==============================================================================
# 扩展窗口算子
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
# 指数加权算子
# ==============================================================================

@register_operator(OperatorCategory.TIME)
def ewm_mean(f: Union[Expr, str], alpha: float = 0.5,
             adjust: bool = True, **kwargs) -> Expr:
    """指数加权移动平均"""
    e = _ensure_expr(f)
    return e.ewm_mean(alpha=alpha, adjust=adjust)


@register_operator(OperatorCategory.TIME)
def ewm_std(f: Union[Expr, str], alpha: float = 0.5,
            adjust: bool = True, **kwargs) -> Expr:
    """指数加权移动标准差"""
    e = _ensure_expr(f)
    return e.ewm_std(alpha=alpha, adjust=adjust)


@register_operator(OperatorCategory.TIME)
def ewm_corr(f1: Union[Expr, str], f2: Union[Expr, str],
             alpha: float = 0.5, **kwargs) -> Expr:
    """指数加权相关系数"""
    e1 = _ensure_expr(f1)
    e2 = _ensure_expr(f2)
    mean1 = e1.ewm_mean(alpha=alpha)
    mean2 = e2.ewm_mean(alpha=alpha)
    var1 = e1.ewm_var(alpha=alpha)
    var2 = e2.ewm_var(alpha=alpha)
    cov = (e1 * e2).ewm_mean(alpha=alpha) - mean1 * mean2
    return cov / ((var1 * var2).sqrt() + 1e-10)


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
# 高级时间序列算子
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


@register_operator(OperatorCategory.TIME)
def rolling_change_rate(f: Union[Expr, str], window: int = 20, **kwargs) -> Expr:
    """滚动变化率（符号保持）"""
    f = _ensure_expr(f)
    numerator = f - f.shift(window)
    denominator = f.shift(window)
    rate = numerator / (denominator.abs() + 1e-8)
    same_sign = (numerator * denominator) >= 0
    return pl.when(same_sign).then(rate).when(numerator > 0).then(pl.lit(1.0)).when(numerator < 0).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))


# ==============================================================================
# QuantAlpha M1 新增别名（Alpha 101 算子命名约定）
# ==============================================================================
# 这些别名遵循 Alpha 101 论文的 `ts_*` / `*_xs` 命名约定，
# 使 Alpha 101 公式可以直接用 eval() 执行，详见
# QuantNodes/research/quant_alpha/operator_vocab/metadata.py

_make_alias(
    "ts_decay_linear",
    decay_linear,
    "线性衰减加权移动平均（Alpha 101 命名约定，等价于 decay_linear）",
    category=OperatorCategory.TIME,
)

_make_alias(
    "ts_skew",
    rolling_skew,
    "滚动偏度（Alpha 101 命名约定，等价于 rolling_skew）",
    category=OperatorCategory.TIME,
)

_make_alias(
    "ts_kurt",
    rolling_kurt,
    "滚动峰度（Alpha 101 命名约定，等价于 rolling_kurt）",
    category=OperatorCategory.TIME,
)

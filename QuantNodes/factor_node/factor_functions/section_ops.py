# coding=utf-8
"""
截面算子

本模块包含所有截面（cross-sectional）相关的因子运算算子。
"""

from __future__ import annotations

from typing import Any, Optional, Union

import polars as pl
from polars import Expr

from QuantNodes.factor_node.factor_functions._helpers import (
    OperatorCategory,
    register_operator,
    _ensure_expr,
    _make_alias,
    _make_nan_wrapper,
)


# ==============================================================================
# 截面算子
# ==============================================================================

@register_operator(OperatorCategory.SECTION)
def standardizeZScore(f: Union[Expr, str], eps: float = 1e-8, **kwargs) -> Expr:
    """Z-score 标准化"""
    e = _ensure_expr(f)
    mean = e.mean()
    std = e.std()
    return (e - mean) / (std + eps)


_make_alias("zscore", standardizeZScore, "Z-score 标准化 (standardizeZScore 别名)",
            category=OperatorCategory.SECTION)


@register_operator(OperatorCategory.SECTION)
def rank(f: Union[Expr, str], method: str = "dense", **kwargs) -> Expr:
    """截面排名 (归一化到 0-1)"""
    e = _ensure_expr(f)
    if method == "dense":
        return (e.rank() - 1) / (e.count() - 1)
    elif method == "ordinal":
        return e.rank()
    elif method == "min":
        r = e.rank()
        return (r - r.min()) / (r.max() - r.min())
    elif method == "average":
        return (e.rank() - 1) / (e.count() - 1)
    else:
        return e.rank() / e.count()


@register_operator(OperatorCategory.SECTION)
def winsorize(f: Union[Expr, str], lower: float = 0.01,
              upper: float = 0.01, method: str = "quantile", **kwargs) -> Expr:
    """去极值"""
    e = _ensure_expr(f)
    if method == "quantile":
        lower_bound = e.quantile(lower)
        upper_bound = e.quantile(1 - upper)
    else:
        mean = e.mean()
        std = e.std()
        lower_bound = mean - lower * std
        upper_bound = mean + upper * std
    return e.clip(lower_bound, upper_bound)


@register_operator(OperatorCategory.SECTION)
def neutralize(f: Union[Expr, str],
               group: Optional[Union[Expr, str]] = None, **kwargs) -> Expr:
    """行业中性的 (减去行业均值)"""
    e = _ensure_expr(f)
    if group:
        g = _ensure_expr(group)
        group_mean = e.mean().over(g)
        return e - group_mean
    return e - e.mean()


@register_operator(OperatorCategory.SECTION)
def neutralize_market(f: Union[Expr, str], **kwargs) -> Expr:
    """市场中性 (减去市场均值)"""
    e = _ensure_expr(f)
    return e - e.mean()


@register_operator(OperatorCategory.SECTION)
def scale(f: Union[Expr, str], method: str = "zscore", **kwargs) -> Expr:
    """归一化"""
    e = _ensure_expr(f)
    col_name = f if isinstance(f, str) else None

    if method == "zscore":
        result = standardizeZScore(e)
    else:
        e_min = e.min()
        e_max = e.max()
        diff = e_max - e_min
        result = pl.when(diff == 0).then(pl.lit(0.0)).otherwise((e - e_min) / diff)

    if col_name:
        return result.alias(col_name)
    return result


@register_operator(OperatorCategory.SECTION)
def ic(f: Union[Expr, str], target: Union[Expr, str], **kwargs) -> Expr:
    """IC (Pearson 相关系数)"""
    e = _ensure_expr(f)
    t = _ensure_expr(target)
    e_mean = e.mean()
    t_mean = t.mean()
    e_centered = e - e_mean
    t_centered = t - t_mean
    cov = (e_centered * t_centered).mean()
    e_std = e_centered.std(ddof=0)
    t_std = t_centered.std(ddof=0)
    return cov / (e_std * t_std + 1e-8)


@register_operator(OperatorCategory.SECTION)
def rank_ic(f: Union[Expr, str], target: Union[Expr, str], **kwargs) -> Expr:
    """Rank IC (Spearman 相关系数)"""
    e = _ensure_expr(f)
    t = _ensure_expr(target)
    e_rank = e.rank()
    t_rank = t.rank()
    e_mean = e_rank.mean()
    t_mean = t_rank.mean()
    e_centered = e_rank - e_mean
    t_centered = t_rank - t_mean
    cov = (e_centered * t_centered).mean()
    e_std = e_centered.std(ddof=0)
    t_std = t_centered.std(ddof=0)
    return cov / (e_std * t_std + 1e-8)


@register_operator(OperatorCategory.SECTION)
def group_norm(f: Union[Expr, str], group: Union[Expr, str],
               method: str = "zscore", **kwargs) -> Expr:
    """分组标准化"""
    e = _ensure_expr(f)
    g = _ensure_expr(group)
    if method == "zscore":
        group_mean = e.mean().over(g)
        group_std = e.std().over(g)
        return (e - group_mean) / (group_std + 1e-8)
    else:
        group_min = e.min().over(g)
        group_max = e.max().over(g)
        return (e - group_min) / (group_max - group_min + 1e-8)


@register_operator(OperatorCategory.SECTION)
def group_winsorize(f: Union[Expr, str], group: Union[Expr, str],
                     lower: float = 0.01, upper: float = 0.01, **kwargs) -> Expr:
    """分组去极值"""
    e = _ensure_expr(f)
    g = _ensure_expr(group)
    lower_bound = e.quantile(lower).over(g)
    upper_bound = e.quantile(1 - upper).over(g)
    return e.clip(lower_bound, upper_bound)


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
    """NaN 填充"""
    return _ensure_expr(f).fill_nan(value)


@register_operator(OperatorCategory.SECTION)
def fillNaNByRegress(f: Union[Expr, str], reference: Union[Expr, str], **kwargs) -> Expr:
    """NaN 填充：用 reference 回归拟合"""
    f = _ensure_expr(f)
    ref = _ensure_expr(reference)
    mask = f.is_not_null()
    f_clean = f.filter(mask)
    ref_clean = ref.filter(mask)
    beta = (f_clean * ref_clean).mean() / (ref_clean ** 2).mean()
    return pl.when(mask).then(f).otherwise(beta * ref)


_make_nan_wrapper("nanmax", "max", "截面最大值（忽略 NaN）")
_make_nan_wrapper("nanmin", "min", "截面最小值（忽略 NaN）")
_make_nan_wrapper("nanmean", "mean", "截面均值（忽略 NaN）")
_make_nan_wrapper("nansum", "sum", "截面求和（忽略 NaN）")
_make_nan_wrapper("nanstd", "std", "截面标准差（忽略 NaN）")
_make_nan_wrapper("nanvar", "var", "截面方差（忽略 NaN）")


@register_operator(OperatorCategory.SECTION)
def mad(f: Union[Expr, str], **kwargs) -> Expr:
    """Median Absolute Deviation (中位绝对偏差)"""
    f = _ensure_expr(f)
    median = f.median()
    return (f - median).abs().median() * 1.4826


@register_operator(OperatorCategory.SECTION)
def cross_sectional_rank(f: Union[Expr, str], **kwargs) -> Expr:
    """截面排名"""
    return rank(f, **kwargs)


@register_operator(OperatorCategory.SECTION)
def cross_sectional_zscore(f: Union[Expr, str], **kwargs) -> Expr:
    """截面 Z-score"""
    return standardizeZScore(f, **kwargs)


@register_operator(OperatorCategory.SECTION)
def cross_sectional_mean(f: Union[Expr, str], **kwargs) -> Expr:
    """截面均值"""
    return _ensure_expr(f).mean()


@register_operator(OperatorCategory.SECTION)
def cross_sectional_std(f: Union[Expr, str], **kwargs) -> Expr:
    """截面标准差"""
    return _ensure_expr(f).std()


@register_operator(OperatorCategory.SECTION)
def cross_sectional_sum(f: Union[Expr, str], **kwargs) -> Expr:
    """截面求和"""
    return _ensure_expr(f).sum()


@register_operator(OperatorCategory.SECTION)
def standardizeRank(f: Union[Expr, str], **kwargs) -> Expr:
    """排名标准化 (0-1 归一化)"""
    return rank(f, method="dense", **kwargs)


@register_operator(OperatorCategory.SECTION)
def weightStandardize(f: Union[Expr, str], weight: Union[Expr, str] = None,
                      **kwargs) -> Expr:
    """加权标准化"""
    e = _ensure_expr(f)
    if weight:
        w = _ensure_expr(weight)
        w_sum = w.sum()
        w_norm = w / w_sum
        weighted_mean = (e * w_norm).sum()
        weighted_std = (((e - weighted_mean) ** 2) * w_norm).sum().sqrt()
        return (e - weighted_mean) / (weighted_std + 1e-8)
    else:
        return standardizeZScore(e)


# ==============================================================================
# QuantAlpha M1 新增别名（Alpha 101 IndNeutralize 命名约定）
# ==============================================================================
# IndNeutralize 是 Alpha 101 的命名约定。
# 直接用 polars 的 over() 实现，避免与 composite_dag_ops 的循环导入。


@register_operator(OperatorCategory.SECTION, "IndNeutralize")
def IndNeutralize(
    f: Union[Expr, str],
    ind_class: str = "citic_1",
    **kwargs,
) -> Expr:
    """行业中性化（Alpha 101 命名约定）

    IndNeutralize(x, ind_class) = x - mean(x) grouped by ind_class

    Args:
        f: 输入表达式
        ind_class: 行业分类列名（默认 'citic_1' 一级行业）
    """
    e = _ensure_expr(f)
    return e - e.mean().over(ind_class)

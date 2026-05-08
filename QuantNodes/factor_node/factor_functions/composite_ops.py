# coding=utf-8
"""
组合算子

本模块包含所有组合（multi-section）相关的因子运算算子。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from polars import Expr

from QuantNodes.factor_node.factor_functions._helpers import (
    OperatorCategory,
    register_operator,
    _ensure_expr,
    _make_aggr_wrapper,
)


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


@register_operator(OperatorCategory.MULTI_SECTION)
def blend(f1: Union[Expr, str], f2: Union[Expr, str],
          alpha: float = 0.5, **kwargs) -> Expr:
    """混合两个因子"""
    return _ensure_expr(f1) * alpha + _ensure_expr(f2) * (1 - alpha)


@register_operator(OperatorCategory.MULTI_SECTION)
def nav(f: Union[Expr, str], **kwargs) -> Expr:
    """NAV (单位净值)"""
    return (1 + _ensure_expr(f)).cum_prod()


@register_operator(OperatorCategory.MULTI_SECTION)
def rebase(f: Union[Expr, str], base: float = 100.0, **kwargs) -> Expr:
    """重定基期"""
    return (_ensure_expr(f) / _ensure_expr(f).first() * base)

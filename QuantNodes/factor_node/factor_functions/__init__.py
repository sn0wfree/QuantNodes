# coding=utf-8
"""
factor_functions - 因子函数实现层

本模块包含所有算子的核心实现，按类别拆分为子模块：
- time_ops: 时间序列算子
- section_ops: 截面算子
- math_ops: 数学算子
- composite_ops: 组合算子
- talib_ops: TA-Lib 技术指标

注册表 API：
- list_operators()
- get_operator()
- operator_info()
- generate_documentation()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

# 导入辅助函数和注册表
from QuantNodes.factor_node.factor_functions._helpers import (
    _OPERATOR_REGISTRY,
    OperatorCategory,
    register_operator,
    _ensure_expr,
)

# 导入子模块（触发算子注册）
from QuantNodes.factor_node.factor_functions import time_ops
from QuantNodes.factor_node.factor_functions import section_ops
from QuantNodes.factor_node.factor_functions import math_ops
from QuantNodes.factor_node.factor_functions import composite_ops

# 导入常用函数供直接使用
from QuantNodes.factor_node.factor_functions.math_ops import (
    ceil, floor, fix, applymap,
    nanargmax, nanargmin, nanmedian, nanquantile, nancount, nanprod,
    astype, replace, fetch,
    abs as ff_abs, log as ff_log, sign, sqrt as ff_sqrt, square, clip,
    isnull, notnull, fill_null, fill_zero, nan_to_null,
    pow as ff_pow,
    nanmax, nanmin, nanmean, nansum, nanstd, nanvar,
    where, fillna,
    add, sub, mul, div,
    log1p, if_then_else, market_cap,
    weighted_sum, combine,
    book_to_market, earnings_to_market,
)

from QuantNodes.factor_node.factor_functions.time_ops import (
    rolling_mean, rolling_std, rolling_max, rolling_min, rolling_sum,
    rolling_median, rolling_var,
    rolling_prod, rolling_skew, rolling_kurt, rolling_count,
    rolling_argmax, rolling_argmin,
    rolling_corr, rolling_cov, rolling_quantile, rolling_rank,
    ewm_var, ewm_mean, ewm_std, ewm_corr, ewm_cov,
    expanding_mean, expanding_std, expanding_sum,
    expanding_max, expanding_min, expanding_median, expanding_count,
    expanding_var, expanding_kurt, expanding_skew, expanding_quantile,
    expanding_corr, expanding_cov,
    ts_mean, ts_std, ts_corr, ts_cov, ts_rank, ts_delta, ts_lag,
    ts_argmax, ts_argmin, ts_lead, ts_pct_change,
    decay_linear, decay_exp, vwap, rolling_change_rate,
    regress, zscored, ts_shift, diff, lag, delay, ref, shift,
    delta, pct_change,
    correlation, covariance,
    ts_prod,
)

from QuantNodes.factor_node.factor_functions.section_ops import (
    rank, zscore, winsorize, neutralize, neutralize_market, scale,
    ic, rank_ic, group_norm, group_winsorize,
    standardizeZScore, orthogonalize, mad,
    fillNaNByFun, fillNaNByRegress,
    cross_sectional_rank, cross_sectional_zscore,
    cross_sectional_mean, cross_sectional_std, cross_sectional_sum,
    standardizeRank, weightStandardize,
)

from QuantNodes.factor_node.factor_functions.composite_ops import (
    aggregate, disaggregate,
    aggr_sum, aggr_mean, aggr_max, aggr_min, aggr_std, aggr_var,
    aggr_median, aggr_count, aggr_prod, aggr_quantile,
    merge, chg_ids, blend, nav, rebase,
)

# 为测试兼容性导出 (shadow builtins)
abs = ff_abs
log = ff_log
pow = ff_pow
sqrt = ff_sqrt


# ==============================================================================
# 注册表查询 API
# ==============================================================================

def list_operators(category: Optional[str] = None, include_custom: bool = True) -> List[str]:
    """列出所有算子名称

    Args:
        category: 可选，限定分类
        include_custom: 是否包含自定义算子（默认 True）
    """
    if include_custom:
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        custom = _CustomOperatorRegistry.list(category)
        builtin = list(_OPERATOR_REGISTRY.get(category, {}).keys()) if category else [
            name for cat in _OPERATOR_REGISTRY for name in _OPERATOR_REGISTRY[cat]
        ]
        seen = set()
        result = []
        for name in custom + builtin:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    if category:
        return list(_OPERATOR_REGISTRY.get(category, {}).keys())
    return [name for cat in _OPERATOR_REGISTRY for name in _OPERATOR_REGISTRY[cat]]


def get_operator(name: str, category: Optional[str] = None) -> Optional[Callable]:
    """根据名称获取算子函数（级联查询：先自定义注册表，再内置注册表）"""
    from QuantNodes.operators.registry import _CustomOperatorRegistry

    func = _CustomOperatorRegistry.get(name, category)
    if func is not None:
        return func

    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op["func"] if op else None

    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]["func"]
    return None


def operator_info(name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """获取算子详细信息（级联查询：先自定义注册表，再内置注册表）"""
    from QuantNodes.operators.registry import _CustomOperatorRegistry

    info = _CustomOperatorRegistry.info(name)
    if info is not None:
        return info

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


__all__ = [
    "list_operators",
    "get_operator",
    "operator_info",
    "generate_documentation",
    "register_operator",
    "OperatorCategory",
    "_OPERATOR_REGISTRY",
    "_ensure_expr",
    # math_ops
    "ceil", "floor", "fix", "applymap",
    "nanargmax", "nanargmin", "nanmedian", "nanquantile", "nancount", "nanprod",
    "astype", "replace", "fetch",
    "ff_abs", "ff_log", "sign", "ff_sqrt", "square", "clip",
    "isnull", "notnull", "fill_null", "fill_zero", "nan_to_null",
    "ff_pow",
    "nanmax", "nanmin", "nanmean", "nansum", "nanstd", "nanvar",
    "where", "fillna",
    "add", "sub", "mul", "div",
    "log1p", "if_then_else", "market_cap",
    "weighted_sum", "combine", "book_to_market", "earnings_to_market",
    # time_ops
    "rolling_mean", "rolling_std", "rolling_max", "rolling_min", "rolling_sum",
    "rolling_median", "rolling_var",
    "rolling_prod", "rolling_skew", "rolling_kurt", "rolling_count",
    "rolling_argmax", "rolling_argmin",
    "rolling_corr", "rolling_cov", "rolling_quantile", "rolling_rank",
    "ewm_var", "ewm_mean", "ewm_std", "ewm_corr", "ewm_cov",
    "expanding_mean", "expanding_std", "expanding_sum",
    "expanding_max", "expanding_min", "expanding_median", "expanding_count",
    "expanding_var", "expanding_kurt", "expanding_skew", "expanding_quantile",
    "expanding_corr", "expanding_cov",
    "ts_mean", "ts_std", "ts_corr", "ts_cov", "ts_rank", "ts_delta", "ts_lag",
    "ts_argmax", "ts_argmin", "ts_lead", "ts_pct_change",
    "decay_linear", "decay_exp", "vwap", "rolling_change_rate",
    "regress", "zscored", "ts_shift", "diff", "lag", "delay", "ref", "shift",
    "delta", "pct_change", "correlation", "covariance", "ts_prod",
    # section_ops
    "rank", "zscore", "winsorize", "neutralize", "neutralize_market", "scale",
    "ic", "rank_ic", "group_norm", "group_winsorize",
    "standardizeZScore", "orthogonalize", "mad",
    "fillNaNByFun", "fillNaNByRegress",
    "cross_sectional_rank", "cross_sectional_zscore",
    "cross_sectional_mean", "cross_sectional_std", "cross_sectional_sum",
    "standardizeRank", "weightStandardize",
    # composite_ops
    "aggregate", "disaggregate",
    "aggr_sum", "aggr_mean", "aggr_max", "aggr_min", "aggr_std", "aggr_var",
    "aggr_median", "aggr_count", "aggr_prod", "aggr_quantile",
    "merge", "chg_ids", "blend", "nav", "rebase",
    # submodules for side-effect registration
    "time_ops",
    "section_ops",
    "math_ops",
    "composite_ops",
]

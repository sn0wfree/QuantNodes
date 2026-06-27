# coding=utf-8
"""
sources.py - Logic Mining 数据源适配

支持从 alpha101_design / alpha158_design / Alpha191 等数据源
提取公式用于逻辑抽取。

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.sources import (
        get_formulas_from_source, SOURCES,
    )

    formulas = get_formulas_from_source("alpha101", max_count=20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["SOURCES", "get_formulas_from_source", "list_available_sources"]


# 预定义的 Alpha101 代表性公式（量价类）
ALPHA101_FORMULAS: Dict[str, str] = {
    "alpha001": "rank(ts_argmax(signedpower(where(close < delay(close, 1), 1, -1) * (close - ts_min(close, 5)), 2), 5)) - 0.5",
    "alpha006": "-ts_corr(open, volume, 10)",
    "alpha012": "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    "alpha018": "-rank(rank(std(abs(close - open), 5) + (close - open) + rank(corr(close, ts_mean(volume, 20), 5))))",
    "alpha033": "rank(-1 + open / close)",
    "alpha038": "-rank(ts_mean(close, 10) / ts_mean(close, 20) * rank(volume))",
    "alpha041": "power(high * low, 0.5) - ts_mean(power(high * low, 0.5), 3)",
    "alpha054": "-1 * rank((low - close) * power(volume, 0.5) / power(ts_mean(volume, 20), 0.5))",
    "alpha055": "-1 * corr(rank(sub((high + low) / 2, ts_mean(high, 20))), rank(volume), 10)",
    "alpha066": "-1 * ts_corr(close, ts_mean(volume, 20), 5)",
    "alpha078": "rank(ts_corr(ts_mean(ts_mean(volume, 30), 37), ts_mean(close, 20), 7))",
    "alpha085": "rank(ts_corr(high, volume, 5))",
    "alpha088": "rank(ts_argmax(close - delay(close, 1), 30))",
    "alpha095": "std(volume, 20) / std(close, 20)",
    "alpha101": "rank(ts_mean(delta(close, 1), 5) - delta(ts_mean(close, 20), 5) / ts_mean(close, 20))",
}

# 预定义的 Alpha158 模板类别（来自 alpha158_design）
ALPHA158_TEMPLATE_CATEGORIES: List[Dict[str, str]] = [
    {"id": "KBAR", "name": "K线形态", "example": "rank(ts_mean(close - open, 20))"},
    {"id": "PRICE", "name": "价格时序", "example": "ts_mean(close / delay(close, 5) - 1, 20)"},
    {"id": "VOLUME", "name": "成交量时序", "example": "rank(volume / ts_mean(volume, 20))"},
    {"id": "ROLLING", "name": "滚动统计", "example": "rank(ts_std(returns, 20))"},
    {"id": "MOMENTUM", "name": "动量", "example": "rank(close / delay(close, 20) - 1)"},
    {"id": "REVERSAL", "name": "反转", "example": "-rank(close / delay(close, 5) - 1)"},
    {"id": "VOLATILITY", "name": "波动率", "example": "rank(ts_std(close, 20) / ts_mean(close, 20))"},
    {"id": "LIQUIDITY", "name": "流动性", "example": "rank(volume / amount)"},
]

# 数据源注册表
SOURCES: Dict[str, Dict[str, Any]] = {
    "alpha101": {
        "name": "WorldQuant Alpha101",
        "description": "WorldQuant 101 formulaic alphas",
        "formulas": ALPHA101_FORMULAS,
        "count": len(ALPHA101_FORMULAS),
    },
    "alpha158": {
        "name": "Qlib Alpha158",
        "description": "Qlib 158 standard features (templates)",
        "templates": ALPHA158_TEMPLATE_CATEGORIES,
        "count": len(ALPHA158_TEMPLATE_CATEGORIES),
    },
}


def get_formulas_from_source(
    source_lib: str,
    max_count: int = 20,
    only_volume_price: bool = True,
) -> List[Dict[str, str]]:
    """从指定数据源获取公式

    Args:
        source_lib: 来源库名称 ("alpha101" / "alpha158" / "alpha191")
        max_count: 最大数量
        only_volume_price: 仅返回量价类公式（过滤掉含财务/基本面数据的）

    Returns:
        List of {"id": ..., "formula": ...}
    """
    if source_lib not in SOURCES:
        logger.warning("Unknown source: %s, available: %s", source_lib, list(SOURCES.keys()))
        return []

    source = SOURCES[source_lib]
    results = []

    if source_lib == "alpha101":
        for fid, formula in source["formulas"].items():
            if only_volume_price and not _is_volume_price(formula):
                continue
            results.append({"id": fid, "formula": formula, "lib": source_lib})
            if len(results) >= max_count:
                break

    elif source_lib == "alpha158":
        for tmpl in source["templates"]:
            if only_volume_price and not _is_volume_price(tmpl["example"]):
                continue
            results.append({
                "id": tmpl["id"],
                "formula": tmpl["example"],
                "lib": source_lib,
                "name": tmpl["name"],
            })
            if len(results) >= max_count:
                break

    elif source_lib == "alpha191":
        # Alpha191 占位（PR-6 实现）
        logger.info("alpha191 not yet implemented, returning empty list")
        return []

    logger.info("Loaded %d formulas from %s", len(results), source_lib)
    return results


def _is_volume_price(formula: str) -> bool:
    """简单判断是否为量价类公式（排除财务/基本面类）"""
    formula_lower = formula.lower()
    # 排除包含财务指标的关键词
    exclude_keywords = [
        "earnings", "revenue", "profit", "bv", "market_cap",
        "pb", "pe", "ps", "dividend", "roe", "roa",
        "营业收入", "净利润", "市值", "市盈率",
    ]
    for kw in exclude_keywords:
        if kw in formula_lower:
            return False
    return True


def list_available_sources() -> List[str]:
    """列出所有可用的数据源"""
    return list(SOURCES.keys())


def get_source_info(source_lib: str) -> Optional[Dict[str, Any]]:
    """获取数据源详细信息"""
    return SOURCES.get(source_lib)
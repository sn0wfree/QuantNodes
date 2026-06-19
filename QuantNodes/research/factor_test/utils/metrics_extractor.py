# coding: utf-8
"""指标提取工具 / Metrics Extractor.

从单次回测的 ``ctx`` (12 节点输出 dict) 提取关键评估指标,
供 ``TrajectoryEntry.metrics`` / Evolution rank 使用.

Phase R2 (2026-06-19): 从 pipeline_runner.py 末尾抽出, 单一职责.
"""

from __future__ import annotations


def extract_metrics_from_ctx(ctx: dict) -> dict:
    """从单次回测 ctx 提取关键指标 (IC, Rank IC, ICIR, Sharpe, ARR, MDD, Calmar).

    Args:
        ctx: ``PipelineRunner.run()`` 的返回值, 通常含
            ``ICAnalyzer`` / ``LongShort`` 两个键.

    Returns:
        ``{ic_mean, rank_ic_mean, ic_ir, sharpe, arr, mdd, calmar}`` 的子集
        (缺失字段不会出现在结果里).
    """
    metrics: dict = {}
    ic = ctx.get("ICAnalyzer") or {}
    ic_result = ic.get("ic_result") if isinstance(ic, dict) else None
    if isinstance(ic_result, dict):
        for src_key, dst_key in (
            ("IC均值", "ic_mean"),
            ("Rank IC均值", "rank_ic_mean"),
            ("ICIR", "ic_ir"),
        ):
            if src_key in ic_result and ic_result[src_key] is not None:
                try:
                    metrics[dst_key] = float(ic_result[src_key])
                except (TypeError, ValueError):
                    pass
    ls = ctx.get("LongShort") or {}
    if isinstance(ls, dict):
        for src_key, dst_key in (
            ("sharpe", "sharpe"),
            ("annualized_return", "arr"),
            ("max_drawdown", "mdd"),
            ("calmar", "calmar"),
        ):
            if src_key in ls and ls[src_key] is not None:
                try:
                    metrics[dst_key] = float(ls[src_key])
                except (TypeError, ValueError):
                    pass
    return metrics


# 向后兼容别名 (老代码或测试可能 import 私有名)
_extract_metrics_from_ctx = extract_metrics_from_ctx

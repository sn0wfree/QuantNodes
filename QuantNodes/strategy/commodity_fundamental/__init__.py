# coding=utf-8
"""Commodity Fundamental 策略模块 (CMF-IC).

商品基本面量化策略族, 与 cta / momentum_etf_rotation 并列的独立策略来源.

核心思路: 卡尔曼滤波中枢 + 符号约束Lasso + 库存象限 + 估值缺口.

结构分层:
    共享信号层 (品种无关, 纯函数可复用)
        kalman_filter          §2.1 价格中枢卡尔曼滤波
        sign_constrained_lasso §2.2 符号约束Lasso + walk-forward
        inventory_quadrant     §3.1 库存四象限 (被动去库/主动补库/...)
        valuation_gap          §3.2 估值缺口 + 行业景气度
        crowding_simplified    拥挤度 (波动率分位)
        portfolio              指标计算与组合回测

    多品种管道
        pipeline / pipeline_v2 / pipeline_v3

    品种子策略 (每个品种一个子包, 后续新品种在此扩展)
        copper/               铜: v5主模型 + carry子策略 + 经济状态等
        (后续: 豆粕/原油/...)

数据: data/cmf/fundamental.duckdb
报告: reports/cmf/summary.md
"""
from __future__ import annotations

from .copper import (
    build_signal_v6,
    backtest_v6,
    apply_bull_enhance,
    build_signal_v5,
    backtest_v5,
    build_signal_carry,
    backtest_carry,
)

__all__ = [
    "build_signal_v6",
    "backtest_v6",
    "apply_bull_enhance",
    "build_signal_v5",
    "backtest_v5",
    "build_signal_carry",
    "backtest_carry",
]

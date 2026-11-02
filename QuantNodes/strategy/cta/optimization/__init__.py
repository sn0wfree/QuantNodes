# coding=utf-8
"""参数标定与自适应选择子模块.

功能:
    1. param_grid: 参数网格定义 (每状态的候选参数)
    2. evaluator: 多目标评估 (Calmar, MaxDD约束, 覆盖率约束)
    3. walk_forward: Walk-forward 优化引擎 (2年训练+1年验证)
    4. stability: 参数稳定性分析 (60%+50%)
    5. adaptive: 运行时 regime-conditional 参数选择

使用方式:
    # 离线标定
    from QuantNodes.strategy.cta.optimization import run_calibration
    result = run_calibration("IF", prices, states, param_grid)

    # 运行时
    from QuantNodes.strategy.cta.optimization import select_params
    params = select_params("IF", "trend_up")
"""
from __future__ import annotations

from .adaptive import load_21_product_calibration, select_params, set_calibrated
from .evaluator import evaluate_strategy
from .param_grid import (
    EXIT_PARAMS,
    GLOBAL_PARAMS,
    PARAM_GRID,
    RANGE_PARAMS,
    TREND_PARAMS,
    combo_count,
    sample_grid,
)
from .stability import analyze_stability
from .walk_forward import run_calibration, walk_forward

__all__ = [
    # param_grid
    "PARAM_GRID",
    "TREND_PARAMS",
    "RANGE_PARAMS",
    "EXIT_PARAMS",
    "GLOBAL_PARAMS",
    "sample_grid",
    "combo_count",
    # evaluator
    "evaluate_strategy",
    # walk_forward
    "walk_forward",
    "run_calibration",
    # stability
    "analyze_stability",
    # adaptive
    "select_params",
    "set_calibrated",
    "load_21_product_calibration",
]

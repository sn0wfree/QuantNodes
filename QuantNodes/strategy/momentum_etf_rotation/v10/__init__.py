# coding=utf-8
"""v10 主体策略 — 4 策略 Vol-parity 组合 (生产版).

包含:
  - dual_momentum.py:        Antonacci GEM 模型 4 大类资产轮动
  - dynamic_weight_schemes.py:  4 方案动态权重 (A/D/F/G) + Static baseline
  - epo_momentum.py:        EPO 动量策略
  - rrg_rotation.py:        RRG 四象限轮动

历史:
  - 5 层架构实验版 (从 v10 分离) 已迁移到 v11 (2026-07-27)
  - 4 策略 Vol-parity 仍是生产首选 (OOS Calmar 1.43)
  - 2026-08-03: 精简方案, 删除 B/C/E (被 D/F/G 覆盖)
"""
from __future__ import annotations

# 4 策略主体
from .dynamic_weight_schemes import (
    load_navs,
    compute_nav,
    metrics,
    scheme_a_regime,
    scheme_d_signal_weighted,
    scheme_f_conservative,
    scheme_g_trend_adaptive,
    BASE_WEIGHTS,
    STRATS,
    main as dynamic_main,
)
from .dual_momentum import (
    main as dual_momentum_main,
    load_etf_daily,
    load_all_assets_daily,
    load_all_assets_weekly,
    dual_momentum_signal,
    compute_nav as dual_compute_nav,
    metrics as dual_metrics,
)
from .epo_momentum import main as epo_main
from .rrg_rotation import main as rrg_main

__all__ = [
    # 4 策略主体 (dynamic_weight_schemes)
    "load_navs", "compute_nav", "metrics",
    "scheme_a_regime", "scheme_d_signal_weighted",
    "scheme_f_conservative", "scheme_g_trend_adaptive",
    "BASE_WEIGHTS", "STRATS",
    "dynamic_main", "dual_momentum_main", "epo_main", "rrg_main",
    "load_etf_daily", "load_all_assets_daily", "load_all_assets_weekly",
    "dual_momentum_signal", "dual_compute_nav", "dual_metrics",
]

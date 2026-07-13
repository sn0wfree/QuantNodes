# coding=utf-8
"""V7 子包入口 — 暴露核心 API + 端到端入口.

公开 API:
    V7_3Config
    V7_3SubStrategy
    run_v7_3_backtest (端到端)
    v7_macro_baseline (锁定 baseline, 2026-07-13)
    RollingSymmetry
    BootstrapLassoMapping
    FactorRiskParityOptimizer
"""
from .bootstrap_lasso import BootstrapLassoMapping
from .data_loader import (
    FACTOR_COLS,
    INDEX_COLS,
    load_factor_returns,
    load_index_panel,
    load_index_prices,
    load_macro_factors,
)
from .factor_risk_parity import FactorRiskParityOptimizer
from .macro_substrategy_v7_3 import (
    V7_3Config,
    V7_3SubStrategy,
    run_v7_3_backtest,
)
from .symmetry import RollingSymmetry

__all__ = [
    "V7_3Config",
    "V7_3SubStrategy",
    "run_v7_3_backtest",
    "v7_macro_baseline",
    "RollingSymmetry",
    "BootstrapLassoMapping",
    "FactorRiskParityOptimizer",
    "load_macro_factors",
    "load_factor_returns",
    "load_index_panel",
    "load_index_prices",
    "FACTOR_COLS",
    "INDEX_COLS",
]


def v7_macro_baseline() -> V7_3Config:
    """v7 宏观子策略 baseline 锁定 (Stage 30.4 完整版, 2026-07-13).

    算法: Symmetry (Klein 2013) + Bootstrap-Lasso × 500 + 源 FactorRiskParity
    数据: 13 指数 (含 中债1-3年国债财富指数) + 9 宏观因子 (8 实际使用)
    调仓: 季度, 8 quarter 滚动窗口
    成本: 5bp 佣金 + 5bp 滑点

    业绩 (3 个 random_state [42, 7, 123] 平均, OOS 2023-至今):
        Ann 5.24%, Vol 6.74%, Sharpe 0.778, DD -8.45%, Calmar 0.620
    OOS 2022-至今:  Ann 3.37%, Vol 6.99%, Calmar 0.371
    全期 2010-2026: Ann 2.94%, Vol 7.20%, Calmar 0.145

    用途: 宏观子策略 baseline, 与 v6.2 (行业轮动) 配合用
    锁定目的: 未来 v7.x 变更 (新因子 / 新池 / 新算法) 必对照此 baseline,
              退化 > 5% 需更新 baseline 并加 migration note.
              详见 tests/.../v7/test_v7_macro_baseline.py
    """
    return V7_3Config(
        bootstrap_times=500,           # 收敛点 (敏感性分析确认)
        bootstrap_resample_min=78,    # 1.5 年
        bootstrap_resample_max=104,   # 2 年
        bootstrap_random_state=42,
        bootstrap_cache_alpha=True,   # 30x 加速
        quarter_window=8,             # 2 年回看
        max_weight=0.5,               # 源 cell 99
        sum_lower=0.9,                # 源 cell 94
        sum_upper=1.0,
        commission_bp=5.0,
        slippage_bp=5.0,
    )

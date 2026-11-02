# coding=utf-8
"""参数网格定义.

每种市场状态对应一组可调参数:
    - TREND_PARAMS: 趋势状态专用 (confirm_window, ema_slow, use_ema_slope)
    - RANGE_PARAMS: 震荡状态专用 (rsi_window, rsi_oversold, rsi_overbought)
    - EXIT_PARAMS: 退出机制参数 (chandelier_mult, max_holding_days)
    - GLOBAL_PARAMS: 全局参数 (target_vol)

设计依据:
    华泰金工《时序CTA方法论综述》(2026-06-12) + 诊断发现
    IF 品种状态识别滞后 → confirm_window 是关键调优参数
    EMA方向不一致 → ema_slow + use_ema_slope 是关键调优参数
"""
from __future__ import annotations

import itertools
from typing import Any


# ── 参数网格 ──────────────────────────────────────────────────────

PARAM_GRID: dict[str, list] = {
    # 状态识别
    "confirm_window": [3, 5, 10, 15],
    # 趋势开仓
    "ema_slow": [30, 60, 120],
    "use_ema_slope": [True, False],
    # 震荡开仓
    "rsi_window": [10, 14, 20],
    "rsi_oversold": [25.0, 30.0, 35.0],
    "rsi_overbought": [65.0, 70.0, 75.0],
    # 退出
    "chandelier_mult": [2.0, 3.0, 4.0, 5.0],
    "max_holding_days": [30, 60, 120],
    # 全局
    "target_vol": [0.10, 0.15, 0.20],
    # ρ 条件 (阶梯式牛市误判 range: AU ρ>0 仅 2.8%)
    "rho_required": [True, False],
}

# ── 按状态分组 (全局闸门参数不在网格中, 保持固定: both_and, threshold=0.6) ──

TREND_PARAMS = ["confirm_window", "ema_slow", "use_ema_slope"]
RANGE_PARAMS = ["rsi_window", "rsi_oversold", "rsi_overbought"]
EXIT_PARAMS = ["chandelier_mult", "max_holding_days"]
GLOBAL_PARAMS = ["target_vol", "rho_required"]


def sample_grid(
    state_type: str = "all",
    n_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """从参数网格采样.

    Args:
        state_type: 'trend' / 'range' / 'exit' / 'global' / 'all'
        n_samples: 采样数量 (None = 全量笛卡尔积)
        seed: 随机种子 (n_samples 有效时)

    Returns:
        list[dict]: 每个元素是一组参数.
    """
    import random

    if state_type == "trend":
        keys = TREND_PARAMS
    elif state_type == "range":
        keys = RANGE_PARAMS
    elif state_type == "exit":
        keys = EXIT_PARAMS
    elif state_type == "global":
        keys = GLOBAL_PARAMS
    else:
        keys = list(PARAM_GRID.keys())

    grid = {k: PARAM_GRID[k] for k in keys}
    all_combos = list(itertools.product(*grid.values()))

    if n_samples is not None and n_samples < len(all_combos):
        rng = random.Random(seed)
        indices = rng.sample(range(len(all_combos)), n_samples)
        all_combos = [all_combos[i] for i in indices]

    return [dict(zip(grid.keys(), combo)) for combo in all_combos]


def combo_count(state_type: str = "all") -> int:
    """返回参数组合总数."""
    return len(sample_grid(state_type))

# coding=utf-8
"""参数稳定性分析.

判定标准 (宽松模式):
    - 参数在 > 60% 的窗口中出现为最优值
    - 对应验证期 Calmar > 0 的比例 > 50%

输出:
    每个参数的最优值、出现率、正收益率、稳定性标记
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .walk_forward import WindowResult


def analyze_stability(
    windows: list[WindowResult],
    param_grid: dict[str, list],
    min_appear_rate: float = 0.6,
    min_positive_rate: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """分析参数稳定性.

    Args:
        windows: walk-forward 各窗口结果
        param_grid: 参数网格
        min_appear_rate: 最低出现率 (默认 60%)
        min_positive_rate: 最低正收益率 (默认 50%)

    Returns:
        dict[param_name → {value, appear_rate, positive_rate, stable}]
    """
    if not windows:
        return {}

    results = {}
    for param_name in param_grid:
        values = [w.best_params.get(param_name) for w in windows]
        # 统计每个值的出现次数
        counts = Counter(values)
        if not counts:
            continue
        most_common_val, most_common_count = counts.most_common(1)[0]
        appear_rate = most_common_count / len(values)
        # 对应验证期正收益比例
        positive_windows = sum(
            1 for w, v in zip(windows, values)
            if v == most_common_val and w.val_calmar > 0
        )
        positive_rate = positive_windows / len(values) if len(values) > 0 else 0
        stable = appear_rate >= min_appear_rate and positive_rate >= min_positive_rate
        results[param_name] = {
            "value": most_common_val,
            "appear_rate": round(appear_rate, 3),
            "positive_rate": round(positive_rate, 3),
            "stable": stable,
            "all_values": dict(counts),
        }
    return results

# coding=utf-8
"""DEPRECATED: 全样本 IR 排序 (含 look-ahead) — 仅供 ablation 脚本对照.

[Stage 28 决策] 该函数被 _orthogonalize_panel 移出生产路径, 改抛 NotImplementedError.
              为保留 ablation 脚本 (5 个) 的对照能力, 复制到本测试 helper 文件,
              生产代码 (factor_orthogonal.py) 不再导出.

[Stage 29 决策] 该函数在 v6.2 文档中明确标记 DEPRECATED, 不应被新代码使用.
              新 ablation 可改用 ir_expanding (无 look-ahead) 作为对照.

原文件: QuantNodes/strategy/momentum_etf_rotation/v6_2/factor_orthogonal.py:224
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def get_factor_ir_order_deprecated(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    factors: Sequence[str],
    horizon: int = 21,
    min_periods: int = 12,
) -> list[str]:
    """[DEPRECATED, Stage 28] 原全样本 IR 排序实现 — 仅供 ablation 对照.

    ⚠️ 警告: 调仓日 t 用此函数 = 用未来 IC 决定正交化顺序, 严重 look-ahead.
              严禁在新代码中使用. 仅 scripts/v6_2_*_ablation.py 对照保留.

    Returns:
        list[str]  按 IR 降序排, 剔除 IR<=0 因子
    """
    from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import (
        compute_cross_section_ic,
    )

    records = {f: [] for f in factors}
    for d in rebalance_dates:
        ic = compute_cross_section_ic(factor_panel, panel_close, d, factors, horizon)
        for f in factors:
            if f in ic.index and not pd.isna(ic[f]):
                records[f].append(ic[f])

    ir = {}
    for f, vals in records.items():
        if len(vals) < min_periods:
            ir[f] = 0.0
            continue
        v = pd.Series(vals).dropna()
        m = v.mean()
        s = v.std()
        ir[f] = m / s if s > 0 else 0.0

    sorted_factors = sorted(ir.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_factors if ir[f] > 0]


__all__ = ["get_factor_ir_order_deprecated"]

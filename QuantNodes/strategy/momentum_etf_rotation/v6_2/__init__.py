# coding=utf-8
"""v6.2 模块 (Stage 27 v6.2) — v5.1.1 量价族 + IC 加权 + 因子正交化.

v6.2 = v5 选股 + IC 加权 + 因子正交化 (去除冗余) + v5.1.1 加权.

设计动机:
- v6.1 IC 加权有进步 (OOS Calmar 0.590 → 0.748, +27%)
- 但 IC 诊断显示因子间高度相关: f8↔f9 (0.78), f3↔f4 (0.60)
- 正交化去除冗余, 让 IC 加权"纯净" → 期望进一步提升

算法:
- 残差化 (Gram-Schmidt), 顺序按 OOS IR 降序
- 保留金融意义 (每个正交化因子仍叫原名)

回测目标 (2018-2026, 无风控):
- OOS Calmar ≥ v6.1 (0.748) → 期望 0.80+
- 全期 Calmar: 维持 ≥ 0.4

参考:
- v6.1: industry_rotation_v6_1.py
- 正交化: factor_orthogonal.py
"""
from .industry_rotation_v6_2 import (
    V6_2Config,
    V6_2SubStrategy,
    run_v6_2_backtest,
)
from .factor_orthogonal import (
    get_factor_ir_order,
    orthogonalize_factor_panel,
)
from ..v6_1.factor_weighting import (
    compute_cross_section_ic,
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
)

__all__ = [
    "V6_2Config",
    "V6_2SubStrategy",
    "run_v6_2_backtest",
    "get_factor_ir_order",
    "orthogonalize_factor_panel",
]

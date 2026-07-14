# coding=utf-8
"""v6.2 模块 (Stage 27 v6.2 + Stage 29 PROMISING + Stage 30 过拟合修复).

[Stage 30 状态] ⚠️ 研究版本 (扣成本后 Calmar 0.3310, 起点 CV% 56.9%)
  - v6.2 扣成本 (5bp+10bp) 后 Calmar: 0.3310 (-25.6% vs 不扣成本)
  - v6.2 起点依赖: CV% 56.9% (阈值 25%, FAIL)
  - v6.2 定位降级: 从 PROMISING → 研究版本

[Stage 29 状态] ⭐ PROMISING (历史记录)
  - v6.2 ir_expanding 5-fold OOS Calmar: mean=1.512, min=-0.016
  - v6.1 IC12 5-fold  OOS Calmar: mean=0.867, min=-0.604
  - v6.2 跨 fold 全部 ≥ v6.1 (除 fold 4)
  - 单次 OOS 2022-2026: 0.821 (v6.1: 0.748, +9.7%)

[Stage 28 状态] DEPRECATED 路径:
  - sort_method="ir_full" 全样本 IR (严重 look-ahead) — 见 tests/_helpers/deprecated_order.py
  - sort_method="qr" 对称正交 (Phase 3, OOS 0.056 失败) — 不推荐

[Stage 28 保留] 备选路径:
  - sort_method="warmup_ir" 12m (Phase 4 主推, 5-fold 不如 expanding)
  - sort_method="predefined" 金融预定义 (固定顺序, OOS 0.674)

v6.2 = v5 选股 + IC 加权 + 因子正交化 (去除冗余) + v5.1.1 加权.

设计动机:
- v6.1 IC 加权有进步 (OOS Calmar 0.590 → 0.748, +27%)
- 但 IC 诊断显示因子间高度相关: f8↔f9 (0.78), f3↔f4 (0.60)
- 正交化去除冗余, 让 IC 加权"纯净" → 期望进一步提升

算法:
- 残差化 (Gram-Schmidt), 顺序按 expanding IR 降序 (Stage 29 默认)
- 保留金融意义 (每个正交化因子仍叫原名)

回测目标 (2018-2026, 含成本):
- OOS Calmar ≥ v6.1 (0.748) → 扣成本后仅 0.3310, 未达成
- 5-fold walk-forward 验证: 已达成 4/5 胜 (历史, 未扣成本)
- 起点依赖: CV% 56.9% > 25%, 未达成

参考:
- v6.1: industry_rotation_v6_1.py
- 正交化: factor_orthogonal.py
- 5-fold 验证: reports/momentum_etf_rotation/combo/v6_2_ir_expanding_5fold.csv
- 过拟合修复: scripts/eval_v6_2_overfitting.py
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

# coding=utf-8
"""v5.1 策略模块 — v5 升级版, 引入逆波动率加权.

vs v5 (industry_rotation_v5):
- 选股逻辑相同: 11 量价因子 + Top-N + 月度调仓
- 唯一差异: 加权方式 (等权 → 逆波动率, 与 v1/v3 一致)
- max_weight 放宽到 0.30 (高波动品种自动降权, 上限可放宽)

回测 (2018-2026 8y, 口径 A 含 5bp 成本):
- v5   等权 Top-5:    Calmar 0.745  Sharpe 0.90  DD -19.41%  OOS 0.488
- v5.1 inv_vol Top-5: Calmar 0.774  Sharpe 0.98  DD -18.59%  OOS 0.589 ⭐

OOS Calmar 提升 +20.7% (0.488 → 0.589).

参考:
- v5: QuantNodes/strategy/momentum_etf_rotation/v5/
- v1 逆波动实现: v1/portfolio_v1.py:79 inverse_vol_weights_v1
- v3 逆波动实现: v3/industry_rotation_v3.py:183 weight()
"""
from .industry_rotation_v5_1 import (
    IndustryRotationV5_1Config,
    IndustryRotationV5_1SubStrategy,
    inverse_vol_weights_v5_1,
)

__all__ = [
    "IndustryRotationV5_1Config",
    "IndustryRotationV5_1SubStrategy",
    "inverse_vol_weights_v5_1",
]

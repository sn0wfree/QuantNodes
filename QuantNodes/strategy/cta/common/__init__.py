# coding=utf-8
"""CTA 共享基础设施.

与 momentum_etf_rotation/common 类似, 存放 CTA 策略共用的基础组件:
    - option_greeks: 期权定价/希腊字母计算 (vendored from ~/Public/option_cal/option_greeks)
    - metrics:       CTA 业绩指标
    - ...            (后续: 回测适配, 因子基类)
"""
from __future__ import annotations

from . import option_greeks

__all__ = ["option_greeks"]

# coding=utf-8
"""CTA 模块子策略包 — 每个模块一个文件, BUILTIN_MODULES 为唯一注册点.

添加新子策略:
    1. 新建 modules/<name>.py: 继承 _ModuleStrategy, 定义 FACTORS (见 base.py)
    2. 在下方 import + BUILTIN_MODULES 追加一行
    3. (可选) ComboConfig(module_classes=(...)) 选择启用子集
"""
from __future__ import annotations

from .event_calendar import EventCalendarModuleStrategy
from .industrial_chain import IndustrialChainModuleStrategy
from .macro import MacroModuleStrategy
from .oscillator_alpha import MeanReversionModuleStrategy, OscillatorAlphaModuleStrategy
from .term_basis import TermBasisModuleStrategy
from .trend_momentum import TrendMomentumModuleStrategy
from .volume_flow import VolumeFlowModuleStrategy

BUILTIN_MODULES = [
    TrendMomentumModuleStrategy,
    OscillatorAlphaModuleStrategy,
    VolumeFlowModuleStrategy,
    EventCalendarModuleStrategy,
    TermBasisModuleStrategy,
    MacroModuleStrategy,
    IndustrialChainModuleStrategy,
]

__all__ = [
    "BUILTIN_MODULES",
    "EventCalendarModuleStrategy",
    "IndustrialChainModuleStrategy",
    "MacroModuleStrategy",
    "MeanReversionModuleStrategy",
    "OscillatorAlphaModuleStrategy",
    "TermBasisModuleStrategy",
    "TrendMomentumModuleStrategy",
    "VolumeFlowModuleStrategy",
]

# coding=utf-8
"""多因子 CTA 策略 — 四模块子策略 + 组合 (策略 v3).

设计依据: docs/94-多因子CTA策略设计.md + 106 因子全库扩展调研

结构:
    信号层: 全因子 compute() 信号 (-1/0/+1) 统一投票, 因子级面板保留 (归因)
    合成层: 模块内等权求和 → 模块方向; 4 组: 趋势动量/均值回归/量能资金流/事件日历
    执行层: 模块加权合成净信号 → sign → vol_target_positions (单一资金池) → 回测
    过滤层: 波动率风险类因子 (HV/GARCH/统计/回撤等) 第二阶段接入

输入类型约定 (FACTORS 元组):
    panel      → 作用于 close 面板
    ohlc       → 逐品种 OHLC
    ohlcv      → 逐品种 OHLCV
    price_vol  → (close 面板, volume 面板)
    price_oi   → (close 面板, OI 面板)
    vols       → 逐品种 volume 面板
    ois        → 逐品种 OI 面板
    spread     → 全品种近远月期限结构 long-form (load_term_structure)
    basis      → 全品种现货基差 long-form (spot_basis_daily)
    macro_vix  → (close 面板, VIX 序列)
    macro_dxy  → (close 面板, DXY 序列)
    series     → 逐品种单序列 (每列独立 compute)
    chain      → 产业链因子, 按模块 CHAIN_MAP 映射到品种列

用法:
    from QuantNodes.strategy.cta.strategies.factor_modules import (
        TrendMomentumModuleStrategy, FactorModuleComboStrategy,
    )

    strat = TrendMomentumModuleStrategy(products=["AU", "RB"], start="2018-01-01")
    result = strat.run()
    print(result.metrics)
"""
from __future__ import annotations





# ── 基类/配置/品种池 (已拆至 base.py, 此处 re-export 保持兼容) ───────────
from .base import (  # noqa: E402,F401
    DEFAULT_PRODUCTS,
    ModuleConfig,
    _ModuleStrategy,
    vol_target_positions,
)

# ── 波动率风险过滤层 (已拆至 filters.py, 此处 re-export 保持兼容) ──────────
from .filters import MarketRegimeFilter, VolRiskFilter  # noqa: E402,F401

# ── 模块子策略 (已拆至 modules/ 包, 此处 re-export 保持兼容) ────────────
from .modules import (  # noqa: E402,F401
    EventCalendarModuleStrategy,
    IndustrialChainModuleStrategy,
    MacroModuleStrategy,
    MeanReversionModuleStrategy,
    OscillatorAlphaModuleStrategy,
    TermBasisModuleStrategy,
    TrendMomentumModuleStrategy,
    VolumeFlowModuleStrategy,
)

# legacy: 历史上因子类可从本模块导入 (scripts/test_combined_regime_factors.py 依赖)
from .modules.trend_momentum import (  # noqa: E402,F401
    FastTurnPointFactor,
    ShortTermReversalTrendFactor,
    TurnPointBreakoutFactor,
)

# ── 组合策略 (已拆至 combo.py, 此处 re-export 保持兼容) ──────────────────
from .combo import ComboConfig, DEFAULT_COMBO_MODULES, FactorModuleComboStrategy  # noqa: E402,F401


def __getattr__(name: str):
    """PEP 562 兜底: 其余历史属性 (因子类等) 延迟委托, 保持完全向后兼容."""
    from .. import factors as _factors
    val = getattr(_factors, name, None)
    if val is not None:
        return val
    from . import base as _base
    return getattr(_base, name)

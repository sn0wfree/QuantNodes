# coding=utf-8
"""对冲层框架: CTA 持仓 → 期权目标仓位 接口定义 (策略逻辑后延).

docs/98 §5.2: 基础设施优先, 本框架只定义接口与参数 (HedgeConfig),
具体对冲策略规则 (哪些品种/何时买/预算分配) 后续在接口上实现.

接口:
    compute_hedge_targets(cta_positions, iv_panel, prices, config)
        → dict[str, float]  {option_symbol: target_qty}

设计约束:
    - 无前视: 只用 t 日及之前数据
    - 预算控制: 年化权利金 ≤ budget_pct × NAV
    - IV 分位择时: 低 IV 全额买 / 高 IV 不买 (阈值可配)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class HedgeConfig:
    """对冲层配置 (参数化接口, 策略逻辑后填)."""

    budget_pct: float = 0.005       # 年化权利金预算 (NAV 的 0.5%)
    target_delta: float = 0.25      # 目标 delta (虚值 Put)
    hedge_ratio: float = 0.4        # 对冲净多头敞口比例
    min_days: int = 30              # 到期天数下限
    max_days: int = 60              # 到期天数上限
    iv_low_pct: float = 0.30        # IV 分位 < 30% → 全额买 (系数 1.0)
    iv_mid_pct: float = 0.70        # IV 分位 30-70% → 减半 (系数 0.5)
                                    # IV 分位 > 70% → 不买 (系数 0.0)
    min_net_notional: float = 0.0   # 净敞口阈值 (低于不对冲, 名义)
    products: list[str] = field(default_factory=list)  # 参与对冲的品种 (空=全部)
    roll_days: int = 7              # 到期前 7 天滚动展期


def iv_coefficient(iv_percentile: float, config: HedgeConfig) -> float:
    """IV 分位 → 买入系数 (0/0.5/1.0)."""
    if pd.isna(iv_percentile):
        return 1.0  # 无 IV 数据默认全额 (保守)
    if iv_percentile < config.iv_low_pct:
        return 1.0
    if iv_percentile < config.iv_mid_pct:
        return 0.5
    return 0.0


# ── 对冲目标计算接口 (策略逻辑占位) ─────────────────────────────────────

def compute_hedge_targets(
    cta_positions: pd.DataFrame,      # CTA 逐日持仓 (index=date, cols=product)
    iv_panel: pd.DataFrame,           # IV 面板 (date × product) + 分位
    prices: pd.DataFrame,             # 期货价格面板 (date × product)
    registry=None,                    # 合约规格 (计算名义价值)
    config: HedgeConfig | None = None,
) -> dict[str, float]:
    """CTA 持仓 → 期权目标仓位 {option_symbol: target_qty}.

    ⚠️ 策略逻辑后延: 本函数当前返回空 dict (框架占位).
    后续在此实现: 净多头敞口 → 品种选择 → IV 择时系数 → 合约选择 →
    数量计算 (budget/hedge_ratio 约束).
    """
    _ = (cta_positions, iv_panel, prices, registry, config)  # 占位, 防未使用
    return {}

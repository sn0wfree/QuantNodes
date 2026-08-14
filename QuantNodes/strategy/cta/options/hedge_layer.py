# coding=utf-8
"""对冲层: CTA 持仓 → 期权目标仓位 (纯 Put 尾部保护, docs/99 §3).

docs/98 §5.2 接口实现. MVP 决策 (docs/99, 用户确认):
    - 结构: 纯 Put (单腿虚值尾部保护)
    - 再平衡: 买入持有 (固定市值再平衡后延)
    - 择时: timing_mode 参数化 A/B 对照
        'iv'    IV 分位 (低 IV 全额 / 中减半 / 高不买)
        'basis' 月差/Backwardation 分位 (深贴水买 / 升水不买)
        'both'  min(iv_coef, basis_coef)    两者均允许才买
        'none'  1.0 (无择时基线)

接口:
    compute_hedge_targets(cta_positions, iv_panel, prices, ...)
        → dict[str, float]  {option_symbol: target_qty}  (单日)

设计约束:
    - 无前视: 只用 t 日及之前数据
    - 预算控制: 对冲名义 = hedge_ratio × 净敞口名义
    - IV / 月差 分位择时 (阈值可配)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .selector import select_put


@dataclass(slots=True)
class HedgeConfig:
    """对冲层配置 (参数化接口)."""

    budget_pct: float = 0.005       # 年化权利金预算 (NAV 的 0.5%)
    target_delta: float = 0.25      # 目标 delta (虚值 Put)
    hedge_ratio: float = 0.4        # 对冲净多头敞口比例
    min_days: int = 30              # 到期天数下限
    max_days: int = 60              # 到期天数上限
    iv_low_pct: float = 0.30        # IV 分位 < 30% → 全额买 (系数 1.0)
    iv_mid_pct: float = 0.70        # IV 分位 30-70% → 减半 (系数 0.5)
                                    # IV 分位 > 70% → 不买 (系数 0.0)
    basis_low_pct: float = 0.30     # 月差分位 < 30% (深升水) → 不买 (系数 0.0)
    basis_mid_pct: float = 0.70     # 月差分位 30-70% → 减半 (系数 0.5)
                                    # 月差分位 > 70% (深贴水) → 全额 (系数 1.0)
    timing_mode: str = "iv"         # 'iv' | 'basis' | 'both' | 'none'
    min_net_notional: float = 0.0   # 净敞口阈值 (低于不对冲, 名义)
    products: list[str] = field(default_factory=list)  # 参与对冲的品种 (空=全部)
    roll_days: int = 7              # 到期前 7 天滚动展期


def iv_coefficient(iv_percentile: float, config: HedgeConfig) -> float:
    """IV 分位 → 买入系数 (0/0.5/1.0). 低 IV 便宜 → 全额买."""
    if pd.isna(iv_percentile):
        return 1.0  # 无 IV 数据默认全额 (保守)
    if iv_percentile < config.iv_low_pct:
        return 1.0
    if iv_percentile < config.iv_mid_pct:
        return 0.5
    return 0.0


def basis_coefficient(basis_percentile: float, config: HedgeConfig) -> float:
    """月差/Backwardation 分位 → 买入系数 (0/0.5/1.0).

    spread_1_2 正值 = 近月 > 次月 = Backwardation (贴水, 市场脆弱).
    深贴水 (分位高) → 全额买 protect; 深升水 (分位低) → 不买.
    """
    if pd.isna(basis_percentile):
        return 1.0  # 无月差数据默认全额 (保守)
    if basis_percentile > config.basis_mid_pct:
        return 1.0
    if basis_percentile > config.basis_low_pct:
        return 0.5
    return 0.0


def timing_coefficient(
    iv_percentile: float,
    basis_percentile: float | None,
    config: HedgeConfig,
) -> float:
    """按 timing_mode 综合择时系数 (0/0.5/1.0)."""
    mode = config.timing_mode
    if mode == "none":
        return 1.0
    if mode == "basis" and basis_percentile is not None:
        return basis_coefficient(basis_percentile, config)
    if mode == "both":
        iv = iv_coefficient(iv_percentile, config)
        b = (basis_coefficient(basis_percentile, config)
             if basis_percentile is not None else 1.0)
        return min(iv, b)
    # 默认 'iv'
    return iv_coefficient(iv_percentile, config)


def _hedge_qty(
    net_qty: float,
    put_delta: float,
    ratio: float,
    premium: float = 0.0,
    multiplier: float = 1.0,
    budget_cap: float = 0.0,
) -> float:
    """Delta 对冲手数: hedge_ratio × qty / |delta|, 预算约束取 min.

    Args:
        net_qty: 净多头手数
        put_delta: put 的 delta (负值)
        ratio: 对冲比例 (hedge_ratio × timing_coef)
        premium: 单位权利金 (可选, 预算约束)
        multiplier: 合约乘数 (可选)
        budget_cap: 权利金预算上限 (可选, 0=不约束)
    Returns:
        目标手数 (向下取整)
    """
    abs_delta = abs(put_delta)
    if abs_delta < 1e-6 or net_qty <= 0:
        return 0.0
    n = net_qty * ratio / abs_delta
    if budget_cap > 0 and premium > 0 and multiplier > 0:
        max_by_budget = budget_cap / (premium * multiplier)
        n = min(n, max_by_budget)
    return float(np.floor(n))


# ── 对冲目标计算接口 ───────────────────────────────────────────────────

def compute_hedge_targets(
    cta_positions: pd.DataFrame,      # CTA 单日持仓 (index=[date], cols=product)
    iv_panel: pd.DataFrame,           # IV 面板 (date × product) 分位
    prices: pd.DataFrame,             # 期货价格面板 (date × product)
    registry=None,                    # 合约规格 (registry.select multiplier)
    config: HedgeConfig | None = None,
    basis_panel: pd.DataFrame | None = None,  # 月差分位面板 (date × product)
    date: str | pd.Timestamp | None = None,   # 目标日 (默认 index 最后一日)
    select_fun: Callable | None = None,       # 合约选择器 (默认 select_put)
) -> dict[str, float]:
    """CTA 持仓 → 期权目标仓位 {option_symbol: target_qty} (单日).

    流程 (docs/99 §3.2):
        1. 净多头检测: 净敞口名义 > min_net_notional → 候选
        2. 择时系数: timing_coefficient (iv/basis/both/none)
        3. 合约选择: select_fun → OptionSelection (premium)
        4. 数量: _hedge_qty (hedge_ratio × 净敞口 / (premium × 乘数))
        5. 返回 {option_symbol: qty}

    Args:
        cta_positions: 单日持仓 (index=[date]) || 面板 (取 date 行)
        iv_panel: IV 分位面板 [0,1]
        prices: 期货价格面板
        registry: 合约规格, 提供 select(product).contract_multiplier
        config: 对冲配置
        basis_panel: 月差分位面板 [0,1] (basis 模式用)
        date: 目标日 (默认取 cta_positions / iv_panel 最后一日)
        select_fun: 自定义合约选择器 (测试注入), 默认 select_put
    """
    config = config or HedgeConfig()
    select_fun = select_fun or select_put

    # 定位目标行
    idx = cta_positions.index
    if len(idx) == 0:
        return {}
    ts = pd.Timestamp(date) if date is not None else idx[-1]
    _row(cta_positions, ts)  # 校验日期存在

    targets: dict[str, float] = {}
    products = config.products or list(cta_positions.columns)
    for product in products:
        qty = _row(cta_positions, ts).get(product, 0.0)
        price = _row(prices, ts).get(product, np.nan)
        if pd.isna(price) or price <= 0 or abs(qty) < 1e-9:
            continue
        mult = _multiplier(registry, product)
        net_notional = abs(qty) * price * (mult or 1.0)
        if net_notional < config.min_net_notional:
            continue

        # 净多头 → 买 Put; 净空头 → MVP 不做 (单边 put 保护)
        if qty <= 0:
            continue

        iv_pct = _row(iv_panel, ts).get(product, np.nan)
        basis_pct = (None if basis_panel is None or basis_panel.empty
                     else _row(basis_panel, ts).get(product, np.nan))
        coef = timing_coefficient(iv_pct, basis_pct, config)
        if coef <= 0:
            continue

        sel = select_fun(product, ts, price, target_delta=config.target_delta,
                         min_days=config.min_days, max_days=config.max_days)
        if sel is None:
            continue
        n = _hedge_qty(
            net_qty=qty, put_delta=sel.delta,
            ratio=config.hedge_ratio * coef,
            premium=sel.premium, multiplier=mult or 1.0,
            budget_cap=config.budget_pct * 1_000_000,  # 年化预算 (简化)
        )
        if n > 0:
            targets[sel.symbol] = n
    return targets


# ── 内部工具 ───────────────────────────────────────────────────────────

def _row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    """取 df 中 ts 对应行 (找不到返回全 NaN 行)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if ts in df.index:
        return df.loc[ts]
    # 面板可能含多日, 取最近 <= ts 的行
    before = df.index[df.index <= ts]
    if len(before) == 0:
        return pd.Series(dtype=float)
    return df.loc[before[-1]]


def _multiplier(registry, product: str) -> float | None:
    """从 registry 取品种乘数 (兼容 registry.select / get)."""
    if registry is None:
        return None
    getter = getattr(registry, "select", None) or getattr(registry, "get", None)
    if getter is None:
        return None
    try:
        spec = getter(product)
        return getattr(spec, "contract_multiplier", None)
    except Exception:
        return None

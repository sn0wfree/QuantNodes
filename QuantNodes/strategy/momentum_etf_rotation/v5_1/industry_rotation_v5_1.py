# coding=utf-8
"""v5.1 行业量价因子行业轮动 (Stage 25) — 逆波动率加权升级版.

v5.1 vs v5 唯一差异: 加权方式 (等权 → 逆波动率, 与 v1/v3 一致).
所有其他逻辑 (11 因子 / 截面 z-score / Top-N / 月度调仓) 保持不变.

逆波动率加权 (inverse_vol_weights):
- 取 as_of 前 vol_window 日 (默认 21) 对数收益
- 各 code 年化波动率 σ = std × √252
- 权重 ∝ 1/max(σ, vol_floor)
- 归一化和为 1
- 套用 max_weight 上限 (默认 0.30, 比 v5 的 0.20 放宽)

回测结果 (口径 A 5bp 成本, 2018-2026):
- 全期 Calmar: 0.745 → 0.774 (+3.9%)
- 全期 Sharpe: 0.90  → 0.98  (+8.9%)
- OOS Calmar:  0.488 → 0.589 (+20.7%) ⭐
- OOS Sharpe:  0.60  → 0.71  (+18.3%)

OOS 月度最大回撤: -19.41% → -18.13% (改善 1.28pp).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..v4.sub_strategy_v4 import (
    SubStrategyConfig,
    SubStrategyResult,
)
from ..v5.industry_factors import FactorEngineConfig
from ..v5.industry_rotation_v5 import (
    IndustryRotationV5SubStrategy,
    cross_section_zscore,
    compute_composite_factor,
)


@dataclass
class IndustryRotationV5_1Config(SubStrategyConfig):
    """v5.1 配置: 11 量价因子 + Top-N + 逆波动率加权.

    唯一区别于 IndustryRotationV5Config: 默认 max_weight=0.30.
    其他字段 (top_n, min_history, factor_cfg) 与 v5 完全相同.
    """
    name: str = "industry_rotation_v5_1"

    # 因子引擎配置 (与 v5 共享)
    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)

    # Top-N 选择
    top_n: int = 5

    # 复合因子权重 (None = 等权)
    factor_weights: dict[str, float] | None = None

    # 调仓 (月度)
    rebalance_freq: str = "M"

    # 冷启动
    min_history: int = 252

    # ETF 池 (None = 全部)
    universe: tuple[str, ...] | None = None

    # 最大单只 ETF 权重
    # v5.1 逆波动下, 高波动品种自动降权, 上限放宽到 0.30
    max_weight: float = 0.30

    # 逆波动率窗口 (与 v1/v2 一致)
    vol_window: int = 21

    # 波动率下限 (防 1/0 爆发)
    vol_floor: float = 1e-4


def inverse_vol_weights_v5_1(
    nav_df: pd.DataFrame,
    codes: Sequence[str],
    as_of: pd.Timestamp,
    vol_window: int = 21,
    vol_floor: float = 1e-4,
) -> dict[str, float]:
    """v5.1 逆波动率加权 (与 v1/v3 一致, 21日窗口).

    逻辑:
    1. 取 as_of 前 vol_window 日的对数收益
    2. 各 code 年化波动率 σ = std × √252
    3. 权重 ∝ 1/max(σ, vol_floor)
    4. 归一化和为 1

    Args:
        nav_df: 价格面板 (close), 列为 code
        codes: 候选 ETF 列表
        as_of: 调仓日
        vol_window: 波动率计算窗口 (默认 21, 与 v1/v2 一致)
        vol_floor: 波动率下限 (防 1/0)

    Returns:
        dict[code, weight], 权重和 = 1
    """
    if not codes:
        return {}
    if nav_df is None or as_of is None:
        return {c: 1.0 / len(codes) for c in codes}

    valid_codes = [c for c in codes if c in nav_df.columns]
    if not valid_codes:
        return {c: 1.0 / len(codes) for c in codes}

    sub = nav_df.loc[:as_of, valid_codes]
    if len(sub) < vol_window + 1:
        return {c: 1.0 / len(codes) for c in codes}

    log_ret = np.log(sub / sub.shift(1))
    recent = log_ret.tail(vol_window)

    vols: dict[str, float] = {}
    for c in valid_codes:
        valid = recent[c].dropna()
        if len(valid) >= 2:
            v_std = valid.std()
            v = float(v_std) if np.isscalar(v_std) else float(v_std.iloc[0])
            vols[c] = v * np.sqrt(252)
        else:
            vols[c] = 1.0

    inv = {c: 1.0 / max(v, vol_floor) for c, v in vols.items()}
    total = sum(inv.values())
    if total <= 0:
        return {c: 1.0 / len(codes) for c in codes}

    weights = {c: v / total for c, v in inv.items()}
    for c in codes:
        weights.setdefault(c, 0.0)
    return weights


class IndustryRotationV5_1SubStrategy(IndustryRotationV5SubStrategy):
    """v5.1 子策略: 复用 v5 的 select/run_step, 仅替换 weight.

    继承 IndustryRotationV5SubStrategy, 避免复制 select/run_step 代码.
    """
    config: IndustryRotationV5_1Config

    def __init__(self, config: IndustryRotationV5_1Config):
        super().__init__(config)
        self.config: IndustryRotationV5_1Config = config

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """逆波动率加权 (权重 ∝ 1/σ, 21日窗口)."""
        if not codes:
            return {}
        cfg = self.config
        weights = inverse_vol_weights_v5_1(
            nav_df, codes, as_of, cfg.vol_window, cfg.vol_floor,
        )
        weights = self._apply_max_weight(weights, cfg.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights


__all__ = [
    "IndustryRotationV5_1Config",
    "IndustryRotationV5_1SubStrategy",
    "inverse_vol_weights_v5_1",
]

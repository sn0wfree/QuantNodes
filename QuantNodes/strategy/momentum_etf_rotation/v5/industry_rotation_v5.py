# coding=utf-8
"""v5 行业量价因子行业轮动 (Stage 22).

基于华西证券《行业有效量价因子与行业轮动策略》 (2022-08-22):
- 11 个量价因子 (动量/交易波动/换手率/多空对比/量价背离/量幅同向)
- 复合因子 = z-score 等权 (论文用 IC 加权)
- 月末选 Top-N ETF 等权

v5 vs Stage 19 industry_factors.py:
- 升级为完整 SubStrategy (继承 v4.sub_strategy_v4)
- 完整接口: select/weight/run_step
- 可与 v3/v4 在 multi_strategy 框架下组合

参考:
  - 华西证券《行业有效量价因子与行业轮动策略》
  - reports/momentum_etf_rotation/v4/INDUSTRY_ROTATION_REPORT.md
  - reports/momentum_etf_rotation/v4/papers/huaxi_industry_rotation.pdf
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..v4.sub_strategy_v4 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
)
from .industry_factors import (
    FactorEngineConfig,
    compute_all_factors_panel,
)


@dataclass
class IndustryRotationV5Config(SubStrategyConfig):
    """v5 行业量价因子行业轮动配置.

    默认值 (基于华西论文 + Stage 19 验证):
    - top_n: 5 (论文推荐, Stage 19 Top-N 扫描确认)
    - min_history: 252 (1 年)
    - max_weight: 0.20 (Top-5 等权 = 0.20)
    - 复合因子: 等权 z-score (论文用 IC 加权, 我们用等权)
    """
    name: str = "industry_rotation_v5"

    # 因子引擎配置
    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)

    # Top-N 选择
    top_n: int = 5

    # 复合因子权重 (None = 等权)
    factor_weights: dict[str, float] | None = None

    # 调仓
    rebalance_freq: str = "M"  # 月频 (论文: M)

    # 冷启动
    min_history: int = 252

    # ETF 池 (None = 全部)
    universe: tuple[str, ...] | None = None

    # 最大单只 ETF 权重 (Top-5 等权 = 0.20)
    max_weight: float = 0.20


def cross_section_zscore(
    factor_panel: dict[str, pd.DataFrame],
    factor: str,
    as_of: pd.Timestamp,
) -> pd.Series:
    """截面 z-score: 在 as_of 日, 各 code 的 factor 值, 去均值/std."""
    values = {}
    for code, df in factor_panel.items():
        if factor not in df.columns:
            continue
        if as_of not in df.index:
            idx = df.index.get_indexer([as_of], method="ffill")[0]
            if idx < 0:
                continue
            val = df[factor].iloc[idx]
        else:
            val = df[factor].loc[as_of]
        if pd.isna(val):
            continue
        values[code] = float(val)
    s = pd.Series(values)
    if s.empty:
        return s
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0


def compute_composite_factor(
    factor_panel: dict[str, pd.DataFrame],
    cfg: FactorEngineConfig,
    as_of: pd.Timestamp,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """计算 as_of 日各 code 的复合因子 = Σ w_i × zscore(f_i)."""
    factors = list(cfg.name_map.keys())
    if weights is None:
        weights = {f: 1.0 / len(factors) for f in factors}

    composite = pd.Series(dtype=float)
    for fac in factors:
        w = weights.get(fac, 0.0) if weights else 1.0 / len(factors)
        if w == 0:
            continue
        z = cross_section_zscore(factor_panel, fac, as_of)
        composite = composite.add(z * w, fill_value=0.0)
    return composite


class IndustryRotationV5SubStrategy(SubStrategy):
    """v5 行业量价因子行业轮动子策略.

    选股逻辑:
        1. 每月末计算 11 个量价因子 (OHLCV 数据)
        2. 各因子 z-score 标准化 (截面)
        3. 复合因子 = 等权加总 (默认)
        4. 选 Top-N (默认 5) ETF, 等权持仓
        5. 月频调仓

    调仓: 月度
    """

    def __init__(self, config: IndustryRotationV5Config):
        super().__init__(config)
        self.config: IndustryRotationV5Config = config
        self._factor_panel: dict[str, pd.DataFrame] | None = None
        self._last_init_date: pd.Timestamp | None = None

    def _init_factor_panel(self, nav_df: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """懒加载: 计算 11 因子. 因计算昂贵, 只在 init/as_of 变化时重算."""
        if (self._factor_panel is not None
                and self._last_init_date is not None
                and as_of <= self._last_init_date + pd.DateOffset(months=1)):
            return
        self._factor_panel = compute_all_factors_panel(nav_df, self.config.factor_cfg)
        self._last_init_date = as_of

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        cfg = self.config
        if cfg.min_history > 0 and len(nav_df) < cfg.min_history:
            return []
        if as_of not in nav_df.index:
            idx = nav_df.index.get_indexer([as_of], method="ffill")[0]
            if idx < 0:
                return []
            as_of = nav_df.index[idx]

        self._init_factor_panel(nav_df, as_of)
        if not self._factor_panel:
            return []

        composite = compute_composite_factor(
            self._factor_panel, cfg.factor_cfg, as_of, cfg.factor_weights,
        )
        if len(composite) < cfg.top_n:
            return []

        top = composite.nlargest(cfg.top_n)
        return list(top.index)

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """Top-N 等权 (默认 max_weight=0.20 = 1/5)."""
        if not codes:
            return {}
        n = len(codes)
        w = 1.0 / n
        weights = {c: w for c in codes}
        weights = self._apply_max_weight(weights, self.config.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        cfg = self.config
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, meta={"strategy": cfg.name})

        weights = self.weight(nav_df, codes, as_of)
        if not weights:
            return SubStrategyResult(date=as_of, meta={"strategy": cfg.name})

        if as_of in nav_df.index:
            self._init_factor_panel(nav_df, as_of)
        if self._factor_panel:
            composite = compute_composite_factor(
                self._factor_panel, cfg.factor_cfg, as_of, cfg.factor_weights,
            )
            if len(composite) > 0:
                signal = float(composite.loc[list(weights.keys())].mean())
            else:
                signal = 0.0
        else:
            signal = 0.0

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            signal_strength=signal,
            meta={
                "strategy": cfg.name,
                "top_n": cfg.top_n,
                "factor_engine": "11 量价因子",
                "composite": "z-score 等权" if cfg.factor_weights is None else "z-score 加权",
            },
        )


__all__ = [
    "IndustryRotationV5Config",
    "IndustryRotationV5SubStrategy",
    "cross_section_zscore",
    "compute_composite_factor",
]

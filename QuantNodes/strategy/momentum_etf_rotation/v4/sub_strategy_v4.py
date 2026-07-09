# coding=utf-8
"""v4 子策略抽象基类 (Stage 17).

v4 的子策略与 v3 完全独立, 复用 SubStrategy 接口但实现简化为:
- 不依赖 ETFPool (v4 universe 是固定 12 ETF)
- 不应用 category cap (v4 ETF 池已分组)
- 通用 select/weight/run_step 接口
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


@dataclass
class SubStrategyResult:
    """子策略单次调仓的输出.

    Attributes:
        date: 调仓日
        chosen: 选中的 ETF code 列表 (按优先级)
        weights: ETF code -> 权重 (和为 1)
        signal_strength: 信号强度 (0~1, 用于因子择时权重)
        meta: 子策略特定元数据
    """
    date: pd.Timestamp
    chosen: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    signal_strength: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class SubStrategyConfig:
    """子策略基类配置."""
    name: str = "base"
    top_n: int = 5
    max_weight: float = 0.20
    min_history: int = 144


class SubStrategy(ABC):
    """v4 子策略抽象基类.

    与 v3 区别:
    - 不需要 pool 参数 (v4 池固定)
    - 不应用 category cap
    - 接口更简洁
    """

    def __init__(self, config: SubStrategyConfig):
        self.config = config

    @abstractmethod
    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选股: 返回 top_n 个 ETF code 列表."""
        raise NotImplementedError

    @abstractmethod
    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """加权: 返回 code -> weight 字典 (和为 1)."""
        raise NotImplementedError

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        """单次调仓: select + weight + max_weight 约束."""
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, meta={"strategy": self.config.name})

        weights = self.weight(nav_df, codes, as_of)
        if self.config.max_weight < 1.0:
            weights = self._apply_max_weight(weights, self.config.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            meta={"strategy": self.config.name},
        )

    @staticmethod
    def _apply_max_weight(
        weights: dict[str, float],
        max_w: float,
    ) -> dict[str, float]:
        """约束单只 ETF 权重上限.

        实现: 迭代将超额部分按"封顶剩余空间"比例分配给非封顶标的.
        调用方应做最终归一化 (本方法不保证 sum=1, 只保证单项 ≤ max_w).
        若无法全部承接 (如非封顶标的只有 1 个且 capacity < excess),
        接受该项封顶, 剩余 excess 留在下一次迭代继续尝试, 至多 50 轮.
        """
        if not weights or max_w >= 1.0:
            return weights
        result = dict(weights)
        for _ in range(50):
            # 1. 截超过 max_w 的, 累计 excess
            excess_total = 0.0
            for c, w in result.items():
                if w > max_w:
                    excess_total += w - max_w
                    result[c] = max_w
            if excess_total <= 1e-9:
                break
            # 2. 找非封顶标的
            non_capped = [c for c, w in result.items() if w < max_w - 1e-12]
            if not non_capped:
                break
            # 3. 按 capacity 比例分配
            capacity_total = sum(max_w - result[c] for c in non_capped)
            if capacity_total <= 1e-9:
                break
            for c in non_capped:
                cap = max_w - result[c]
                share = excess_total * (cap / capacity_total)
                result[c] = min(max_w, result[c] + share)
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.config.name}>"


__all__ = [
    "SubStrategy",
    "SubStrategyConfig",
    "SubStrategyResult",
]

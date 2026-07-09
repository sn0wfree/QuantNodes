# coding=utf-8
"""子策略抽象基类 (Stage 16A, v3.0).

多策略组合 (Multi-Strategy Combination) 的基础:
- 动量策略 (现有 v2, 从 v2 复用)
- 均值反转策略 (新)
- 行业轮动策略 (新)

所有子策略必须实现 SubStrategy 接口, 统一:
- select()       : 选股 (返回 ETF code 列表)
- weight()       : 加权 (返回 code -> weight 字典)
- run_step()     : 单次调仓 (返回 SubStrategyResult)

参考 stage16a_plan.md §2.1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from ..common.universe import ETFPool


@dataclass
class SubStrategyResult:
    """子策略单次调仓的输出.

    Attributes:
        date: 调仓日
        chosen: 选中的 ETF code 列表 (按优先级)
        weights: ETF code -> 权重 (和为 1)
        signal_strength: 信号强度 (0~1, 用于风险平价等场景)
        meta: 子策略特定元数据 (e.g. 动量得分、反转得分、行业评分)
    """
    date: pd.Timestamp
    chosen: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    signal_strength: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class SubStrategyConfig:
    """子策略基类配置.

    Attributes:
        name: 子策略名称 (e.g. "momentum", "reversion", "industry_rotation")
        top_n: 持仓数
        max_weight: 单只最大权重 (默认 0.15)
        min_history: 最少需要的历史天数
    """
    name: str = "base"
    top_n: int = 10
    max_weight: float = 0.15
    min_history: int = 144


class SubStrategy(ABC):
    """子策略抽象基类.

    所有多策略组合的子策略 (动量/反转/行业轮动) 都继承本类.

    实现要求:
        1. select(): 从 ETF 池中选出 top_n 个 ETF code
        2. weight(): 对选中的 ETF 加权 (返回 code -> weight 字典)
        3. run_step(): 单次调仓 (组合 select + weight)

    子策略独立性:
        - 不依赖其他子策略的状态
        - 不修改外部状态 (除自身 config)
        - 可独立回测
    """

    def __init__(self, config: SubStrategyConfig, pool: ETFPool):
        self.config = config
        self.pool = pool

    @abstractmethod
    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选股: 返回 top_n 个 ETF code 列表 (按优先级降序).

        Args:
            nav_df: 价格面板 (index=date, columns=code)
            as_of: 当前日期

        Returns:
            list[str]: ETF code 列表, 长度 <= top_n

        Notes:
            - 子类应根据自身信号 (动量/反转/行业) 排序
            - 必须过滤: 排除不在 pool 中的、NaN 价格太长的
            - 不需要应用 caps (caps 在主策略统一应用)
        """
        raise NotImplementedError

    @abstractmethod
    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """加权: 对选中的 ETF 加权.

        Args:
            nav_df: 价格面板
            codes: 选中的 ETF code 列表
            as_of: 当前日期

        Returns:
            dict[str, float]: code -> weight, 和为 1
        """
        raise NotImplementedError

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        """单次调仓: 组合 select + weight.

        默认实现: select() → weight() → 归一化 → 应用 max_weight 约束

        子类可重写以加入额外逻辑 (e.g. 动量策略加 trend filter)
        """
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of)

        weights = self.weight(nav_df, codes, as_of)

        # 应用 max_weight 约束
        if self.config.max_weight < 1.0:
            weights = self._apply_max_weight(weights, self.config.max_weight)

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
        )

    @staticmethod
    def _apply_max_weight(
        weights: dict[str, float],
        max_w: float,
    ) -> dict[str, float]:
        """约束单只 ETF 权重上限 (max_w).

        算法: 迭代缩放 (类似 v2 concentration caps)
        """
        if not weights or max_w >= 1.0:
            return weights
        result = dict(weights)
        for _ in range(10):  # 最多 10 轮迭代
            excess_total = 0.0
            for c, w in result.items():
                if w > max_w:
                    excess_total += w - max_w
                    result[c] = max_w
            if excess_total <= 1e-6:
                break
            # 将超出部分按比例分配到未超额的部分
            non_capped = [c for c, w in result.items() if w < max_w]
            non_capped_sum = sum(result[c] for c in non_capped)
            if non_capped_sum > 0 and non_capped:
                for c in non_capped:
                    result[c] += excess_total * (result[c] / non_capped_sum)
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.config.name} top_n={self.config.top_n}>"


# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------
def validate_sub_strategy_result(
    result: SubStrategyResult,
    pool: ETFPool,
) -> SubStrategyResult:
    """校验子策略结果.

    - 检查所有 code 在 pool 中
    - 检查权重和为 1
    - 检查权重非负
    """
    if not result.chosen:
        return result

    # 过滤未知 code
    valid_chosen = [c for c in result.chosen if c in pool.codes]
    valid_weights = {c: w for c, w in result.weights.items() if c in pool.codes}

    # 归一化
    total = sum(valid_weights.values())
    if total > 0:
        valid_weights = {c: w / total for c, w in valid_weights.items()}

    # 权重非负
    valid_weights = {c: max(0.0, w) for c, w in valid_weights.items()}

    return SubStrategyResult(
        date=result.date,
        chosen=valid_chosen,
        weights=valid_weights,
        signal_strength=result.signal_strength,
        meta=result.meta,
    )


__all__ = [
    "SubStrategy",
    "SubStrategyConfig",
    "SubStrategyResult",
    "validate_sub_strategy_result",
]

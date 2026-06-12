"""ParentSelector — 5 种父辈选择策略。

借鉴 QuantaAlpha `configs/experiment.yaml:73 parent_selection_strategy`。
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from .entry import TrajectoryEntry


class SelectionStrategy(str, Enum):
    """5 种选择策略。"""
    BEST = "best"
    RANDOM = "random"
    WEIGHTED = "weighted"
    WEIGHTED_INVERSE = "weighted_inverse"
    TOP_PERCENT_PLUS_RANDOM = "top_percent_plus_random"


class ParentSelector:
    """父辈选择策略 — 从 TrajectoryPool 选 n 个 entry 作为下一轮 parent。

    Args:
        strategy: 策略名 (BEST/RANDOM/WEIGHTED/WEIGHTED_INVERSE/TOP_PERCENT_PLUS_RANDOM)
        metric: 用于 best / weighted 的指标 (默认 'sharpe')
        top_percent_threshold: top_percent_plus_random 中 top 比例 (默认 0.3)
        seed: 随机种子 (None=不固定)
        temperature: M4 weighted 策略的 softmax 温度 (默认 1.0,
            > 1.0 更均匀采样, < 1.0 更集中于高分)
    """

    def __init__(
        self,
        strategy: str = "best",
        metric: str = "sharpe",
        top_percent_threshold: float = 0.3,
        seed: int | None = None,
        temperature: float = 1.0,
    ):
        valid = {s.value for s in SelectionStrategy}
        if strategy not in valid:
            raise ValueError(
                f"未知 strategy: {strategy!r}, 应为 {sorted(valid)}"
            )
        self.strategy = strategy
        self.metric = metric
        self.top_percent_threshold = top_percent_threshold
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        pool: list[TrajectoryEntry] | "TrajectoryPool",
        n: int = 1,
    ) -> list[TrajectoryEntry]:
        """从 pool 选 n 个 entry (只选 feedback.decision=True 的)。"""
        from .pool import TrajectoryPool  # 避免循环 import

        if isinstance(pool, TrajectoryPool):
            valid = [e for e in pool.all() if e.feedback and e.feedback.decision]
        else:
            valid = [e for e in pool if e.feedback and e.feedback.decision]
        if not valid:
            return []

        if self.strategy == SelectionStrategy.BEST.value:
            return self._best(valid, n)
        if self.strategy == SelectionStrategy.RANDOM.value:
            return self._random(valid, n)
        if self.strategy == SelectionStrategy.WEIGHTED.value:
            return self._weighted_sample(valid, n, inverse=False)
        if self.strategy == SelectionStrategy.WEIGHTED_INVERSE.value:
            return self._weighted_sample(valid, n, inverse=True)
        if self.strategy == SelectionStrategy.TOP_PERCENT_PLUS_RANDOM.value:
            return self._top_percent_plus_random(valid, n)
        return []

    # ------------------------------------------------------------------
    # 5 个策略
    # ------------------------------------------------------------------

    def _best(self, valid: list[TrajectoryEntry], n: int) -> list[TrajectoryEntry]:
        sorted_entries = sorted(
            valid,
            key=lambda e: float(e.metrics.get(self.metric, 0) or 0),
            reverse=True,
        )
        return sorted_entries[:n]

    def _random(self, valid: list[TrajectoryEntry], n: int) -> list[TrajectoryEntry]:
        k = min(n, len(valid))
        indices = self._rng.choice(len(valid), size=k, replace=False)
        return [valid[int(i)] for i in indices]

    def _weighted_sample(
        self,
        valid: list[TrajectoryEntry],
        n: int,
        inverse: bool,
    ) -> list[TrajectoryEntry]:
        scores = np.array(
            [float(e.metrics.get(self.metric, 0) or 0) for e in valid],
            dtype=float,
        )
        if inverse:
            scores = -scores
        scores = scores - scores.max()
        # M4: temperature 调节 softmax 锐度
        #   T → 0: 趋近 argmax; T → ∞: 趋近均匀; T=1: 标准 softmax
        weights = np.exp(scores / max(self.temperature, 1e-9))
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            return self._random(valid, n)
        weights = weights / total
        k = min(n, len(valid))
        indices = self._rng.choice(len(valid), size=k, replace=False, p=weights)
        return [valid[int(i)] for i in indices]

    def _top_percent_plus_random(
        self,
        valid: list[TrajectoryEntry],
        n: int,
    ) -> list[TrajectoryEntry]:
        top_n = max(1, int(len(valid) * self.top_percent_threshold))
        top = self._best(valid, top_n)
        if n <= top_n:
            return top[:n]
        rest = [e for e in valid if e.entry_id not in {t.entry_id for t in top}]
        if not rest:
            return top[:n]
        k = min(n - top_n, len(rest))
        extra_indices = self._rng.choice(len(rest), size=k, replace=False)
        return top + [rest[int(i)] for i in extra_indices]

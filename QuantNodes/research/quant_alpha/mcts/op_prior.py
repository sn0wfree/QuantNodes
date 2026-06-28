# coding=utf-8
"""
op_prior.py - 算子先验分布（Tier 4: feature/thinking-chain）

OpPrior 维护一个 operator_name -> weight 的分布，用于 MCTS _expand() 的
加权算子采样。权重基于历史成功公式（|IR| > 阈值）的算子出现频率。

设计要点:
1. **指数衰减**: alpha=0.7 保留历史 70%，新信号贡献 30%
2. **Floor 保护**: 最小权重 0.1，避免算子被永久遗忘
3. **持久化**: save/load JSON，支持跨 pipeline 累积
4. **混合采样**: 与均匀分布按 mix_ratio 混合，避免过拟合
5. **零依赖**: 仅用标准库（json, dataclasses, numpy）

Usage::

    prior = OpPrior()
    prior.update(ops=["rank", "ts_std"], ir=0.5)
    weights = prior.mix(["rank", "ts_std", "div", "mul"], mix_ratio=0.5)
    prior.save(Path("op_prior.json"))
    prior2 = OpPrior.load(Path("op_prior.json"))
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


__all__ = ["OpPrior"]


@dataclass
class OpPrior:
    """算子先验分布。

    Attributes:
        weights: op_name -> 权重 ∈ [floor, 1.0]
        alpha: 历史保留率（指数衰减），新信号 = 1 - alpha
        floor: 最小权重（避免零概率）
        total_updates: 累计 update() 调用次数（带正信号的）
        ir_threshold: |IR| 低于此值不更新（避免被噪声污染）
        ir_full_strength: |IR| 满强度参考值（默认 0.5）
    """

    weights: Dict[str, float] = field(default_factory=dict)
    alpha: float = 0.7
    floor: float = 0.1
    total_updates: int = 0
    ir_threshold: float = 0.01
    ir_full_strength: float = 0.5

    def update(self, ops: Iterable[str], ir: float) -> None:
        """根据一次成功公式更新先验。

        Args:
            ops: 公式中出现的算子列表（去重前后均可）
            ir: 公式的 IR（可正可负，使用 |IR|）
        """
        ops = list(ops)
        if not ops or abs(ir) < self.ir_threshold:
            return

        # 强度 = |IR| / reference，clip 到 [0, 1]
        strength = min(abs(ir) / self.ir_full_strength, 1.0)
        if strength <= 0:
            return

        for op in ops:
            if not isinstance(op, str) or not op:
                continue
            old = self.weights.get(op, 0.5)
            # 指数衰减：新值 = α · 旧值 + (1 - α) · strength
            new = old * self.alpha + strength * (1.0 - self.alpha)
            self.weights[op] = max(self.floor, min(1.0, new))

        self.total_updates += 1
        logger.debug(
            "OpPrior.update: ops=%s, ir=%.4f, strength=%.3f, total_updates=%d",
            ops, ir, strength, self.total_updates,
        )

    def get_weight(self, op: str) -> float:
        """获取单个算子的权重（未在 weights 中的返回默认 0.5）。"""
        return self.weights.get(op, 0.5)

    def sample_weights(self, all_ops: List[str]) -> np.ndarray:
        """返回与 all_ops 对齐的权重数组。

        Args:
            all_ops: 所有候选算子列表

        Returns:
            np.ndarray 形状 (len(all_ops),)，每个位置是算子权重
        """
        if not all_ops:
            return np.array([])
        return np.array([self.weights.get(op, 0.5) for op in all_ops])

    def mix(
        self,
        all_ops: List[str],
        mix_ratio: float = 0.5,
    ) -> np.ndarray:
        """先验与均匀分布的混合，返回归一化的概率分布。

        Args:
            all_ops: 所有候选算子列表
            mix_ratio: 0.0=纯均匀, 1.0=纯先验

        Returns:
            np.ndarray 形状 (len(all_ops),)，和为 1.0
        """
        if not all_ops:
            return np.array([])
        prior = self.sample_weights(all_ops)
        uniform = np.ones(len(all_ops))
        mixed = mix_ratio * prior + (1.0 - mix_ratio) * uniform
        total = mixed.sum()
        if total <= 0:
            return uniform / len(all_ops)
        return mixed / total

    def top_k(self, k: int = 10) -> List[tuple]:
        """返回权重最高的 k 个算子。"""
        sorted_ops = sorted(
            self.weights.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_ops[:k]

    def save(self, path: Path) -> None:
        """持久化到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "alpha": self.alpha,
            "floor": self.floor,
            "total_updates": self.total_updates,
            "ir_threshold": self.ir_threshold,
            "ir_full_strength": self.ir_full_strength,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("OpPrior saved: %d ops, %d updates → %s",
                    len(self.weights), self.total_updates, path)

    @classmethod
    def load(cls, path: Path) -> "OpPrior":
        """从 JSON 文件加载。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"OpPrior file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            weights=data.get("weights", {}),
            alpha=data.get("alpha", 0.7),
            floor=data.get("floor", 0.1),
            total_updates=data.get("total_updates", 0),
            ir_threshold=data.get("ir_threshold", 0.01),
            ir_full_strength=data.get("ir_full_strength", 0.5),
        )

    def to_dict(self) -> Dict:
        return {
            "weights": dict(self.weights),
            "alpha": self.alpha,
            "floor": self.floor,
            "total_updates": self.total_updates,
            "ir_threshold": self.ir_threshold,
            "ir_full_strength": self.ir_full_strength,
        }

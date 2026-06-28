# coding=utf-8
"""
mcts/extension_ops.py - MCTS 扩展操作（从 OperatorVocab 动态生成）

vs 旧 mcts_search.py:44-56 的 7 硬编码 EXTENSION_OPS 列表：
- 旧：7 个固定模板
- 新：从 OperatorVocab（162 算子）动态生成

设计：
- 包裹型：rank / zscore / scale / winsorize 等
- 窗口型：ts_mean / ts_std / ts_max / ts_min / ts_delta / ts_argmax / ts_argmin / ts_corr / ts_cov 等
- 差值型：{f} - ts_mean({f}, w)
- 比值型：{f} / ts_lag({f}, w) - 1
- 组合型：signedpower({f}, 2)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from QuantNodes.research.quant_alpha.operator_vocab import (
    OperatorVocab,
    OperatorVocabConfig,
)


# 默认窗口期
DEFAULT_WINDOWS = [5, 10, 20, 60]


@dataclass
class ExtensionOp:
    """MCTS 扩展操作（动态生成）"""
    name: str  # 操作类别名（用于统计）
    template: str  # 公式模板，含 {f} 和 {w} 占位符
    requires_window: bool = True  # 是否需要 {w} 参数
    min_inputs: int = 1  # 最少输入数
    max_inputs: int = 1  # 最多输入数（1=unary, 2=binary）
    category: str = "general"  # 分类：wrap / window / diff / ratio / unary / binary
    description: str = ""

    def instantiate(self, f: str, w: Optional[int] = None) -> str:
        """把模板实例化：替换 {f} 和 {w}"""
        if self.requires_window and w is not None:
            return self.template.replace("{f}", f).replace("{w}", str(w))
        return self.template.replace("{f}", f)


def _build_wrap_ops(vocab: OperatorVocab) -> List[ExtensionOp]:
    """从 OperatorVocab 构造包裹型操作（rank/zscore/scale/winsorize）"""
    ops = []
    # section 类算子
    section_candidates = ["rank", "zscore", "scale", "winsorize"]
    for name in section_candidates:
        if vocab.get_operator(name) is not None:
            ops.append(ExtensionOp(
                name=f"wrap_{name}",
                template=f"{name}({{f}})",
                requires_window=False,
                min_inputs=1,
                max_inputs=1,
                category="wrap",
                description=f"包裹 {name}（截面算子）",
            ))
    return ops


def _build_window_ops(vocab: OperatorVocab) -> List[ExtensionOp]:
    """从 OperatorVocab 构造窗口型操作（ts_*/rolling_*/ewm_*）"""
    ops = []
    # 时序类算子（unary, 需要窗口）
    window_candidates = [
        "ts_mean", "ts_std", "ts_max", "ts_min",
        "ts_delta", "ts_argmax", "ts_argmin", "ts_rank",
        "ts_skew", "ts_kurt", "ts_decay_linear",
    ]
    for name in window_candidates:
        if vocab.get_operator(name) is not None:
            ops.append(ExtensionOp(
                name=f"window_{name}",
                template=f"{name}({{f}}, {{w}})",
                requires_window=True,
                min_inputs=1,
                max_inputs=1,
                category="window",
                description=f"窗口算子 {name}(x, window)",
            ))

    # 二元窗口算子（ts_corr, ts_cov）
    binary_window_candidates = ["ts_corr", "ts_cov"]
    for name in binary_window_candidates:
        if vocab.get_operator(name) is not None:
            ops.append(ExtensionOp(
                name=f"window_{name}",
                template=f"{name}({{f}}, {{f2}}, {{w}})",  # {f2} = 第二输入
                requires_window=True,
                min_inputs=2,
                max_inputs=2,
                category="window_binary",
                description=f"二元窗口算子 {name}(x, y, window)",
            ))
    return ops


def _build_unary_ops(vocab: OperatorVocab) -> List[ExtensionOp]:
    """从 OperatorVocab 构造一元数学操作（abs/log/signedpower/sqrt/sign）"""
    ops = []
    unary_candidates = ["abs", "log", "signedpower", "sqrt", "sign"]
    for name in unary_candidates:
        if vocab.get_operator(name) is not None:
            if name == "signedpower":
                # signedpower 需要 2 参数
                ops.append(ExtensionOp(
                    name=f"unary_{name}",
                    template=f"{name}({{f}}, 2)",  # 默认指数=2
                    requires_window=False,
                    min_inputs=1,
                    max_inputs=1,
                    category="unary",
                    description=f"一元算子 {name}(x, 2)",
                ))
            else:
                ops.append(ExtensionOp(
                    name=f"unary_{name}",
                    template=f"{name}({{f}})",
                    requires_window=False,
                    min_inputs=1,
                    max_inputs=1,
                    category="unary",
                    description=f"一元算子 {name}(x)",
                ))
    return ops


def _build_diff_ops() -> List[ExtensionOp]:
    """差值型操作（无需注册表，硬编码）"""
    return [
        ExtensionOp(
            name="diff_mean",
            template="{f} - ts_mean({f}, {w})",
            requires_window=True,
            min_inputs=1,
            max_inputs=1,
            category="diff",
            description="差值：x - mean(x, w)",
        ),
        ExtensionOp(
            name="diff_lag",
            template="{f} - ts_lag({f}, {w})",
            requires_window=True,
            min_inputs=1,
            max_inputs=1,
            category="diff",
            description="差值：x - lag(x, w)",
        ),
    ]


def _build_ratio_ops() -> List[ExtensionOp]:
    """比值型操作（无需注册表，硬编码）"""
    return [
        ExtensionOp(
            name="ratio_lag",
            template="{f} / ts_lag({f}, {w}) - 1",
            requires_window=True,
            min_inputs=1,
            max_inputs=1,
            category="ratio",
            description="比值：x / lag(x, w) - 1（动量/收益率）",
        ),
        ExtensionOp(
            name="ratio_mean",
            template="{f} / ts_mean({f}, {w})",
            requires_window=True,
            min_inputs=1,
            max_inputs=1,
            category="ratio",
            description="比值：x / mean(x, w)",
        ),
    ]


class ExtensionOpPool:
    """MCTS 扩展操作池

    从 OperatorVocab 动态生成可用操作。
    集中管理 MCTS 搜索时的操作选择。
    """

    def __init__(
        self,
        vocab: Optional[OperatorVocab] = None,
        windows: Optional[List[int]] = None,
        include_categories: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.vocab = vocab or OperatorVocab.default()
        self.windows = windows or DEFAULT_WINDOWS
        self.rng = random.Random(seed)

        # 动态构建操作池
        all_ops: List[ExtensionOp] = []
        all_ops.extend(_build_wrap_ops(self.vocab))
        all_ops.extend(_build_window_ops(self.vocab))
        all_ops.extend(_build_unary_ops(self.vocab))
        all_ops.extend(_build_diff_ops())
        all_ops.extend(_build_ratio_ops())

        # 按 category 过滤
        if include_categories:
            self.ops = [op for op in all_ops if op.category in include_categories]
        else:
            self.ops = all_ops

        # 按 category 索引
        self._by_category: Dict[str, List[ExtensionOp]] = {}
        for op in self.ops:
            self._by_category.setdefault(op.category, []).append(op)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def list_categories(self) -> List[str]:
        return list(self._by_category.keys())

    def count_by_category(self) -> Dict[str, int]:
        return {cat: len(ops) for cat, ops in self._by_category.items()}

    def sample(self, category: Optional[str] = None) -> ExtensionOp:
        """随机选一个操作"""
        if category:
            ops = self._by_category.get(category, [])
        else:
            ops = self.ops
        if not ops:
            raise ValueError(
                f"No operations available"
                f"{f' in category {category}' if category else ''}"
            )
        return self.rng.choice(ops)

    def sample_weighted(
        self,
        op_prior: Optional[Any] = None,
        mix_ratio: float = 0.5,
        category: Optional[str] = None,
    ) -> ExtensionOp:
        """按 OpPrior 权重采样一个操作（Tier 4: feature/thinking-chain）。

        Args:
            op_prior: OpPrior 实例（None 时回退到均匀采样）
            mix_ratio: 0.0=纯均匀, 1.0=纯先验
            category: 可选 category 过滤

        Returns:
            ExtensionOp 实例
        """
        if category:
            ops = self._by_category.get(category, [])
        else:
            ops = self.ops
        if not ops:
            raise ValueError(
                f"No operations available"
                f"{f' in category {category}' if category else ''}"
            )
        if op_prior is None:
            return self.rng.choice(ops)

        # 加权采样
        op_names = [op.name for op in ops]
        weights = op_prior.mix(op_names, mix_ratio=mix_ratio)
        # numpy choice 用概率
        import numpy as _np
        idx = int(_np.random.choice(len(ops), p=weights))
        return ops[idx]

    def sample_window(self) -> int:
        """随机选一个窗口"""
        return self.rng.choice(self.windows)

    def get_seed_formulas(self, columns: List[str]) -> List[str]:
        """生成种子公式

        覆盖各 category：
        - wrap: rank(col), zscore(col), scale(col)
        - window: ts_mean(col, 20), ts_std(col, 20), ts_delta(col, 20)
        - unary: signedpower(col, 2), log(col)
        - diff: col - ts_mean(col, 20)
        - ratio: col / ts_lag(col, 20) - 1
        """
        # 取前 5 列做种子（避免组合爆炸）
        cols = columns[:5]
        seeds = []
        for col in cols:
            seeds.extend([
                f"rank({col})",
                f"zscore({col})",
                f"ts_mean({col}, 20)",
                f"ts_std({col}, 20)",
                f"ts_delta({col}, 20)",
                f"signedpower({col}, 2)",
                f"{col} - ts_mean({col}, 20)",
                f"{col} / ts_lag({col}, 20) - 1",
            ])
        return seeds

    def stats(self) -> Dict[str, Any]:
        """操作池统计"""
        return {
            "total": len(self.ops),
            "by_category": self.count_by_category(),
            "windows": list(self.windows),
        }

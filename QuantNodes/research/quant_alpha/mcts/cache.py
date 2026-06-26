# coding=utf-8
"""MCTSCache — 持久化共享缓存，跨运行复用公式评估结果。

缓存布局：
    {cache_root}/{data_fingerprint}/
        cache.json          # 反馈条目（人类可读）
        series_cache.pkl    # {formula: pl.Series} 字典（二进制，快速）

Usage::

    from QuantNodes.research.quant_alpha.mcts.cache import MCTSCache, MCTSCacheConfig

    cache = MCTSCache(MCTSCacheConfig(enabled=True))
    cache.load(data)  # 加载缓存

    # 使用缓存
    if cache.has_series(formula):
        result = cache.get_series(formula)
    else:
        result = evaluate(formula, data)
        cache.put_series(formula, result)

    cache.save()  # 持久化
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

import polars as pl

from QuantNodes.core.feedback import FactorFeedback

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path.home() / ".quantnodes" / "mcts_cache"


@dataclass
class MCTSCacheConfig:
    """缓存配置。"""
    cache_root: Path = DEFAULT_CACHE_ROOT
    enabled: bool = True
    max_series_entries: int = 5000
    max_series_bytes: int = 500_000_000  # 500MB


class MCTSCache:
    """持久化共享缓存，用于 MCTS 公式评估。

    特性：
    - 数据指纹自动失效：不同数据 = 不同缓存目录
    - 混合存储：JSON（反馈）+ pickle（Series）
    - 增量更新：只保存新评估的公式
    - 向后兼容：cache=None 保持当前行为
    """

    def __init__(self, config: Optional[MCTSCacheConfig] = None):
        self.config = config or MCTSCacheConfig()
        self._formula_cache: Dict[str, pl.Series] = {}
        self._feedback_cache: Dict[str, FactorFeedback] = {}
        self._cache_dir: Optional[Path] = None
        self._dirty = False

    # ----------------------------------------------------------
    # 数据指纹
    # ----------------------------------------------------------

    @staticmethod
    def compute_data_fingerprint(data: pl.DataFrame) -> str:
        """计算 DataFrame 的稳定哈希。

        使用：shape、column names、dtypes、首尾行采样。
        两个相同指纹的 DataFrame 对同一公式会产生相同的 pl.Series。
        """
        parts = [
            str(data.shape),
            str(sorted(data.columns)),
            str(sorted(str(d) for d in data.dtypes)),
        ]
        sample_size = min(100, len(data))
        if sample_size > 0:
            # 转换为字符串以确保确定性
            head_str = str(data.head(sample_size).rows())
            tail_str = str(data.tail(sample_size).rows())
            parts.append(hashlib.md5(head_str.encode()).hexdigest())
            parts.append(hashlib.md5(tail_str.encode()).hexdigest())
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

    # ----------------------------------------------------------
    # 加载 / 保存
    # ----------------------------------------------------------

    def load(self, data: pl.DataFrame) -> None:
        """从磁盘加载缓存。

        如果没有缓存或数据指纹不匹配，从空缓存开始。
        """
        if not self.config.enabled:
            return

        fp = self.compute_data_fingerprint(data)
        self._cache_dir = self.config.cache_root / fp
        json_path = self._cache_dir / "cache.json"
        pkl_path = self._cache_dir / "series_cache.pkl"

        if not self._cache_dir.exists():
            logger.info("No MCTS cache found at %s, starting fresh", self._cache_dir)
            return

        # 加载反馈
        if json_path.exists():
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                self._feedback_cache = {
                    k: FactorFeedback.from_dict(v)
                    for k, v in raw.get("entries", {}).items()
                }
                logger.info(
                    "Loaded %d feedback entries from %s",
                    len(self._feedback_cache), json_path,
                )
            except Exception as e:
                logger.warning("Failed to load feedback cache: %s", e)

        # 加载 Series
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    self._formula_cache = pickle.load(f)
                logger.info(
                    "Loaded %d series entries from %s",
                    len(self._formula_cache), pkl_path,
                )
            except Exception as e:
                logger.warning("Failed to load series cache: %s", e)

        self._dirty = False

    def save(self) -> None:
        """持久化当前缓存到磁盘。"""
        if not self.config.enabled or self._cache_dir is None:
            return
        if not self._dirty:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 保存反馈为 JSON
        json_path = self._cache_dir / "cache.json"
        payload = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "entry_count": len(self._feedback_cache),
            "series_count": len(self._formula_cache),
            "entries": {
                k: v.to_dict() for k, v in self._feedback_cache.items()
            },
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 保存 Series 为 pickle
        pkl_path = self._cache_dir / "series_cache.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self._formula_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "Saved MCTS cache: %d formulas, %d feedback entries to %s",
            len(self._formula_cache), len(self._feedback_cache), self._cache_dir,
        )
        self._dirty = False

    # ----------------------------------------------------------
    # 字典式 API（直接替换 _formula_cache / _feedback_cache）
    # ----------------------------------------------------------

    def get_series(self, formula: str) -> Optional[pl.Series]:
        """获取公式的评估结果。"""
        return self._formula_cache.get(formula)

    def put_series(self, formula: str, series: pl.Series) -> None:
        """存储公式的评估结果。"""
        self._formula_cache[formula] = series
        self._dirty = True

    def has_series(self, formula: str) -> bool:
        """检查公式是否已评估。"""
        return formula in self._formula_cache

    def get_feedback(self, formula: str) -> Optional[FactorFeedback]:
        """获取公式的 5 通道反馈。"""
        return self._feedback_cache.get(formula)

    def put_feedback(self, formula: str, fb: FactorFeedback) -> None:
        """存储公式的 5 通道反馈。"""
        self._feedback_cache[formula] = fb
        self._dirty = True

    def has_feedback(self, formula: str) -> bool:
        """检查公式是否已有反馈。"""
        return formula in self._feedback_cache

    def clear(self) -> None:
        """清空内存缓存。"""
        self._formula_cache.clear()
        self._feedback_cache.clear()
        self._dirty = False

    @property
    def formula_count(self) -> int:
        """缓存的公式数量。"""
        return len(self._formula_cache)

    @property
    def feedback_count(self) -> int:
        """缓存的反馈数量。"""
        return len(self._feedback_cache)

    # ----------------------------------------------------------
    # 失效 / 清理
    # ----------------------------------------------------------

    def invalidate(self) -> None:
        """删除当前数据集的整个缓存目录。"""
        if self._cache_dir and self._cache_dir.exists():
            import shutil
            shutil.rmtree(self._cache_dir)
            logger.info("Invalidated MCTS cache at %s", self._cache_dir)
        self.clear()
        self._cache_dir = None

    def prune(self, keep_formulas: Set[str]) -> int:
        """移除不在 keep_formulas 中的条目。返回移除数量。"""
        removed = 0
        for formula in list(self._formula_cache.keys()):
            if formula not in keep_formulas:
                del self._formula_cache[formula]
                removed += 1
        for formula in list(self._feedback_cache.keys()):
            if formula not in keep_formulas:
                del self._feedback_cache[formula]
                removed += 1
        if removed:
            self._dirty = True
        return removed


__all__ = [
    "MCTSCache",
    "MCTSCacheConfig",
    "DEFAULT_CACHE_ROOT",
]

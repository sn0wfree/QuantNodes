# coding=utf-8
"""Tests for MCTSCache — 持久化共享缓存。

覆盖：
- MCTSCacheConfig：默认值、自定义配置
- MCTSCache：数据指纹、加载/保存、增删改查
- 缓存失效：数据变更、手动失效
- 缓存清理：prune
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from QuantNodes.core.feedback import FactorFeedback, FeedbackChannel, ChannelFeedback
from QuantNodes.research.quant_alpha.mcts.cache import (
    MCTSCache,
    MCTSCacheConfig,
    DEFAULT_CACHE_ROOT,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成测试数据"""
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            rows.append({
                "date": date,
                "code": code,
                "close": float(np.random.randn() * 5 + 100),
                "open": float(np.random.randn() * 5 + 100),
                "high": float(np.random.randn() * 5 + 102),
                "low": float(np.random.randn() * 5 + 98),
                "vol": float(np.random.randint(1000, 5000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def different_data() -> pl.DataFrame:
    """不同的测试数据（用于测试缓存失效）"""
    np.random.seed(99)
    dates = [f"2024-02-{d:02d}" for d in range(1, 6)]
    rows = []
    for date in dates:
        for code in ["X", "Y"]:
            rows.append({
                "date": date,
                "code": code,
                "close": float(np.random.randn() * 10 + 200),
                "open": float(np.random.randn() * 10 + 200),
                "high": float(np.random.randn() * 10 + 210),
                "low": float(np.random.randn() * 10 + 190),
                "vol": float(np.random.randint(5000, 10000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def cache_config(tmp_path: Path) -> MCTSCacheConfig:
    """临时目录的缓存配置"""
    return MCTSCacheConfig(
        cache_root=tmp_path / "mcts_cache",
        enabled=True,
        max_series_entries=100,
        max_series_bytes=10_000_000,
    )


@pytest.fixture
def cache(cache_config: MCTSCacheConfig) -> MCTSCache:
    """MCTSCache 实例"""
    return MCTSCache(cache_config)


def _make_feedback(formula: str, score: float = 0.8) -> FactorFeedback:
    """创建测试用 FactorFeedback"""
    from datetime import datetime
    channels = {
        FeedbackChannel.EXECUTION: ChannelFeedback(
            channel=FeedbackChannel.EXECUTION,
            passed=True,
            score=score,
            detail="test",
        ),
    }
    return FactorFeedback(
        factor_id="test-id",
        factor_name=formula,
        channels=channels,
        decision=True,
        summary="OK",
        timestamp=datetime.now(),
        duration_ms=0.0,
        metadata={"score": score},
    )


# ==============================================================================
# TestMCTSCacheConfig
# ==============================================================================


class TestMCTSCacheConfig:
    """MCTSCacheConfig 测试"""

    def test_default_values(self):
        """默认值正确"""
        config = MCTSCacheConfig()
        assert config.cache_root == DEFAULT_CACHE_ROOT
        assert config.enabled is True
        assert config.max_series_entries == 5000
        assert config.max_series_bytes == 500_000_000

    def test_custom_values(self, tmp_path: Path):
        """自定义值"""
        config = MCTSCacheConfig(
            cache_root=tmp_path / "custom",
            enabled=False,
            max_series_entries=100,
            max_series_bytes=1_000_000,
        )
        assert config.cache_root == tmp_path / "custom"
        assert config.enabled is False
        assert config.max_series_entries == 100
        assert config.max_series_bytes == 1_000_000


# ==============================================================================
# TestMCTSCacheFingerprint
# ==============================================================================


class TestMCTSCacheFingerprint:
    """数据指纹测试"""

    def test_same_data_same_fingerprint(self, sample_data: pl.DataFrame):
        """相同数据产生相同指纹"""
        fp1 = MCTSCache.compute_data_fingerprint(sample_data)
        fp2 = MCTSCache.compute_data_fingerprint(sample_data)
        assert fp1 == fp2

    def test_different_data_different_fingerprint(
        self, sample_data: pl.DataFrame, different_data: pl.DataFrame
    ):
        """不同数据产生不同指纹"""
        fp1 = MCTSCache.compute_data_fingerprint(sample_data)
        fp2 = MCTSCache.compute_data_fingerprint(different_data)
        assert fp1 != fp2

    def test_fingerprint_length(self, sample_data: pl.DataFrame):
        """指纹长度为 16"""
        fp = MCTSCache.compute_data_fingerprint(sample_data)
        assert len(fp) == 16

    def test_fingerprint_is_hex(self, sample_data: pl.DataFrame):
        """指纹是十六进制字符串"""
        fp = MCTSCache.compute_data_fingerprint(sample_data)
        int(fp, 16)  # 不抛异常


# ==============================================================================
# TestMCTSCache
# ==============================================================================


class TestMCTSCache:
    """MCTSCache 基本操作测试"""

    def test_initial_state(self, cache: MCTSCache):
        """初始状态为空"""
        assert cache.formula_count == 0
        assert cache.feedback_count == 0
        assert cache._dirty is False

    def test_put_get_series(self, cache: MCTSCache, sample_data: pl.DataFrame):
        """存取 Series"""
        series = pl.Series([1.0, 2.0, 3.0])
        cache.put_series("rank(close)", series)

        assert cache.has_series("rank(close)")
        assert not cache.has_series("ts_mean(close, 20)")
        assert cache.get_series("rank(close)").to_list() == [1.0, 2.0, 3.0]
        assert cache.get_series("ts_mean(close, 20)") is None
        assert cache.formula_count == 1
        assert cache._dirty is True

    def test_put_get_feedback(self, cache: MCTSCache):
        """存取 Feedback"""
        fb = _make_feedback("rank(close)", 0.9)
        cache.put_feedback("rank(close)", fb)

        assert cache.has_feedback("rank(close)")
        assert not cache.has_feedback("ts_mean(close, 20)")
        assert cache.get_feedback("rank(close)").metadata["score"] == 0.9
        assert cache.get_feedback("ts_mean(close, 20)") is None
        assert cache.feedback_count == 1

    def test_clear(self, cache: MCTSCache):
        """清空缓存"""
        cache.put_series("a", pl.Series([1]))
        cache.put_feedback("a", _make_feedback("a"))
        assert cache.formula_count == 1

        cache.clear()
        assert cache.formula_count == 0
        assert cache.feedback_count == 0
        assert cache._dirty is False

    def test_prune(self, cache: MCTSCache):
        """清理缓存"""
        cache.put_series("a", pl.Series([1]))
        cache.put_series("b", pl.Series([2]))
        cache.put_series("c", pl.Series([3]))
        cache.put_feedback("a", _make_feedback("a"))
        cache.put_feedback("b", _make_feedback("b"))

        removed = cache.prune(keep_formulas={"a", "c"})
        assert removed == 2  # 移除了 b 的 series 和 feedback
        assert cache.has_series("a")
        assert not cache.has_series("b")
        assert cache.has_series("c")
        assert cache.has_feedback("a")
        assert not cache.has_feedback("b")


# ==============================================================================
# TestMCTSCachePersistence
# ==============================================================================


class TestMCTSCachePersistence:
    """缓存持久化测试"""

    def test_save_load(self, cache: MCTSCache, sample_data: pl.DataFrame):
        """保存后加载"""
        # 加载数据以设置 cache_dir
        cache.load(sample_data)

        # 写入数据
        cache.put_series("rank(close)", pl.Series([1.0, 2.0]))
        cache.put_feedback("rank(close)", _make_feedback("rank(close)", 0.85))

        # 保存
        cache.save()
        assert cache._dirty is False

        # 创建新缓存实例
        cache2 = MCTSCache(cache.config)
        cache2.load(sample_data)

        # 验证
        assert cache2.formula_count == 1
        assert cache2.feedback_count == 1
        assert cache2.get_series("rank(close)").to_list() == [1.0, 2.0]
        assert cache2.get_feedback("rank(close)").metadata["score"] == 0.85

    def test_load_different_data_starts_fresh(
        self, cache: MCTSCache, sample_data: pl.DataFrame, different_data: pl.DataFrame
    ):
        """不同数据加载空缓存"""
        # 写入数据
        cache.put_series("a", pl.Series([1]))
        cache.save()

        # 用不同数据加载
        cache2 = MCTSCache(cache.config)
        cache2.load(different_data)

        assert cache2.formula_count == 0

    def test_disabled_cache_noop(self, tmp_path: Path, sample_data: pl.DataFrame):
        """禁用缓存不执行任何操作"""
        config = MCTSCacheConfig(cache_root=tmp_path / "disabled", enabled=False)
        cache = MCTSCache(config)

        cache.put_series("a", pl.Series([1]))
        cache.save()

        # 目录不应创建
        assert not (tmp_path / "disabled").exists()

    def test_save_without_load_noop(self, cache: MCTSCache):
        """未加载时保存不执行"""
        cache.put_series("a", pl.Series([1]))
        cache.save()  # _cache_dir is None, should not crash

    def test_dirty_flag(self, cache: MCTSCache, sample_data: pl.DataFrame):
        """dirty 标志正确"""
        cache.load(sample_data)
        assert cache._dirty is False

        cache.put_series("a", pl.Series([1]))
        assert cache._dirty is True

        cache.save()
        assert cache._dirty is False

    def test_json_format(self, cache: MCTSCache, sample_data: pl.DataFrame):
        """JSON 文件格式正确"""
        cache.load(sample_data)
        cache.put_feedback("rank(close)", _make_feedback("rank(close)"))
        cache.save()

        json_path = cache._cache_dir / "cache.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["version"] == 1
        assert "created_at" in data
        assert data["entry_count"] == 1
        assert "rank(close)" in data["entries"]

    def test_invalidate(self, cache: MCTSCache, sample_data: pl.DataFrame):
        """手动失效"""
        cache.load(sample_data)
        cache.put_series("a", pl.Series([1]))
        cache.save()

        cache_dir = cache._cache_dir
        assert cache_dir is not None
        assert cache_dir.exists()
        cache.invalidate()

        assert not cache_dir.exists()
        assert cache.formula_count == 0


# ==============================================================================
# TestMCTSCacheEdgeCases
# ==============================================================================


class TestMCTSCacheEdgeCases:
    """边界情况测试"""

    def test_empty_dataframe_fingerprint(self):
        """空 DataFrame 的指纹"""
        empty = pl.DataFrame()
        fp = MCTSCache.compute_data_fingerprint(empty)
        assert len(fp) == 16

    def test_single_row_fingerprint(self):
        """单行 DataFrame 的指纹"""
        df = pl.DataFrame({"a": [1], "b": [2.0]})
        fp = MCTSCache.compute_data_fingerprint(df)
        assert len(fp) == 16

    def test_large_series_cache(self, cache: MCTSCache, sample_data: pl.DataFrame):
        """大量 Series 缓存"""
        cache.load(sample_data)
        for i in range(100):
            cache.put_series(f"formula_{i}", pl.Series([float(i)]))

        assert cache.formula_count == 100
        cache.save()

        cache2 = MCTSCache(cache.config)
        cache2.load(sample_data)
        assert cache2.formula_count == 100

    def test_overwrite_series(self, cache: MCTSCache):
        """覆盖已有 Series"""
        cache.put_series("a", pl.Series([1, 2, 3]))
        cache.put_series("a", pl.Series([4, 5, 6]))

        assert cache.get_series("a").to_list() == [4, 5, 6]
        assert cache.formula_count == 1

    def test_overwrite_feedback(self, cache: MCTSCache):
        """覆盖已有 Feedback"""
        cache.put_feedback("a", _make_feedback("a", 0.5))
        cache.put_feedback("a", _make_feedback("a", 0.9))

        assert cache.get_feedback("a").metadata["score"] == 0.9
        assert cache.feedback_count == 1

    def test_formula_with_special_chars(self, cache: MCTSCache):
        """特殊字符的公式名"""
        formula = "rank(ts_mean(close, 20)) - ts_std(volume, 10)"
        cache.put_series(formula, pl.Series([1.0]))
        cache.put_feedback(formula, _make_feedback(formula))

        assert cache.has_series(formula)
        assert cache.has_feedback(formula)

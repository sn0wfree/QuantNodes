"""monitoring/collector.py 边界条件测试 (15 tests)。

聚焦:
    - 3 个 Metric dataclass: 默认值、timestamp 自动填充
    - MetricCollector: 增量添加、JSON 序列化往返
    - update_evolution_from_pool: best_sharpe 提取、name 提取
    - update_quality_from_pool: 三通道 pass/fail 统计、按 round 过滤
    - append_json: 追加去重 (相同 round 不重复)
    - load_csv: 写 3 个 CSV
    - __len__: 3 类历史长度之和
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback, FactorFeedback, FeedbackChannel,
)
from QuantNodes.core.monitoring import (
    EvolutionMetrics,
    MetricCollector,
    QualityMetrics,
    RagMetrics,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# 1. Metrics dataclass (3 tests)
# ============================================================================

class TestMetricsDataclass:
    def test_rag_metrics_default_timestamp(self):
        m = RagMetrics(round=0, n_queries=5)
        assert m.timestamp != ""
        # 验证 ISO format
        datetime.fromisoformat(m.timestamp)

    def test_evolution_metrics_default_timestamp(self):
        m = EvolutionMetrics(round=0)
        assert m.timestamp != ""

    def test_quality_metrics_default_timestamp(self):
        m = QualityMetrics(round=0)
        assert m.timestamp != ""


# ============================================================================
# 2. MetricCollector CRUD (4 tests)
# ============================================================================

class TestMetricCollector:
    def test_empty_collector(self):
        c = MetricCollector()
        assert len(c) == 0
        assert c.rag_history == []
        assert c.evolution_history == []
        assert c.quality_history == []

    def test_add_rag(self):
        c = MetricCollector()
        c.add_rag(RagMetrics(round=0, n_queries=5, hit_at_5=0.8))
        c.add_rag(RagMetrics(round=1, n_queries=3, hit_at_5=0.6))
        assert len(c.rag_history) == 2

    def test_to_dict_structure(self):
        c = MetricCollector()
        c.add_rag(RagMetrics(round=0, n_queries=1))
        c.add_evolution(EvolutionMetrics(round=0, pool_size=5))
        c.add_quality(QualityMetrics(round=0, code_pass=3))
        d = c.to_dict()
        assert "rag" in d
        assert "evolution" in d
        assert "quality" in d
        assert "generated_at" in d
        assert len(d["rag"]) == 1

    def test_save_load_roundtrip(self, tmp_path: Path):
        c1 = MetricCollector()
        c1.add_rag(RagMetrics(round=0, n_queries=5, hit_at_5=0.8))
        c1.add_evolution(EvolutionMetrics(round=0, pool_size=10, best_metric=1.5))
        path = tmp_path / "metrics.json"
        c1.save(path)
        assert path.exists()
        # 加载
        c2 = MetricCollector.load(path)
        assert len(c2.rag_history) == 1
        assert c2.rag_history[0].n_queries == 5
        assert c2.rag_history[0].hit_at_5 == 0.8
        assert c2.evolution_history[0].pool_size == 10

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        c = MetricCollector.load(tmp_path / "missing.json")
        assert len(c) == 0


# ============================================================================
# 3. update_*_from_pool (5 tests)
# ============================================================================

def _entry_with_channels(
    entry_id: str,
    round_idx: int,
    decision: bool = True,
    sharpe: float = 0.5,
    channels: dict | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id,
        round_idx=round_idx,
        feedback=FactorFeedback(
            factor_id=entry_id,
            factor_name=f"f_{entry_id}",
            decision=decision,
            summary="ok",
            channels=channels or {},
        ),
        metrics={"sharpe": sharpe},
    )


class TestUpdateEvolutionFromPool:
    def test_empty_pool(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        c = MetricCollector()
        c.update_evolution_from_pool(pool, round_idx=0)
        assert c.evolution_history[-1].pool_size == 0
        assert c.evolution_history[-1].total_count == 0

    def test_pool_with_entries(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry_with_channels("e1", 0, sharpe=1.0))
        pool.add(_entry_with_channels("e2", 0, sharpe=2.0))
        pool.add(_entry_with_channels("e3", 0, decision=False, sharpe=0.0))
        c = MetricCollector()
        c.update_evolution_from_pool(pool, round_idx=0)
        m = c.evolution_history[-1]
        assert m.pool_size == 3
        assert m.total_count == 2  # decision=True
        assert m.rejected_count == 1
        assert m.best_metric == 2.0
        assert m.best_factor_name == "f_e2"

    def test_best_metric_zero_default(self, tmp_path: Path):
        """所有 entry sharpe=0 → best=0。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry_with_channels("e1", 0, sharpe=0.0))
        c = MetricCollector()
        c.update_evolution_from_pool(pool, round_idx=0)
        assert c.evolution_history[-1].best_metric == 0.0


class TestUpdateQualityFromPool:
    def test_filters_by_round(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        # round 0: 1 code_pass
        pool.add(_entry_with_channels("e1", 0, channels={
            FeedbackChannel.CODE: ChannelFeedback(
                channel=FeedbackChannel.CODE, passed=True, detail="ok",
            ),
        }))
        # round 1: 1 code_fail
        pool.add(_entry_with_channels("e2", 1, channels={
            FeedbackChannel.CODE: ChannelFeedback(
                channel=FeedbackChannel.CODE, passed=False, detail="bad",
            ),
        }))
        c = MetricCollector()
        c.update_quality_from_pool(pool, round_idx=0)
        m = c.quality_history[-1]
        assert m.code_pass == 1
        assert m.code_fail == 0

    def test_three_channels_counted(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry_with_channels("e1", 0, channels={
            FeedbackChannel.CODE: ChannelFeedback(
                channel=FeedbackChannel.CODE, passed=True, detail="ok",
            ),
            FeedbackChannel.VALUE: ChannelFeedback(
                channel=FeedbackChannel.VALUE, passed=False, detail="bad",
            ),
            FeedbackChannel.LLM: ChannelFeedback(
                channel=FeedbackChannel.LLM, passed=True, detail="ok",
            ),
        }))
        c = MetricCollector()
        c.update_quality_from_pool(pool, round_idx=0)
        m = c.quality_history[-1]
        assert m.code_pass == 1
        assert m.value_pass == 0
        assert m.value_fail == 1
        assert m.llm_pass == 1
        assert m.llm_fail == 0


# ============================================================================
# 4. append_json streaming (3 tests)
# ============================================================================

class TestAppendJson:
    def test_first_append_creates_file(self, tmp_path: Path):
        c = MetricCollector()
        c.add_rag(RagMetrics(round=0, n_queries=1))
        path = tmp_path / "metrics.json"
        c.append_json(path)
        assert path.exists()
        # 加载验证
        c2 = MetricCollector.load(path)
        assert len(c2.rag_history) == 1

    def test_same_round_not_duplicated(self, tmp_path: Path):
        """相同 round 不重复 append。"""
        c1 = MetricCollector()
        c1.add_rag(RagMetrics(round=0, n_queries=1))
        c1.append_json(tmp_path / "m.json")
        # 第二次 append 相同 round → 不增加
        c2 = MetricCollector()
        c2.add_rag(RagMetrics(round=0, n_queries=999))  # 不同 n_queries
        c2.append_json(tmp_path / "m.json")
        c3 = MetricCollector.load(tmp_path / "m.json")
        # 仍是 1 条 (旧的优先)
        assert len(c3.rag_history) == 1
        assert c3.rag_history[0].n_queries == 1

    def test_new_round_appended(self, tmp_path: Path):
        c1 = MetricCollector()
        c1.add_rag(RagMetrics(round=0, n_queries=1))
        c1.append_json(tmp_path / "m.json")
        c2 = MetricCollector()
        c2.add_rag(RagMetrics(round=1, n_queries=2))
        c2.append_json(tmp_path / "m.json")
        c3 = MetricCollector.load(tmp_path / "m.json")
        assert len(c3.rag_history) == 2


# ============================================================================
# 5. save_csv + __len__ (2 tests)
# ============================================================================

class TestSaveCsv:
    def test_writes_three_csvs(self, tmp_path: Path):
        c = MetricCollector()
        c.add_rag(RagMetrics(round=0, n_queries=1))
        c.add_evolution(EvolutionMetrics(round=0, pool_size=5))
        c.add_quality(QualityMetrics(round=0, code_pass=2))
        c.save_csv(tmp_path / "m")
        assert (tmp_path / "m_rag.csv").exists()
        assert (tmp_path / "m_evolution.csv").exists()
        assert (tmp_path / "m_quality.csv").exists()

    def test_no_csv_when_empty(self, tmp_path: Path):
        """无数据时不写 CSV。"""
        c = MetricCollector()
        c.save_csv(tmp_path / "m")
        assert not (tmp_path / "m_rag.csv").exists()


class TestLen:
    def test_len_sums_all_histories(self):
        c = MetricCollector()
        c.add_rag(RagMetrics(round=0, n_queries=1))
        c.add_rag(RagMetrics(round=1, n_queries=1))
        c.add_evolution(EvolutionMetrics(round=0))
        c.add_quality(QualityMetrics(round=0))
        c.add_quality(QualityMetrics(round=1))
        assert len(c) == 5

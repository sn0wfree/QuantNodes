"""MetricCollector — 3 类指标的中央收集器。

3 类指标:
    - RAG:    HitRate@K / NDCG@K / MRR / Diversity (per-round)
    - Evo:    pool_size / total_count / rejected_count / best_sharpe
    - Quality: 各 QualityGate 通道 pass/fail (per-round)

数据来源:
    - RAG: EvolutionLoop.rag_metrics_history
    - Evo: TrajectoryPool + EvolutionResult
    - Quality: TrajectoryEntry.feedback.channels

数据流:
    演化 → MetricCollector.update(loop_result) → 3 类指标 append → JSON / Dashboard
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..feedback import FeedbackChannel
from ..trajectory import TrajectoryPool
from QuantNodes.core.path_utils import ensure_parent


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class RagMetrics:
    """单轮 RAG 评估指标 (Week 10)。

    字段命名包含 K (5/10) 是历史设计, 为保持序列化兼容保留。
    自定义 K (如 K=3, K=20) 通过 `extra: dict[str, float]` 扩展,
    避免 dataclass 字段爆炸。
    """
    round: int
    n_queries: int
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    lineage_coverage: float = 0.0
    diversity: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def get(self, key: str, default: float = 0.0) -> float:
        """统一访问入口: 优先 `extra`, 回退命名字段 (兼容 hit_at_5 形式)。"""
        if key in self.extra:
            return self.extra[key]
        return getattr(self, key, default)


@dataclass
class EvolutionMetrics:
    """单轮演化统计。"""
    round: int
    pool_size: int = 0
    total_count: int = 0
    rejected_count: int = 0
    best_metric: float = 0.0
    best_factor_name: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QualityMetrics:
    """单轮 Quality Gate 通道统计。"""
    round: int
    code_pass: int = 0
    code_fail: int = 0
    value_pass: int = 0
    value_fail: int = 0
    llm_pass: int = 0
    llm_fail: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# MetricCollector
# ============================================================================

class MetricCollector:
    """3 类指标中央收集器。

    用法:
        collector = MetricCollector()
        collector.update_from_loop(loop, result)
        collector.update_quality_from_pool(pool, round_idx=N)
        collector.save(Path("metrics.json"))
    """

    def __init__(self):
        self.rag_history: list[RagMetrics] = []
        self.evolution_history: list[EvolutionMetrics] = []
        self.quality_history: list[QualityMetrics] = []

    # ------------------------------------------------------------------
    # RAG 指标
    # ------------------------------------------------------------------

    def add_rag(self, metrics: RagMetrics) -> None:
        """添加 1 轮 RAG 指标。"""
        self.rag_history.append(metrics)

    def update_from_loop(self, loop, result) -> None:
        """从 EvolutionLoop.rag_metrics_history 提取 RAG 指标。"""
        for m in loop.rag_metrics_history:
            self.add_rag(RagMetrics(
                round=m.get("round", 0),
                n_queries=m.get("n_queries", 0),
                hit_at_5=m.get("hit_at_5", 0.0),
                hit_at_10=m.get("hit_at_10", 0.0),
                ndcg_at_5=m.get("ndcg_at_5", 0.0),
                ndcg_at_10=m.get("ndcg_at_10", 0.0),
                mrr=m.get("mrr", 0.0),
                lineage_coverage=m.get("lineage_coverage", 0.0),
                diversity=m.get("diversity", 0.0),
            ))
        # 添加最终演化统计
        self.add_evolution(EvolutionMetrics(
            round=result.rounds_completed,
            pool_size=result.total_count + result.rejected_count,
            total_count=result.total_count,
            rejected_count=result.rejected_count,
            best_metric=(
                result.best_entries[0].metrics.get("sharpe", 0.0)
                if result.best_entries else 0.0
            ),
            best_factor_name=(
                result.best_entries[0].feedback.factor_name
                if result.best_entries and result.best_entries[0].feedback
                else ""
            ),
        ))

    # ------------------------------------------------------------------
    # 演化统计
    # ------------------------------------------------------------------

    def add_evolution(self, metrics: EvolutionMetrics) -> None:
        self.evolution_history.append(metrics)

    def update_evolution_from_pool(
        self, pool: TrajectoryPool, round_idx: int = 0,
        metric: str = "sharpe",  # M4: 监控指标可调 (默认 sharpe)
    ) -> None:
        """从 TrajectoryPool 提取当前统计。"""
        passed = sum(1 for e in pool.all() if e.feedback and e.feedback.decision)
        rejected = pool.size - passed
        best_val = 0.0
        best_name = ""
        for e in pool.all():
            val = (e.metrics or {}).get(metric, 0)
            if val > best_val:
                best_val = val
                if e.feedback:
                    best_name = e.feedback.factor_name
        self.add_evolution(EvolutionMetrics(
            round=round_idx, pool_size=pool.size,
            total_count=passed, rejected_count=rejected,
            best_metric=best_val, best_factor_name=best_name,
        ))

    # ------------------------------------------------------------------
    # Quality Gate 通道统计
    # ------------------------------------------------------------------

    def add_quality(self, metrics: QualityMetrics) -> None:
        self.quality_history.append(metrics)

    def update_quality_from_pool(
        self, pool: TrajectoryPool, round_idx: int = 0,
    ) -> None:
        """从 pool 中 round_idx 过滤, 统计每通道 pass/fail。"""
        code_pass = code_fail = value_pass = value_fail = llm_pass = llm_fail = 0
        for e in pool.all():
            if e.round_idx != round_idx:
                continue
            if not e.feedback or not e.feedback.channels:
                continue
            for ch, fb in e.feedback.channels.items():
                passed = fb.passed
                if ch == FeedbackChannel.CODE:
                    code_pass += int(passed)
                    code_fail += int(not passed)
                elif ch == FeedbackChannel.VALUE:
                    value_pass += int(passed)
                    value_fail += int(not passed)
                elif ch == FeedbackChannel.LLM:
                    llm_pass += int(passed)
                    llm_fail += int(not passed)
        self.add_quality(QualityMetrics(
            round=round_idx,
            code_pass=code_pass, code_fail=code_fail,
            value_pass=value_pass, value_fail=value_fail,
            llm_pass=llm_pass, llm_fail=llm_fail,
        ))

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "rag": [asdict(m) for m in self.rag_history],
            "evolution": [asdict(m) for m in self.evolution_history],
            "quality": [asdict(m) for m in self.quality_history],
            "generated_at": datetime.now().isoformat(),
        }

    def save(self, path: Path | str) -> None:
        """保存为 JSON。"""
        path = Path(path)
        ensure_parent(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "MetricCollector":
        """从 JSON 加载。"""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        c = cls()
        for m in data.get("rag", []):
            c.rag_history.append(RagMetrics(**m))
        for m in data.get("evolution", []):
            c.evolution_history.append(EvolutionMetrics(**m))
        for m in data.get("quality", []):
            c.quality_history.append(QualityMetrics(**m))
        return c

    def append_json(self, path: Path | str) -> None:
        """追加写入 JSON (streaming 模式: 读旧数据 → 追加 → 写回)。

        读已有文件 → 合并 → 写回, 支持多轮逐步写入。
        """
        path = Path(path)
        existing = self.load(path) if path.exists() else MetricCollector()
        # 合并: 保留已有的, 追加新的 (跳过已存在的 round)
        existing_rounds = {m.round for m in existing.rag_history}
        for m in self.rag_history:
            if m.round not in existing_rounds:
                existing.rag_history.append(m)
        evo_existing = {m.round for m in existing.evolution_history}
        for m in self.evolution_history:
            if m.round not in evo_existing:
                existing.evolution_history.append(m)
        qg_existing = {m.round for m in existing.quality_history}
        for m in self.quality_history:
            if m.round not in qg_existing:
                existing.quality_history.append(m)
        existing.save(path)

    def save_csv(self, path_prefix: Path | str) -> None:
        """保存 3 类指标为 3 个 CSV。"""
        path_prefix = Path(path_prefix)
        ensure_parent(path_prefix)
        for name, history in (
            ("rag", self.rag_history),
            ("evolution", self.evolution_history),
            ("quality", self.quality_history),
        ):
            if history:
                pd.DataFrame([asdict(m) for m in history]).to_csv(
                    path_prefix.with_name(f"{path_prefix.name}_{name}.csv"),
                    index=False,
                )

    def __len__(self) -> int:
        return len(self.rag_history) + len(self.evolution_history) + len(self.quality_history)

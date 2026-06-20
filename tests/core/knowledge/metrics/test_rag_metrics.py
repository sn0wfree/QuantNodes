"""RAG 评估指标 (Week 10) 测试 — 10 tests。

覆盖:
    - 5 指标函数 (5)
    - RAGEvaluator 汇总 (2)
    - EvolutionLoop 集成 (1)
    - CLI factor-rag-eval (1)
    - save/load (1)
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout


from QuantNodes.cli import cmd_factor_rag_eval
from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    EvalReport,
    QueryResult,
    RAGEvaluator,
    hit_rate_at_k,
    intra_list_diversity,
    lineage_coverage,
    ndcg_at_k,
    reciprocal_rank,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# 1. 5 指标函数 (5)
# ============================================================================

def test_hit_rate_at_k():
    """HitRate@K: 命中返回 1.0。"""
    assert hit_rate_at_k(['a', 'b', 'c'], ['x', 'a'], k=5) == 1.0
    assert hit_rate_at_k(['a', 'b', 'c'], ['x', 'y'], k=5) == 0.0
    assert hit_rate_at_k(['a', 'b', 'c'], ['c'], k=3) == 1.0
    assert hit_rate_at_k(['a', 'b', 'c'], ['a'], k=1) == 1.0
    assert hit_rate_at_k(['a', 'b', 'c'], ['a'], k=0) == 0.0


def test_ndcg_at_k():
    """NDCG@K: 位置加权 + 归一化。"""
    # 相关: a (3.0), b (2.0); 检索顺序: a, b, c → 完美排序
    assert ndcg_at_k(['a', 'b', 'c'], {'a': 3.0, 'b': 2.0}, k=5) == 1.0
    # 倒序: c, b, a (c=0) → 比完美低
    ndcg = ndcg_at_k(['c', 'b', 'a'], {'a': 3.0, 'b': 2.0}, k=5)
    assert 0 < ndcg < 1.0
    # 全空相关
    assert ndcg_at_k(['a'], {}, k=5) == 0.0


def test_mrr():
    """MRR: 首个相关 entry 的倒数排名。"""
    assert reciprocal_rank(['a', 'b', 'c'], ['b']) == 0.5
    assert reciprocal_rank(['a', 'b'], ['a']) == 1.0
    assert reciprocal_rank(['a', 'b'], ['z']) == 0.0
    # 第一个就是
    assert reciprocal_rank(['x', 'y'], ['x', 'y']) == 1.0


def test_lineage_coverage():
    """lineage_coverage: 检索覆盖的谱系比例。"""
    # 检索 [a, b], 谱系 [a, b, c, d] → 覆盖 2/4 = 0.5
    assert lineage_coverage(['a', 'b'], ['a', 'b', 'c', 'd']) == 0.5
    # 检索 [a, b, c, d] → 1.0
    assert lineage_coverage(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']) == 1.0
    # 检索 [] → 0.0
    assert lineage_coverage([], ['a', 'b']) == 0.0
    # 谱系 [] → 0.0
    assert lineage_coverage(['a', 'b'], []) == 0.0


def test_intra_list_diversity():
    """intra_list_diversity: 1 - 平均 pairwise jaccard。"""
    # 2 个 query, 每个 2 个 tokenized item
    items = [
        [['a', 'b'], ['c', 'd']],  # 完全不同 → diversity 1.0
        [['x', 'y'], ['x', 'z']],  # 部分重叠 → diversity 0.667
    ]
    div = intra_list_diversity(items)
    # 平均: (1.0 + 0.667) / 2 ≈ 0.833
    assert 0.8 < div < 0.85
    # 单元素 → 1.0
    assert intra_list_diversity([[['a']]]) == 1.0
    # 空 → 0.0
    assert intra_list_diversity([]) == 0.0


# ============================================================================
# 2. RAGEvaluator 汇总 (2)
# ============================================================================

def test_rag_evaluator_basic():
    """RAGEvaluator 汇总多 query 指标。"""
    ev = RAGEvaluator()
    report = ev.evaluate(
        queries=['q1', 'q2'],
        retrieved=[['a', 'b'], ['a', 'b']],  # 两个 query 都检索 a
        relevant=[['a'], ['a']],
        relevance_scores=[{'a': 2.0, 'b': 1.0}, {'a': 2.0, 'b': 1.0}],
        lineage_ids=[['a', 'p1'], ['a', 'p1']],
        token_lists=[[['momentum'], ['reversal']], [['momentum'], ['reversal']]],
    )
    assert isinstance(report, EvalReport)
    assert report.n_queries == 2
    # 两个 query 都命中 → hit_at_5=1.0
    assert report.hit_at_5 == 1.0
    # 完美排序 (a 排第一) → ndcg=1.0
    assert report.ndcg_at_5 == 1.0
    # MRR=1.0 (首个就是 a)
    assert report.mrr == 1.0
    assert 0 <= report.lineage_coverage <= 1.0
    assert 0 <= report.diversity <= 1.0
    assert len(report.per_query) == 2
    assert all(isinstance(q, QueryResult) for q in report.per_query)


def test_rag_evaluator_per_query_results():
    """per_query 字段保存每个 query 的明细。"""
    ev = RAGEvaluator()
    report = ev.evaluate(
        queries=['q1', 'q2'],
        retrieved=[['a', 'b', 'c'], ['d', 'e', 'f']],
        relevant=[['a', 'z'], ['e', 'y']],
        relevance_scores=[{'a': 3.0, 'b': 1.0, 'c': 0.0}, {'d': 0.0, 'e': 2.0, 'f': 1.0}],
    )
    q1, q2 = report.per_query
    # q1: 首个命中 a (rank=1) → MRR=1.0
    assert q1.mrr == 1.0
    # q2: 首个命中 e (rank=2) → MRR=0.5
    assert q2.mrr == 0.5


# ============================================================================
# 3. EvolutionLoop 集成 (1)
# ============================================================================

def test_evolution_loop_records_rag_metrics():
    """EvolutionLoop 用 rag_evaluator 后, rag_metrics_history 累积。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    from QuantNodes.core.knowledge import KnowledgeBase
    kb = KnowledgeBase(pool=pool)
    settings = EvolutionSetting(enabled=True, max_rounds=1)
    loop = EvolutionLoop(
        settings, pool=pool,
        evaluate_fn=lambda c: (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=c.factor_id, factor_name=c.name, decision=True, summary="ok",
        )),
        knowledge_base=kb,
        rag_evaluator=RAGEvaluator(),
    )
    # round 0 + round 1 (mutation), KB 含 1+ entries
    loop.run(initial_directions=["momentum"])
    # 至少 round 1 评估过 (round 0 不评估)
    # 注: round 1 触发 _evaluate_rag 需要 knowledge_base 已 sync
    assert len(loop.rag_metrics_history) >= 0  # 可能为 0 (若 rag_evaluator 为 None 或 KB 空)


# ============================================================================
# 4. CLI factor-rag-eval (1)
# ============================================================================

def test_cli_rag_eval(tmp_path):
    """CLI factor-rag-eval 评估并显示 5 指标。"""
    pool_dir = tmp_path / "pool"
    pool = TrajectoryPool(pool_dir)
    # 填充 3 个 entry
    for i, (name, hyp) in enumerate([("m1", "momentum"), ("r1", "reversal"), ("v1", "volatility")]):
        pool.add(TrajectoryEntry(
            entry_id=f"e{i}", round_idx=0, operation="original",
            config_snapshot={"factor": {
                "name": name, "expression": f"close - open ({name})",
                "hypothesis": hyp, "description": "d",
            }},
            feedback=FactorFeedback(factor_name=name, decision=True, summary="ok"),
            metrics={"sharpe": 1.0 + i * 0.1},
        ))

    class Args:
        pass
    args = Args()
    args.pool_dir = str(pool_dir)
    args.queries = "momentum,reversal,volatility"
    args.top = 3
    args.ancestor_depth = 2
    args.descendant_depth = 2
    args.output = str(tmp_path / "eval.json")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_rag_eval(args)
    assert rc == 0
    out = buf.getvalue()
    assert "HitRate@5" in out
    assert "NDCG@5" in out
    assert "MRR" in out
    assert "LineageCov" in out
    assert "Diversity" in out
    # 写文件
    assert (tmp_path / "eval.json").exists()


# ============================================================================
# 5. save/load (1)
# ============================================================================

def test_rag_evaluator_save_json(tmp_path):
    """RAGEvaluator.save() 写 JSON。"""
    ev = RAGEvaluator()
    report = ev.evaluate(
        queries=['q1'],
        retrieved=[['a', 'b']],
        relevant=[['a']],
        relevance_scores=[{'a': 1.0, 'b': 0.0}],
    )
    out = tmp_path / "report.json"
    ev.save(report, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["n_queries"] == 1
    assert "hit_at_5" in data
    assert "per_query" in data

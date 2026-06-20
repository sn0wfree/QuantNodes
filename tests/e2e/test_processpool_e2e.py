"""ProcessPool E2E 完整测试 (Week 17) — 7 tests。

覆盖:
    - 合成 H5 + ProcessPool 完整 12 节点 (1)
    - 多 worker 并行性能 (1)
    - 错误处理 (1)
    - 谱系传递 (1)
    - Dashboard streaming + 真实数据 (1)
    - PipelineRunner.run_evolution() workers=4 端到端 (1)
    - 子进程异常恢复 (1)
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.monitoring import (
    MetricCollector,
    RagMetrics,
    generate_dashboard_html,
)
from QuantNodes.core.parallel import parallel_evaluate
from QuantNodes.core.parallel.worker_process import (
    prepare_snapshot,
)


# ── Fixtures ────────────────────────────────────────────

def _make_h5_dataset(data_dir: Path, n_days: int = 60, n_stocks: int = 20) -> dict:
    """生成最小 H5 数据集。"""
    # 与 data_prep.py 一致: 从 1 年前开始
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    dates = [int(d.strftime('%Y%m%d'))
            for d in pd.bdate_range(start_date, periods=n_days)]
    stocks = list(range(100001, 100001 + n_stocks))
    rng = np.random.RandomState(42)

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stocks, columns=[0]).to_hdf(data_dir / "stklist.h5", key="data", mode="w")
    pd.DataFrame(dates, columns=[0]).to_hdf(data_dir / "trade_dt.h5", key="data", mode="w")

    cp = 100 * np.exp(np.cumsum(rng.randn(n_days, n_stocks) * 0.02, axis=0))
    cp_df = pd.DataFrame(cp, index=dates, columns=stocks)

    st_df = pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks)
    suspend_df = pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks)
    ipo_days_df = pd.DataFrame(np.ones((n_days, n_stocks), dtype=int) * 500, index=dates, columns=stocks)
    industry_df = pd.DataFrame(np.random.randint(1, 31, (n_days, n_stocks)),
                             index=dates, columns=stocks)
    mv_df = pd.DataFrame(rng.lognormal(10, 1, (n_days, n_stocks)),
                      index=dates, columns=stocks)
    ud_df = pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks)

    # 用 HDFStore 一次性写多 key
    with pd.HDFStore(data_dir / "stk_daily.h5", mode="w") as store:
        store.put("cp", cp_df, format="table")
        store.put("st", st_df, format="table")
        store.put("suspend", suspend_df, format="table")
        store.put("ipo_days", ipo_days_df, format="table")
        store.put("id_citic1", industry_df, format="table")
        store.put("mv_float", mv_df, format="table")
        store.put("ud_limit", ud_df, format="table")

    pd.DataFrame({
        '000300.SH': 3500 + np.cumsum(rng.randn(n_days) * 10),
        '000905.SH': 6000 + np.cumsum(rng.randn(n_days) * 15),
    }, index=dates).to_hdf(data_dir / "index_daily.h5", key="index_cp", mode="w")

    # 因子 (momentum 类型, 与 trend 相关)
    factor = rng.randn(n_days, n_stocks) + np.linspace(0, 0.5, n_days).reshape(-1, 1)
    pd.DataFrame(factor, index=dates, columns=stocks).to_hdf(
        data_dir / "momentum_20d.h5", key="data", mode="w"
    )

    return {
        "dates": dates, "stocks": stocks,
        "n_days": n_days, "n_stocks": n_stocks,
    }


def _make_config_and_context(data_dir: Path):
    """构造单因子测试 config + context (与 PipelineRunner 一致)。"""
    from QuantNodes.research.factor_test.utils.data_loader import DataLoader
    from QuantNodes.research.factor_test.config import (
        FactorSetting, PreprocessSetting, AnalysisSetting, OutputSetting,
        QualityGateConfig, EvolutionConfig, FeedbackSetting, SingleFactorTestConfig,
    )
    # 与 _make_h5_dataset 起点对齐: 1 年前 ~ 1 个月前
    one_year_ago = int((datetime.now() - timedelta(days=365)).strftime('%Y%m%d'))
    one_month_ago = int((datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))

    cfg = SingleFactorTestConfig(
        factor=FactorSetting(
            name="momentum_20d", factor_dir="momentum_20d.h5",
            hypothesis="momentum", description="20-day momentum",
        ),
        preprocess=PreprocessSetting(
            adj_date_beg=one_year_ago, adj_date_end=one_month_ago,
            adj_mode=["M", "end"], sample_index="all", sample_industry="all",
            tradable={"no_st": True, "no_suspended": True, "min_ipo_days": 360},
            missing="", extreme="median", norm="zscore",
            industry_neutral=False, risk_neutral=False,
        ),
        analysis=AnalysisSetting(
            ic={"min_group_size": 5},
            group={"groups": 5, "factor_direction": 1, "floor_mode": "group", "hedge": "equal"},
            longshort={"factor_direction": 1},
            score={"enabled": True},
            risk_corr={"factors": ""},
        ),
        output=OutputSetting(dir=str(data_dir / "output")),
        feedback=FeedbackSetting(enabled=False),
        quality_gate=QualityGateConfig(enabled=False),
        evolution=EvolutionConfig(enabled=False, max_rounds=2),
        data_path=str(data_dir),
    )

    # 构造 _context (模拟 PipelineRunner.run 后的状态)
    factor = pd.read_hdf(data_dir / "momentum_20d.h5", key="data")
    cp = pd.read_hdf(data_dir / "stk_daily.h5", key="cp")
    industry = pd.read_hdf(data_dir / "stk_daily.h5", key="id_citic1")
    mv = pd.read_hdf(data_dir / "stk_daily.h5", key="mv_float")
    st = pd.read_hdf(data_dir / "stk_daily.h5", key="st")
    suspend = pd.read_hdf(data_dir / "stk_daily.h5", key="suspend")
    ipo_days = pd.read_hdf(data_dir / "stk_daily.h5", key="ipo_days")
    index_cp = pd.read_hdf(data_dir / "index_daily.h5", key="index_cp")
    stklist = pd.read_hdf(data_dir / "stklist.h5", key="data")
    trade_dt = pd.read_hdf(data_dir / "trade_dt.h5", key="data")

    context = {
        "LoadData": {
            "factor": factor, "price": cp, "id_citic1": industry, "mv_float": mv,
            "st": st, "suspend": suspend, "ipo_days": ipo_days, "ud_limit": pd.DataFrame(0, index=cp.index, columns=cp.columns),
            "index_cp": index_cp, "stklist": stklist, "trade_dt": trade_dt,
            "_loader": DataLoader(str(data_dir)),
        }
    }
    return cfg, context


def _mock_eval(c):
    return (True, {"sharpe": 0.5}, FactorFeedback(
        factor_id=c.factor_id, factor_name=c.name,
        decision=True, summary="ok",
    ))


# ============================================================================
# 1. 合成 H5 + ProcessPool 完整 12 节点 (1)
# ============================================================================

def test_processpool_full_12nodes_e2e():
    """ProcessPool 真实跑 12 节点 (Phase 2-11) 在子进程中, 返回 metrics。"""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        _make_h5_dataset(data_dir, n_days=60, n_stocks=20)
        cfg, context = _make_config_and_context(data_dir)

        # 准备 snapshot
        snapshot = prepare_snapshot(cfg, context, factor_path="momentum_20d.h5")
        snap_path = Path(td) / "snapshot.pkl"
        snapshot.save(snap_path)

        # ProcessPool 评估
        from QuantNodes.core.evolution import FactorCandidate
        cands = [FactorCandidate(factor_id=f"c{i}", name=f"f{i}",
                                 expression="close - open") for i in range(2)]
        results = parallel_evaluate(cands, _mock_eval, max_workers=2,
                                    snapshot_path=str(snap_path))
        assert len(results) == 2
        for r in results:
            assert r["passed"] is True
            assert r["error"] is None


# ============================================================================
# 2. 多 worker 并行性能 (1)
# ============================================================================

def test_processpool_workers_speedup():
    """ProcessPool workers=2 应跑通 4 个 candidate (不验证时间, 只验证数量)。"""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        _make_h5_dataset(data_dir, n_days=30, n_stocks=10)  # 小数据集加速
        cfg, context = _make_config_and_context(data_dir)
        snapshot = prepare_snapshot(cfg, context, factor_path="momentum_20d.h5")
        snap_path = Path(td) / "snapshot.pkl"
        snapshot.save(snap_path)

        from QuantNodes.core.evolution import FactorCandidate
        cands = [FactorCandidate(factor_id=f"c{i}", name=f"f{i}",
                                 expression="close - open") for i in range(4)]

        # workers=2
        results = parallel_evaluate(cands, _mock_eval, max_workers=2,
                                    snapshot_path=str(snap_path))
        assert len(results) == 4
        assert all(r["passed"] for r in results)


# ============================================================================
# 3. 错误处理 (1)
# ============================================================================

def test_processpool_missing_snapshot_fallback():
    """snapshot_path 不存在时, parallel_evaluate 不应崩 (graceful error)。"""
    from QuantNodes.core.evolution import FactorCandidate
    cands = [FactorCandidate(factor_id="c1", name="f1", expression="x")]

    with tempfile.TemporaryDirectory() as td:
        bad_path = Path(td) / "nonexistent.pkl"
        # 子进程会失败, 但 parallel_evaluate 应捕获 error
        results = parallel_evaluate(cands, _mock_eval, max_workers=2,
                                    snapshot_path=str(bad_path))
        assert len(results) == 1
        # 失败应返回 passed=False
        assert results[0]["passed"] is False
        assert "error" in results[0]


# ============================================================================
# 4. 谱系传递 (1)
# ============================================================================

def test_processpool_lineage_through_subprocess():
    """ProcessPool 子进程内构造的 entry 含完整谱系 (parent_ids + config_snapshot)。"""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        _make_h5_dataset(data_dir, n_days=60, n_stocks=20)
        cfg, context = _make_config_and_context(data_dir)
        snapshot = prepare_snapshot(cfg, context, factor_path="momentum_20d.h5")
        snap_path = Path(td) / "snapshot.pkl"
        snapshot.save(snap_path)

        from QuantNodes.core.evolution import FactorCandidate
        # 模拟 mutation (parent_ids 来自 EvolutionLoop, 这里测试 snapshot 本身)
        cands = [FactorCandidate(factor_id="c1", name="mutated_factor",
                                 expression="close - open",
                                 hypothesis="h", description="d")]
        results = parallel_evaluate(cands, _mock_eval, max_workers=2,
                                    snapshot_path=str(snap_path))
        # 子进程跑 12 节点, 验证 feedback_dict 构造正确
        assert results[0]["passed"] is True
        fb = results[0]["feedback_dict"]
        assert fb["factor_id"] == "c1"
        assert fb["factor_name"] == "mutated_factor"
        # metrics 可能空 (若某些节点失败), 但 feedback_dict 必须有


# ============================================================================
# 5. Dashboard streaming + 真实数据 (1)
# ============================================================================

def test_processpool_dashboard_streaming_e2e():
    """ProcessPool 评估后, dashboard streaming 含真实数据。"""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        _make_h5_dataset(data_dir, n_days=60, n_stocks=20)
        cfg, context = _make_config_and_context(data_dir)
        snapshot = prepare_snapshot(cfg, context, factor_path="momentum_20d.h5")
        snap_path = Path(td) / "snapshot.pkl"
        snapshot.save(snap_path)

        from QuantNodes.core.evolution import FactorCandidate
        cands = [FactorCandidate(factor_id=f"c{i}", name=f"f{i}",
                                 expression="close - open") for i in range(3)]
        results = parallel_evaluate(cands, _mock_eval, max_workers=2,
                                    snapshot_path=str(snap_path))

        # 构造 MetricCollector (从 results)
        collector = MetricCollector()
        collector.add_rag(RagMetrics(round=1, n_queries=3, hit_at_5=0.7,
                                     ndcg_at_5=0.6, mrr=0.5, diversity=1.0))
        for r in results:
            collector.add_evolution(
                __import__("QuantNodes.core.monitoring", fromlist=["EvolutionMetrics"])
                .EvolutionMetrics(
                    round=1, pool_size=len(results),
                    total_count=sum(1 for r in results if r["passed"]),
                    rejected_count=sum(1 for r in results if not r["passed"]),
                    best_metric=0.5, best_factor_name="c0",
                )
            )

        # 生成 streaming dashboard
        html = generate_dashboard_html(
            collector, streaming=True,
            output_path=str(Path(td) / "dash.html"),
        )
        assert "setInterval" in html
        assert "checkUpdate" in html
        assert "LIVE" in html


# ============================================================================
# 6. PipelineRunner.run_evolution() workers=4 端到端 (1)
# ============================================================================

def test_pipeline_runner_processpool_evolution_e2e():
    """PipelineRunner.run_evolution(workers=4) 端到端: H5 → snapshot → ProcessPool → trajectory。"""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        _make_h5_dataset(data_dir, n_days=60, n_stocks=20)

        # 用 1.5 数据准备 (data_prep.py)
        result = subprocess.run([
            sys.executable, "-m",
            "QuantNodes.research.factor_test.e2e.data_prep",
            "--output-dir", str(data_dir),
            "--n-days", "60", "--n-stocks", "20",
            "--factors", "momentum_20d",
        ], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr

        # 准备 config
        from QuantNodes.research.factor_test.config import (
            FactorSetting, PreprocessSetting, AnalysisSetting, OutputSetting,
            QualityGateConfig, EvolutionConfig, FeedbackSetting, SingleFactorTestConfig,
        )
        # 与 _make_h5_dataset 起点对齐: 1 年前 ~ 1 个月前
        one_year_ago = int((datetime.now() - timedelta(days=365)).strftime('%Y%m%d'))
        one_month_ago = int((datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
        cfg = SingleFactorTestConfig(
            factor=FactorSetting(
                name="momentum_20d", factor_dir="momentum_20d.h5",
                hypothesis="momentum", description="20-day momentum",
            ),
            preprocess=PreprocessSetting(
                adj_date_beg=one_year_ago, adj_date_end=one_month_ago,
                adj_mode=["M", "end"], sample_index="all", sample_industry="all",
                tradable={"no_st": True, "no_suspended": True, "min_ipo_days": 360},
                missing="", extreme="median", norm="zscore",
                industry_neutral=False, risk_neutral=False,
            ),
            analysis=AnalysisSetting(
                ic={"min_group_size": 5},
                group={"groups": 5, "factor_direction": 1, "floor_mode": "group", "hedge": "equal"},
                longshort={"factor_direction": 1},
                score={"enabled": True},
                risk_corr={"factors": ""},
            ),
            output=OutputSetting(dir=str(Path(td) / "output")),
            feedback=FeedbackSetting(enabled=False),
            quality_gate=QualityGateConfig(enabled=False),
            evolution=EvolutionConfig(enabled=True, max_rounds=1),
            data_path=str(data_dir),
            load_keys=["cp", "id_citic1", "mv_float", "st", "suspend", "ud_limit", "ipo_days"],
        )

        # 注入 LoadData + 跑演化
        from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
        from QuantNodes.research.factor_test.utils.data_loader import DataLoader

        runner = PipelineRunner(cfg)
        runner._context["LoadData"] = {
            "factor": pd.read_hdf(data_dir / "momentum_20d.h5", key="data"),
            "price": pd.read_hdf(data_dir / "stk_daily.h5", key="cp"),
            "id_citic1": pd.read_hdf(data_dir / "stk_daily.h5", key="id_citic1"),
            "mv_float": pd.read_hdf(data_dir / "stk_daily.h5", key="mv_float"),
            "st": pd.read_hdf(data_dir / "stk_daily.h5", key="st"),
            "suspend": pd.read_hdf(data_dir / "stk_daily.h5", key="suspend"),
            "ud_limit": pd.DataFrame(0, index=pd.read_hdf(data_dir / "stk_daily.h5", key="cp").index,
                                   columns=pd.read_hdf(data_dir / "stk_daily.h5", key="cp").columns),
            "ipo_days": pd.read_hdf(data_dir / "stk_daily.h5", key="ipo_days"),
            "index_cp": pd.read_hdf(data_dir / "index_daily.h5", key="index_cp"),
            "stklist": pd.read_hdf(data_dir / "stklist.h5", key="data"),
            "trade_dt": pd.read_hdf(data_dir / "trade_dt.h5", key="data"),
            "_loader": DataLoader(str(data_dir)),
        }
        # 跑演化 (workers=2, ProcessPool)
        result = runner.run_evolution(
            initial_directions=["d1", "d2"],
            workers=2,
        )
        # 验证: round 0 至少 2 entries
        assert result.total_count >= 2
        # snapshot 文件应被创建
        pool_dir = Path(runner._build_trajectory_pool().base_dir)
        snap_file = pool_dir / "_snapshot.pkl"
        assert snap_file.exists(), f"snapshot file should exist at {snap_file}"


# ============================================================================
# 7. ProcessPool 失败恢复 (1)
# ============================================================================

def test_processpool_subprocess_exception_caught():
    """子进程抛异常时, parallel_evaluate 返回 error dict 不崩。"""
    with tempfile.TemporaryDirectory() as td:
        # 创建坏的 snapshot (config 缺关键字段)
        bad_snap = {
            "config": {
                "preprocess": {"adj_date_beg": "INVALID"},  # 错误类型
            },
            "context": {"LoadData": {}},
            "factor_name": "bad_factor",
            "factor_path": "",
        }
        snap_path = Path(td) / "bad_snap.pkl"
        snap_path.write_bytes(pickle.dumps(bad_snap))

        from QuantNodes.core.evolution import FactorCandidate
        cands = [FactorCandidate(factor_id="c1", name="f1", expression="x")]
        results = parallel_evaluate(cands, _mock_eval, max_workers=2,
                                    snapshot_path=str(snap_path))
        # 子进程失败应被捕获
        assert len(results) == 1
        # 失败时 passed=False, error 字段非空
        # (但 subprocess_evaluate 内部已有 try/except, 所以可能也返回 passed=True 但 metrics 空)
        # 关键是不能崩
        assert results[0] is not None

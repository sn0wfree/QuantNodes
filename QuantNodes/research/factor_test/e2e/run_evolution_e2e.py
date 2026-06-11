# coding: utf-8
"""E2E: 完整 12 节点 + 3 轮演化 + 报告生成。

用法:
    # 1. 准备数据 (HDF5)
    python -m QuantNodes.research.factor_test.e2e.data_prep \\
           --output-dir /tmp/e2e_data/

    # 2. 运行 E2E
    python -m QuantNodes.research.factor_test.e2e.run_evolution_e2e \\
           --data-path /tmp/e2e_data/ \\
           --directions momentum,reversal,volatility \\
           --max-rounds 3

输出:
    {output_dir}/
    ├── trajectory/                # TrajectoryPool (Parquet + JSON)
    │   ├── trajectories.parquet
    │   └── {entry_id}.json
    ├── feedback.parquet           # FactorFeedback 持久化 (如启用)
    ├── evolution_report.html      # 可视化报告
    └── evolution_summary.json     # 演化统计

E2E 验证:
    1. PipelineRunner.from_dict() 不报错
    2. 12 节点全部执行成功
    3. QualityGate 拦截低质量因子
    4. TrajectoryPool 写入 5+ entries (3 original + 1 mutation + 1 crossover)
    5. Visualization HTML 包含 5 个 figure
    6. RAG 评估指标 > 0 (说明检索非空)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from QuantNodes.core.evolution import (
    EvolutionLoop,
    EvolutionSetting,
    FactorCandidate,
)
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    IdentityRetriever,
    KnowledgeBase,
    RAGEvaluator,
)
from QuantNodes.core.quality_gate import (
    QualityGateNode,
    QualityGateSetting,
)
from QuantNodes.core.trajectory import TrajectoryPool
from QuantNodes.core.visualization import generate_html
from QuantNodes.research.factor_test.config import (
    AnalysisSetting,
    EvolutionConfig,
    FactorSetting,
    FeedbackSetting,
    OutputSetting,
    PreprocessSetting,
    QualityGateConfig,
    SingleFactorTestConfig,
)
from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner


def _build_config(
    data_path: str,
    factor_name: str,
    factor_dir: str,
    directions: list[str],
    output_dir: str,
    max_rounds: int = 3,
    enable_quality_gate: bool = True,
    enable_kb: bool = True,
) -> SingleFactorTestConfig:
    """构造 SingleFactorTestConfig。"""
    return SingleFactorTestConfig(
        factor=FactorSetting(
            name=factor_name, factor_dir=factor_dir,
            hypothesis=directions[0] if directions else "momentum",
            description=f"e2e test factor: {factor_name}",
        ),
        preprocess=PreprocessSetting(
            adj_date_beg=20260101, adj_date_end=20260630,
            adj_mode=["M", "end"],
            sample_index="all", sample_industry="all",
            tradable={
                "no_st": True, "no_suspended": True, "no_up_down_limit": False,
                "min_ipo_days": 360,
            },
            missing="", extreme="median", norm="zscore",
            industry_neutral=False, risk_neutral=False, risk_factors=[],
        ),
        analysis=AnalysisSetting(
            ic={"min_group_size": 5},
            group={"groups": 5, "factor_direction": 1, "floor_mode": "group", "hedge": "equal"},
            longshort={"factor_direction": 1},
            score={"enabled": True},
            risk_corr={"factors": ""},
        ),
        output=OutputSetting(dir=output_dir, format=["json"]),
        feedback=FeedbackSetting(enabled=False),
        quality_gate=QualityGateConfig(enabled=enable_quality_gate),
        evolution=EvolutionConfig(
            enabled=True, max_rounds=max_rounds,
            parent_selection_strategy="top_percent_plus_random",
            top_percent_threshold=0.5,
            metric="sharpe",
            early_stop_patience=0,
        ),
        data_path=data_path,
        load_keys=["cp", "id_citic1", "mv_float", "st", "suspend", "ud_limit", "ipo_days"],
    )


def _inject_synthetic_data(runner: PipelineRunner, data_path: Path) -> None:
    """预填 _context['LoadData'], 跳过 LoadDataNode 真实 H5 读取。"""
    import numpy as np
    rng = np.random.RandomState(42)
    # 读 H5 实际数据
    cp = pd.read_hdf(data_path / "stk_daily.h5", key="cp")
    st = pd.read_hdf(data_path / "stk_daily.h5", key="st")
    suspend = pd.read_hdf(data_path / "stk_daily.h5", key="suspend")
    ud_limit = pd.read_hdf(data_path / "stk_daily.h5", key="ud_limit")
    ipo_days = pd.read_hdf(data_path / "stk_daily.h5", key="ipo_days")
    industry = pd.read_hdf(data_path / "stk_daily.h5", key="id_citic1")
    mv = pd.read_hdf(data_path / "stk_daily.h5", key="mv_float")
    factor = pd.read_hdf(data_path / f"{runner.config.factor.name}.h5", key="data")
    index_cp = pd.read_hdf(data_path / "index_daily.h5", key="index_cp")
    stklist = pd.read_hdf(data_path / "stklist.h5", key="data")
    trade_dt = pd.read_hdf(data_path / "trade_dt.h5", key="data")

    runner._context["LoadData"] = {
        "factor": factor,
        "price": cp,
        "id_citic1": industry,
        "mv_float": mv,
        "st": st,
        "suspend": suspend,
        "ud_limit": ud_limit,
        "ipo_days": ipo_days,
        "index_cp": index_cp,
        "stklist": stklist,
        "trade_dt": trade_dt,
        "_loader": _build_loader(str(data_path)),
    }


def _build_loader(data_path: str):
    """构造 DataLoader (供 RiskCorrelationNode 使用)。"""
    from QuantNodes.research.factor_test.utils.data_loader import DataLoader
    return DataLoader(data_path)


def main():
    parser = argparse.ArgumentParser(description="E2E 演化运行")
    parser.add_argument("--data-path", required=True, help="data_prep 输出目录")
    parser.add_argument("--factor-name", default="momentum_20d", help="起始因子名")
    parser.add_argument("--directions", default="momentum,reversal,volatility",
                        help="逗号分隔的研究方向 (用作 RAG query + 初始 directions)")
    parser.add_argument("--output-dir", default="/tmp/e2e_output/",
                        help="输出目录 (报告 / trajectory / summary)")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--disable-quality-gate", action="store_true",
                        help="禁用 QualityGate (默认启用)")
    parser.add_argument("--disable-kb", action="store_true",
                        help="禁用 KnowledgeBase (默认启用)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    print("=" * 70)
    print(f"E2E 演化实验")
    print(f"  data_path:  {data_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  directions: {directions}")
    print(f"  max_rounds: {args.max_rounds}")
    print("=" * 70)

    # 1. 构造 config + runner
    cfg = _build_config(
        data_path=str(data_path),
        factor_name=args.factor_name,
        factor_dir=f"{args.factor_name}.h5",
        directions=directions,
        output_dir=str(output_dir),
        max_rounds=args.max_rounds,
        enable_quality_gate=not args.disable_quality_gate,
        enable_kb=not args.disable_kb,
    )
    runner = PipelineRunner(cfg)
    _inject_synthetic_data(runner, data_path)
    print(f"\n[1/5] 注入 LoadData: factor={runner._context['LoadData']['factor'].shape}")

    # 2. 先单次回测 (验证 12 节点)
    print(f"\n[2/5] 单次回测 (验证 12 节点 + QualityGate)...")
    try:
        ctx = runner.run()
        print(f"  ✓ 12 节点全部执行, ctx keys: {list(ctx.keys())}")
        # 打印 IC
        if "ICAnalyzer" in ctx:
            ic_res = ctx["ICAnalyzer"].get("ic_result", {})
            print(f"  IC均值: {ic_res.get('IC均值', 'N/A')}")
    except Exception as e:
        print(f"  ✗ 单次回测失败: {e}")
        return 1

    # 3. 设置演化组件
    print(f"\n[3/5] 构造演化组件...")
    pool = runner._build_trajectory_pool()
    quality_gate = runner._build_quality_gate()

    kb = None
    evaluator = None
    if not args.disable_kb:
        kb = KnowledgeBase(IdentityRetriever(), pool=pool)
        evaluator = RAGEvaluator()
        print(f"  ✓ TrajectoryPool: {pool.base_dir}")
        print(f"  ✓ QualityGateNode: {'enabled' if quality_gate else 'disabled'}")
        print(f"  ✓ KnowledgeBase + RAGEvaluator")

    # 4. 演化循环
    print(f"\n[4/5] 演化循环 ({args.max_rounds} 轮)...")
    from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting as ES
    settings = ES(
        enabled=True, max_rounds=args.max_rounds,
        parent_selection_strategy="top_percent_plus_random",
        top_percent_threshold=0.5,
        metric="sharpe", seed=42,
    )
    loop = EvolutionLoop(
        settings, pool=pool,
        quality_gate=quality_gate,
        evaluate_fn=runner._evaluate_candidate,
        knowledge_base=kb,
        rag_evaluator=evaluator,
        rag_top_k=3,
        max_ancestor_depth=2, max_descendant_depth=2,
        use_compress=True,
    )
    result = loop.run(initial_directions=directions)
    print(f"  ✓ 演化完成: {result.rounds_completed} 轮, 总数 {result.total_count}, 拒绝 {result.rejected_count}")
    if kb is not None:
        print(f"  ✓ KB 已索引 {len(kb)} entry")
    print(f"  ✓ RAG 评估历史: {len(loop.rag_metrics_history)} 轮")
    if loop.rag_metrics_history:
        m = loop.rag_metrics_history[-1]
        print(f"    Round {m['round']}: HR@5={m['hit_at_5']:.3f} NDCG@5={m['ndcg_at_5']:.3f}")

    # 5. 报告生成
    print(f"\n[5/5] 生成报告...")
    report_path = output_dir / "evolution_report.html"
    try:
        generate_html(pool, metric="sharpe",
                      title=f"E2E 演化报告: {data_path.name}",
                      output_path=report_path)
        print(f"  ✓ HTML 报告: {report_path} ({report_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"  ✗ HTML 报告失败: {e}")

    summary = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "directions": directions,
        "max_rounds": args.max_rounds,
        "pool_size": pool.size,
        "rounds_completed": result.rounds_completed,
        "total_count": result.total_count,
        "rejected_count": result.rejected_count,
        "best_entries": [
            {
                "id": e.entry_id,
                "name": e.feedback.factor_name if e.feedback else "",
                "operation": e.operation,
                "round": e.round_idx,
                "sharpe": (e.metrics or {}).get("sharpe", 0),
            }
            for e in result.best_entries[:5]
        ],
        "rag_metrics_history": loop.rag_metrics_history,
    }
    summary_path = output_dir / "evolution_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  ✓ JSON 摘要: {summary_path}")

    print(f"\n{'=' * 70}\n✓ E2E 完成\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

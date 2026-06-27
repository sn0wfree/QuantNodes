#!/usr/bin/env python3
# coding=utf-8
"""
large_scale_e2e_test.py - 大规模端到端测试

测试场景:
1. 基线对照 (无逻辑约束, 标准 Pipeline)
2. Gamma 约束测试 (量价背离逻辑)
3. LogicDrivenPipeline 测试 (外层 + 内层 + MCTS)
4. 一致性评分测试 (结构化逻辑匹配)

Usage:
    export QUANTNODES__LLM__API_KEY=...
    export QUANTNODES__LLM__BASE_URL=...
    export QUANTNODES__LLM__MODEL=...
    python3.11 tests/quant_alpha/large_scale_e2e_test.py
"""

import json
import sys
import time
from pathlib import Path

import polars as pl

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicBehavior,
    LogicCondition,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.logic_mining.compiler import compile_to_constraint
from QuantNodes.research.quant_alpha.pipeline import (
    AlphaPipeline,
    PipelineConfig,
    TerminationConfig,
)
from QuantNodes.research.quant_alpha.logic_driven_pipeline import (
    LogicDrivenPipeline,
    LogicDrivenPipelineConfig,
)
from QuantNodes.research.quant_alpha.mcts.feedback import collect_llm_channel


DATA_PATH = "data/cache/full_a_2019_2024.parquet"
OUTPUT_DIR = Path("pipeline_output_e2e")


def setup_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_test_result(name: str, result: dict):
    """保存测试结果"""
    out_file = OUTPUT_DIR / f"{name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {out_file}")


def test_1_baseline(data: pl.DataFrame):
    """测试 1: 基线对照（无 Gamma 约束）"""
    print("\n=== Test 1: 基线对照（无 Gamma 约束） ===")
    start = time.time()

    config = PipelineConfig(
        objective="capture A-share reversal effect",
        termination=TerminationConfig(
            max_rounds=1,
            target_factors=5,
            early_stopping=True,
            patience=2,
        ),
        alphagpt_iterations=1,
        alphagpt_pool_size=3,
        mcts_iterations=10,
        mcts_max_depth=3,
        llm_provider="minimax",
        output_dir=str(OUTPUT_DIR / "test1"),
    )
    pipeline = AlphaPipeline(config)
    result = pipeline.run(data)

    summary = {
        "test": "baseline",
        "total_rounds": len(result.rounds),
        "final_factors": len(result.final_pool),
        "elapsed_seconds": result.elapsed_seconds,
        "best_ir": max(
            (m.ir for m in result.final_pool), default=0.0
        ),
        "round_details": [
            {
                "round": r.round_num,
                "best_ir": r.feedback.best_ir if r.feedback else 0.0,
                "valid_factors": r.feedback.valid_count if r.feedback else 0,
            }
            for r in result.rounds
        ],
    }
    save_test_result("test1_baseline", summary)
    print(f"  Final factors: {len(result.final_pool)}")
    print(f"  Best IR: {summary['best_ir']:.4f}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
    return summary


def test_2_gamma(data: pl.DataFrame):
    """测试 2: Gamma 约束（量价背离逻辑）"""
    print("\n=== Test 2: Gamma 约束（量价背离） ===")
    start = time.time()

    logic = WikiLogicStructured(
        predicates=[
            LogicCondition(variable="open", op="rank", threshold=0),
            LogicCondition(variable="volume", op="rank", threshold=0),
            LogicCondition(
                variable="open", op="ts_corr", threshold=-0.5, window=10,
                second_variable="volume",
            ),
        ],
        behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        operator_whitelist=["rank", "ts_corr", "sign", "sub", "mul", "div"],
        parameter_ranges={"ts_corr": (5, 30)},
        sign_constraint=-1,
    )
    gamma = compile_to_constraint(logic, source_logic="price_volume_divergence")

    config = PipelineConfig(
        objective="capture A-share reversal effect",
        termination=TerminationConfig(
            max_rounds=1,
            target_factors=5,
            early_stopping=True,
            patience=2,
        ),
        alphagpt_iterations=1,
        alphagpt_pool_size=3,
        mcts_iterations=10,
        mcts_max_depth=3,
        llm_provider="minimax",
        output_dir=str(OUTPUT_DIR / "test2"),
        gamma=gamma,
        structured_logic=logic,
    )
    pipeline = AlphaPipeline(config)
    result = pipeline.run(data)

    summary = {
        "test": "gamma_constrained",
        "logic_name": "price_volume_divergence",
        "total_rounds": len(result.rounds),
        "final_factors": len(result.final_pool),
        "elapsed_seconds": result.elapsed_seconds,
        "best_ir": max(
            (m.ir for m in result.final_pool), default=0.0
        ),
    }
    save_test_result("test2_gamma", summary)
    print(f"  Final factors: {len(result.final_pool)}")
    print(f"  Best IR: {summary['best_ir']:.4f}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
    return summary


def test_3_logic_driven(data: pl.DataFrame):
    """测试 3: LogicDrivenPipeline（外层循环）"""
    print("\n=== Test 3: LogicDrivenPipeline ===")
    start = time.time()

    config = LogicDrivenPipelineConfig(
        objective="capture A-share reversal effect",
        wiki_path="wiki_logic_e2e",
        output_dir=str(OUTPUT_DIR / "test3"),
        logic_driven=True,
        alphalogics_inner_iterations=1,
        alphalogics_inner_pool_size=3,
        alphalogics_max_outer_rounds=2,
        alphalogics_initial_sources=("alpha101",),
        alphalogics_initial_max_per_lib=2,
        alphagpt_iterations=1,
        alphagpt_pool_size=3,
        mcts_iterations=5,
        mcts_max_depth=3,
        llm_provider="minimax",
        timeout_seconds=1800,
    )
    pipeline = LogicDrivenPipeline(config=config)
    result = pipeline.run(data)

    summary = {
        "test": "logic_driven",
        "best_logic_name": result.best_logic_name,
        "best_ir": (
            result.best_evidence.best_ir if result.best_evidence else 0.0
        ),
        "best_n_factors": (
            result.best_evidence.n_factors_explored
            if result.best_evidence else 0
        ),
        "final_pool_size": len(result.final_pool),
        "wiki_pages": len(result.wiki_pages),
        "elapsed_seconds": result.elapsed_seconds,
    }
    save_test_result("test3_logic_driven", summary)
    print(f"  Best logic: {result.best_logic_name}")
    print(f"  Best IR: {summary['best_ir']:.4f}")
    print(f"  Final factors: {len(result.final_pool)}")
    print(f"  Wiki pages: {len(result.wiki_pages)}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
    return summary


def test_4_consistency_hook():
    """测试 4: 一致性评分（结构化逻辑匹配）"""
    print("\n=== Test 4: 一致性评分 ===")

    logic = WikiLogicStructured(
        predicates=[
            LogicCondition(variable="open", op="rank", threshold=0),
            LogicCondition(variable="volume", op="rank", threshold=0),
            LogicCondition(
                variable="open", op="ts_corr", threshold=-0.5, window=10,
                second_variable="volume",
            ),
        ],
        behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        operator_whitelist=["rank", "ts_corr", "sign"],
        parameter_ranges={"ts_corr": (5, 30)},
        sign_constraint=-1,
    )

    test_cases = [
        ("sign(-ts_corr(rank(open), rank(volume), 10))", "应高匹配"),
        ("ts_argmax(close, 5)", "应低匹配（算子不在白名单）"),
        ("rank(close)", "应中匹配（close 不在变量白名单）"),
        ("rank(ts_corr(close, close, 10))", "应中匹配（close 不在变量白名单）"),
    ]

    results = []
    for formula, desc in test_cases:
        fb = collect_llm_channel(
            formula=formula,
            structured_logic=logic,
            score_threshold=0.5,
        )
        results.append({
            "formula": formula,
            "description": desc,
            "score": fb.score,
            "passed": fb.passed,
            "mode": fb.metadata.get("mode"),
            "operator_overlap": fb.metadata.get("operator_overlap"),
            "variable_overlap": fb.metadata.get("variable_overlap"),
            "direction_match": fb.metadata.get("direction_match"),
        })
        print(f"  {formula[:50]:50s} score={fb.score:.2f} passed={fb.passed} [{desc}]")

    summary = {
        "test": "consistency_hook",
        "total_cases": len(results),
        "passed_cases": sum(1 for r in results if r["passed"]),
        "results": results,
    }
    save_test_result("test4_consistency", summary)
    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("大规模端到端测试 (Large-Scale E2E Test)")
    print("=" * 60)

    setup_output_dir()

    print(f"\n加载数据: {DATA_PATH}")
    data = pl.read_parquet(DATA_PATH)
    print(f"  Rows: {len(data)}, Columns: {data.columns}")

    all_results = {}
    all_results["test1_baseline"] = test_1_baseline(data)
    all_results["test2_gamma"] = test_2_gamma(data)
    all_results["test3_logic_driven"] = test_3_logic_driven(data)
    all_results["test4_consistency"] = test_4_consistency_hook()

    # 汇总报告
    print("\n" + "=" * 60)
    print("测试汇总 (Test Summary)")
    print("=" * 60)

    for name, result in all_results.items():
        print(f"\n[{name}]")
        for k, v in result.items():
            if k in ("round_details", "results"):
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # 保存汇总
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总保存到: {summary_file}")


if __name__ == "__main__":
    main()
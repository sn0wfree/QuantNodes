#!/usr/bin/env python3
# coding=utf-8
"""
run_logic_driven.py - LogicDrivenPipeline 端到端运行

使用 LogicDrivenPipeline 端到端运行 4 个市场逻辑。
包括外层循环（AlphaLogics）+ 内层循环（Alpha-GPT）+ MCTS 增强 + Wiki 持久化。

Usage:
    export QUANTNODES__LLM__API_KEY=...
    export QUANTNODES__LLM__BASE_URL=...
    export QUANTNODES__LLM__MODEL=...
    python3.11 tests/quant_alpha/run_logic_driven.py
"""

import json
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
from QuantNodes.research.quant_alpha.workflow.alpha_logics import (
    AlphaLogicsConfig,
    AlphaLogicsWorkflow,
)

DATA_PATH = "data/cache/full_a_2019_2024.parquet"
OUTPUT_DIR = Path("pipeline_output_logic_driven")
WIKI_PATH = "wiki_logic_driven"


def define_logics() -> list:
    """定义 4 个市场逻辑（基于经验放宽约束）"""
    return [
        # 1. 量价背离反转（放宽：增加 ts_mean, abs, log）
        {
            "name": "price_volume_divergence",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="open", op="rank", threshold=0),
                    LogicCondition(variable="volume", op="rank", threshold=0),
                    LogicCondition(
                        variable="open", op="ts_corr", threshold=-0.5, window=10,
                        second_variable="volume",
                    ),
                ],
                behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
                operator_whitelist=[
                    "rank", "ts_corr", "sign", "sub", "mul", "div",
                    "ts_mean", "abs",  # 放宽：增加基础算子
                ],
                parameter_ranges={"ts_corr": (5, 60), "ts_mean": (5, 60)},
                sign_constraint=-1,
            ),
        },
        # 2. 均线反转（增加 pool_size 提高多样性）
        {
            "name": "mean_reversion",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
                    LogicCondition(variable="close", op="ts_mean", threshold=0, window=5),
                ],
                behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
                operator_whitelist=["rank", "ts_mean", "ts_std", "sub", "div", "sign"],
                parameter_ranges={"ts_mean": (5, 60), "ts_std": (5, 60)},
                sign_constraint=-1,
            ),
        },
        # 3. 动量因子（多轮迭代）
        {
            "name": "momentum",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
                ],
                behavior=LogicBehavior(target="forward_return_20", direction=+1, horizon=20),
                operator_whitelist=["rank", "ts_mean", "ts_std", "sub", "div", "mul"],
                parameter_ranges={"ts_mean": (10, 120)},
                sign_constraint=+1,
            ),
        },
        # 4. 波动率因子（多轮迭代）
        {
            "name": "volatility",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="close", op="ts_std", threshold=0, window=20),
                ],
                behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
                operator_whitelist=["rank", "ts_std", "ts_mean", "div"],
                parameter_ranges={"ts_std": (5, 60), "ts_mean": (5, 60)},
                sign_constraint=-1,
            ),
        },
    ]


def run_logic_driven_pipeline(
    data: pl.DataFrame,
    logic_name: str,
    logic: WikiLogicStructured,
    max_outer_rounds: int = 2,
    inner_iterations: int = 1,
    inner_pool_size: int = 3,
    initial_max_per_lib: int = 2,
) -> dict:
    """使用 LogicDrivenPipeline 跑单个逻辑"""
    print(f"\n{'=' * 60}")
    print(f"LogicDrivenPipeline: {logic_name}")
    print(f"{'=' * 60}")

    gamma = compile_to_constraint(logic, source_logic=logic_name)
    print(f"Γ 约束: ops={len(gamma.operator_whitelist)}, "
          f"vars={len(gamma.variable_whitelist or [])}")

    # 配置 AlphaLogicsWorkflow（外层循环）
    alphalogics_config = AlphaLogicsConfig(
        inner_iterations=inner_iterations,
        inner_pool_size=inner_pool_size,
        inner_early_stop=2,
        max_outer_rounds=max_outer_rounds,
        inner_objective="ir",
        data=data,
        date_column="date",
        code_column="code",
        forward_returns=(1, 5, 20),
        wiki_path=WIKI_PATH,
        persist_best_logic=True,
        initial_logic_sources=("alpha101", "alpha158"),
        initial_logic_max_per_lib=initial_max_per_lib,
        min_ir_threshold=0.05,
    )

    start = time.time()
    workflow = AlphaLogicsWorkflow(config=alphalogics_config)
    alphalogics_result = workflow.run()
    alphalogics_elapsed = time.time() - start

    print(f"\nAlphaLogics 结果:")
    print(f"  最佳逻辑: {alphalogics_result.best_logic.name if alphalogics_result.best_logic else None}")
    print(f"  最佳 IR: {alphalogics_result.best_evidence.best_ir if alphalogics_result.best_evidence else 0.0:.4f}")
    print(f"  库大小: {len(alphalogics_result.library)}")
    print(f"  耗时: {alphalogics_elapsed:.1f}s")

    # MCTS 增强（基于 best logic 的 Gamma 约束）
    print(f"\nMCTS 增强（基于 {logic_name}）...")
    mcts_start = time.time()

    pipeline_config = PipelineConfig(
        objective=f"Apply logic {logic_name}",
        termination=TerminationConfig(
            max_rounds=1, target_factors=5,
            early_stopping=True, patience=2,
        ),
        alphagpt_iterations=1,
        alphagpt_pool_size=3,
        mcts_iterations=10,
        mcts_max_depth=3,
        llm_provider="minimax",
        output_dir=str(OUTPUT_DIR / logic_name),
        gamma=gamma,
        structured_logic=logic,
    )
    pipeline = AlphaPipeline(pipeline_config)
    pipeline_result = pipeline.run(data)
    mcts_elapsed = time.time() - mcts_start

    print(f"  MCTS 因子: {len(pipeline_result.final_pool)}")
    print(f"  最佳 IR: {max((m.ir for m in pipeline_result.final_pool), default=0.0):.4f}")
    print(f"  耗时: {mcts_elapsed:.1f}s")

    # 收集所有因子
    all_factors = []
    for ir in alphalogics_result.inner_results:
        if ir.alphagpt_result:
            from QuantNodes.research.quant_alpha.evaluation.contracts import (
                FactorMetrics,
            )
            for f in ir.alphagpt_result.final_pool:
                all_factors.append(FactorMetrics(
                    formula_id=f.formula_id,
                    status="success",
                    ic_mean=f.ic_mean,
                    ir=f.ir,
                    overall_score=f.ir,
                ))

    for f in pipeline_result.final_pool:
        all_factors.append(f)

    # 去重（按 IR 排序，取 Top-5）
    all_factors.sort(key=lambda m: abs(m.ir), reverse=True)
    top_factors = all_factors[:5]

    summary = {
        "logic_name": logic_name,
        "alphalogics_elapsed": alphalogics_elapsed,
        "mcts_elapsed": mcts_elapsed,
        "total_elapsed": alphalogics_elapsed + mcts_elapsed,
        "best_logic_name": (
            alphalogics_result.best_logic.name
            if alphalogics_result.best_logic else None
        ),
        "best_evidence_ir": (
            alphalogics_result.best_evidence.best_ir
            if alphalogics_result.best_evidence else 0.0
        ),
        "library_size": len(alphalogics_result.library),
        "inner_rounds": len(alphalogics_result.inner_results),
        "mcts_factors": len(pipeline_result.final_pool),
        "total_factors": len(all_factors),
        "top_factors": [
            {
                "formula_id": f.formula_id,
                "ir": f.ir,
                "ic_mean": f.ic_mean,
            }
            for f in top_factors
        ],
    }

    # 持久化结果
    out_file = OUTPUT_DIR / f"{logic_name}_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("LogicDrivenPipeline 端到端运行")
    print("=" * 60)
    print(f"数据: {DATA_PATH}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"Wiki: {WIKI_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pl.read_parquet(DATA_PATH)
    print(f"数据: {len(data)} 行, {data['code'].n_unique()} 只股票")

    logics = define_logics()
    print(f"\n待挖掘逻辑: {len(logics)} 个")

    all_results = []
    total_start = time.time()

    for lg in logics:
        try:
            result = run_logic_driven_pipeline(
                data,
                lg["name"],
                lg["logic"],
                max_outer_rounds=2,      # 2 轮外层
                inner_iterations=1,        # 1 轮内层
                inner_pool_size=3,         # 3 个公式
                initial_max_per_lib=2,     # 初始库 2 个
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ {lg['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "logic_name": lg["name"],
                "error": str(e),
                "total_factors": 0,
            })

    total_elapsed = time.time() - total_start

    # 汇总
    print("\n" + "=" * 60)
    print("LogicDrivenPipeline 汇总")
    print("=" * 60)

    for r in all_results:
        if "error" in r:
            print(f"  ✗ {r['logic_name']}: {r['error']}")
        else:
            print(f"  ✓ {r['logic_name']}: "
                  f"best_ir={r['best_evidence_ir']:.4f}, "
                  f"total_factors={r['total_factors']}, "
                  f"{r['total_elapsed']:.1f}s")

    # 全局汇总
    total_factors = sum(r.get("total_factors", 0) for r in all_results)
    summary = {
        "total_logics": len(logics),
        "total_factors": total_factors,
        "total_elapsed_seconds": total_elapsed,
        "results": all_results,
    }

    with open(OUTPUT_DIR / "logic_driven_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n总因子: {total_factors}")
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"汇总保存到: {OUTPUT_DIR}/logic_driven_summary.json")


if __name__ == "__main__":
    main()
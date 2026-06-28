#!/usr/bin/env python3
# coding=utf-8
"""
run_4_logic_v5.py - V5 实验: 思维链利用版 4 逻辑 E2E

与 V4 完全相同的设置（4 逻辑 × 3 ideas × MCTS 20），
但启用 Tier 1+2+4:
- Tier 1: LLM 思维链捕获 & 持久化
- Tier 2: 结构化推理 prompt
- Tier 4: OpPrior 算子先验

输出:
- pipeline_output_v5/{logic_name}/...
- pipeline_output_v5/v5_summary.json
- pipeline_output_v5/op_prior.json (跨 4 逻辑累积)

Usage:
    export QUANTNODES__LLM__API_KEY=...
    export QUANTNODES__LLM__BASE_URL=...
    export QUANTNODES__LLM__MODEL=minimax-M3
    python3.11 tests/quant_alpha/run_4_logic_v5.py
"""

import json
import os
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
from QuantNodes.research.quant_alpha.mcts.op_prior import OpPrior

DATA_PATH = "data/cache/full_a_2019_2024.parquet"
OUTPUT_DIR = Path("pipeline_output_v5")


def define_logics():
    """4 个市场逻辑（与 V4 完全相同）"""
    return [
        {
            "name": "price_volume_divergence",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="open", op="rank", threshold=0),
                    LogicCondition(variable="volume", op="rank", threshold=0),
                    LogicCondition(
                        variable="open", op="ts_corr",
                        threshold=-0.5, window=10,
                        second_variable="volume",
                    ),
                ],
                behavior=LogicBehavior(
                    target="forward_return_5", direction=-1, horizon=5,
                ),
                operator_whitelist=["rank", "ts_corr", "sign", "sub", "mul", "div"],
                parameter_ranges={"ts_corr": (5, 30)},
                sign_constraint=-1,
            ),
        },
        {
            "name": "mean_reversion",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
                    LogicCondition(variable="close", op="ts_mean", threshold=0, window=5),
                ],
                behavior=LogicBehavior(
                    target="forward_return_5", direction=-1, horizon=5,
                ),
                operator_whitelist=["rank", "ts_mean", "ts_std", "sub", "div", "sign"],
                parameter_ranges={"ts_mean": (5, 60), "ts_std": (5, 60)},
                sign_constraint=-1,
            ),
        },
        {
            "name": "momentum",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
                ],
                behavior=LogicBehavior(
                    target="forward_return_20", direction=+1, horizon=20,
                ),
                operator_whitelist=["rank", "ts_mean", "ts_std", "sub", "div", "mul"],
                parameter_ranges={"ts_mean": (10, 120)},
                sign_constraint=+1,
            ),
        },
        {
            "name": "volatility",
            "logic": WikiLogicStructured(
                predicates=[
                    LogicCondition(variable="close", op="ts_std", threshold=0, window=20),
                ],
                behavior=LogicBehavior(
                    target="forward_return_5", direction=-1, horizon=5,
                ),
                operator_whitelist=["rank", "ts_std", "ts_mean", "div"],
                parameter_ranges={"ts_std": (5, 60), "ts_mean": (5, 60)},
                sign_constraint=-1,
            ),
        },
    ]


def run_for_logic(
    data: pl.DataFrame,
    logic_name: str,
    logic: WikiLogicStructured,
) -> dict:
    """为单个逻辑跑 V5 (with Tier 1+2+4)"""
    print(f"\n{'=' * 60}")
    print(f"V5 - 逻辑: {logic_name} (Tier 1+2+4)")
    print(f"{'=' * 60}")

    gamma = compile_to_constraint(logic, source_logic=logic_name)
    print(f"Γ 约束:")
    print(f"  算子: {sorted(gamma.operator_whitelist)}")
    print(f"  符号: {gamma.sign_constraint}")

    config = PipelineConfig(
        objective=f"Apply logic {logic_name}",
        termination=TerminationConfig(
            max_rounds=1,
            target_factors=3,
            early_stopping=False,
        ),
        alphagpt_iterations=1,
        alphagpt_pool_size=3,
        alphagpt_top_k=3,
        mcts_iterations=20,
        mcts_max_depth=4,
        mcts_dedup_threshold=0.7,
        max_mutual_ic=0.7,
        min_ir_threshold=0.05,
        forward_returns=(1, 5, 20),
        llm_provider="minimax",
        output_dir=str(OUTPUT_DIR / logic_name),
        gamma=gamma,
        structured_logic=logic,
    )

    start = time.time()
    pipeline = AlphaPipeline(config)
    result = pipeline.run(data)
    elapsed = time.time() - start

    # Tier 4 报告：OpPrior 状态
    op_prior_path = Path(OUTPUT_DIR) / "op_prior.json"
    if op_prior_path.exists():
        try:
            op_prior = OpPrior.load(op_prior_path)
            top_5 = op_prior.top_k(5)
        except Exception:
            top_5 = []
    else:
        top_5 = []

    summary = {
        "logic_name": logic_name,
        "total_rounds": len(result.rounds),
        "final_factors": len(result.final_pool),
        "elapsed_seconds": elapsed,
        "best_ir": max((m.ir for m in result.final_pool), default=0.0),
        "avg_ir": (
            sum(m.ir for m in result.final_pool) / len(result.final_pool)
            if result.final_pool else 0.0
        ),
        "best_abs_ir": max(
            (abs(m.ir) for m in result.final_pool), default=0.0
        ),
        "factors": [
            {
                "formula_id": m.formula_id,
                "ir": m.ir,
                "ic_mean": m.ic_mean,
            }
            for m in result.final_pool
        ],
        "op_prior_top5": top_5,
    }

    print(f"\nV5 - {logic_name} 结果:")
    print(f"  最终因子: {summary['final_factors']}")
    print(f"  最佳 IR: {summary['best_ir']:.4f}")
    print(f"  最佳 |IR|: {summary['best_abs_ir']:.4f}")
    print(f"  平均 IR: {summary['avg_ir']:.4f}")
    print(f"  耗时: {elapsed:.1f}s")
    if top_5:
        print(f"  OpPrior top-5: {top_5}")
    for f in summary["factors"]:
        print(f"    {f['formula_id']}: IR={f['ir']:.4f}, IC={f['ic_mean']:.4f}")

    return summary


def main():
    print("=" * 60)
    print("V5 Experiment - 4 逻辑 E2E（思维链利用版）")
    print("=" * 60)

    if not os.environ.get("QUANTNODES__LLM__API_KEY"):
        print("ERROR: QUANTNODES__LLM__API_KEY not set")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

    print(f"\n加载数据: {DATA_PATH}")
    data = pl.read_parquet(DATA_PATH)
    print(f"  Rows: {len(data)}, Stocks: {data['code'].n_unique()}")

    logics = define_logics()
    print(f"\n待跑逻辑: {len(logics)} 个")
    for lg in logics:
        print(f"  - {lg['name']}")

    all_results = []
    total_start = time.time()

    for lg in logics:
        try:
            result = run_for_logic(data, lg["name"], lg["logic"])
            all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ {lg['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "logic_name": lg["name"],
                "error": str(e),
                "final_factors": 0,
            })

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("V5 汇总 (Summary)")
    print("=" * 60)

    total_factors = sum(r.get("final_factors", 0) for r in all_results)
    best_abs_ir_overall = max(
        (r.get("best_abs_ir", 0.0) for r in all_results), default=0.0
    )

    # 加载最终 OpPrior
    op_prior_path = Path(OUTPUT_DIR) / "op_prior.json"
    final_op_prior = None
    if op_prior_path.exists():
        try:
            final_op_prior = OpPrior.load(op_prior_path)
        except Exception:
            pass

    summary = {
        "version": "V5",
        "description": "4-logic E2E with Tier 1+2+4 (thinking chain utilization)",
        "tiers_enabled": ["Tier 1 (capture)", "Tier 2 (structured reasoning)", "Tier 4 (OpPrior)"],
        "total_logics": len(logics),
        "total_factors": total_factors,
        "best_abs_ir_overall": best_abs_ir_overall,
        "total_elapsed_seconds": total_elapsed,
        "results": all_results,
        "op_prior_final_top10": final_op_prior.top_k(10) if final_op_prior else [],
        "op_prior_total_updates": final_op_prior.total_updates if final_op_prior else 0,
    }

    for r in all_results:
        if "error" in r:
            print(f"  ✗ {r['logic_name']}: {r['error']}")
        else:
            print(f"  ✓ {r['logic_name']}: "
                  f"{r['final_factors']} 因子, "
                  f"best_|IR|={r['best_abs_ir']:.4f}, "
                  f"{r['elapsed_seconds']:.1f}s")

    print(f"\n总因子: {total_factors}")
    print(f"总耗时: {total_elapsed:.1f}s")
    if final_op_prior:
        print(f"OpPrior 总更新: {final_op_prior.total_updates}")
        print(f"OpPrior 覆盖算子: {len(final_op_prior.weights)}")

    summary_file = OUTPUT_DIR / "v5_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n汇总保存到: {summary_file}")


if __name__ == "__main__":
    main()

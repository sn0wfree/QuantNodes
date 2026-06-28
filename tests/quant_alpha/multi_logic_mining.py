#!/usr/bin/env python3
# coding=utf-8
"""
multi_logic_mining.py - 多逻辑并行因子挖掘

使用 LogicDrivenPipeline 同时挖掘多个市场逻辑的因子。

逻辑列表:
1. 量价背离反转 (price_volume_divergence)
2. 均线反转 (mean_reversion)
3. 动量因子 (momentum)
4. 波动率因子 (volatility)

Usage:
    export QUANTNODES__LLM__API_KEY=...
    export QUANTNODES__LLM__BASE_URL=...
    export QUANTNODES__LLM__MODEL=...
    python3.11 tests/quant_alpha/multi_logic_mining.py
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

DATA_PATH = "data/cache/full_a_2019_2024.parquet"
OUTPUT_DIR = Path("pipeline_output_mining")


def define_logics() -> list:
    """定义多个市场逻辑"""
    logics = [
        # 1. 量价背离反转
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
                operator_whitelist=["rank", "ts_corr", "sign", "sub", "mul", "div"],
                parameter_ranges={"ts_corr": (5, 30)},
                sign_constraint=-1,
            ),
        },
        # 2. 均线反转
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
        # 3. 动量因子
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
        # 4. 波动率因子
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
    return logics


def run_mining_for_logic(
    data: pl.DataFrame,
    logic_name: str,
    logic: WikiLogicStructured,
    max_rounds: int = 2,
) -> dict:
    """为单个逻辑运行挖掘"""
    print(f"\n{'=' * 60}")
    print(f"挖掘逻辑: {logic_name}")
    print(f"{'=' * 60}")

    gamma = compile_to_constraint(logic, source_logic=logic_name)
    print(f"Γ 约束:")
    print(f"  算子: {sorted(gamma.operator_whitelist)}")
    print(f"  变量: {sorted(gamma.variable_whitelist)}")
    print(f"  符号: {gamma.sign_constraint}")

    config = PipelineConfig(
        objective=f"Apply logic {logic_name}",
        termination=TerminationConfig(
            max_rounds=max_rounds,
            target_factors=5,
            early_stopping=True,
            patience=2,
        ),
        alphagpt_iterations=2,
        alphagpt_pool_size=5,
        mcts_iterations=20,
        mcts_max_depth=4,
        llm_provider="minimax",
        output_dir=str(OUTPUT_DIR / logic_name),
        gamma=gamma,
        structured_logic=logic,
    )

    start = time.time()
    pipeline = AlphaPipeline(config)
    result = pipeline.run(data)
    elapsed = time.time() - start

    summary = {
        "logic_name": logic_name,
        "total_rounds": len(result.rounds),
        "final_factors": len(result.final_pool),
        "elapsed_seconds": elapsed,
        "wiki_pages": result.wiki_pages,
        "best_ir": max(
            (m.ir for m in result.final_pool), default=0.0
        ),
        "avg_ir": (
            sum(m.ir for m in result.final_pool) / len(result.final_pool)
            if result.final_pool else 0.0
        ),
        "factors": [
            {
                "formula_id": m.formula_id,
                "ir": m.ir,
                "ic_mean": m.ic_mean,
            }
            for m in result.final_pool
        ],
    }

    print(f"\n结果:")
    print(f"  总轮次: {summary['total_rounds']}")
    print(f"  最终因子: {summary['final_factors']}")
    print(f"  最佳 IR: {summary['best_ir']:.4f}")
    print(f"  平均 IR: {summary['avg_ir']:.4f}")
    print(f"  Wiki 页面: {len(summary['wiki_pages'])}")
    print(f"  耗时: {elapsed:.1f}s")

    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("多逻辑因子挖掘 (Multi-Logic Factor Mining)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")

    print(f"\n加载数据: {DATA_PATH}")
    data = pl.read_parquet(DATA_PATH)
    print(f"  Rows: {len(data)}, Stocks: {data['code'].n_unique()}")

    logics = define_logics()
    print(f"\n待挖掘逻辑: {len(logics)} 个")
    for lg in logics:
        print(f"  - {lg['name']}")

    all_results = []
    total_start = time.time()

    for lg in logics:
        try:
            result = run_mining_for_logic(
                data, lg["name"], lg["logic"], max_rounds=2
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ {lg['name']} 失败: {e}")
            all_results.append({
                "logic_name": lg["name"],
                "error": str(e),
                "final_factors": 0,
            })

    total_elapsed = time.time() - total_start

    # 汇总
    print("\n" + "=" * 60)
    print("挖掘汇总 (Mining Summary)")
    print("=" * 60)

    total_factors = sum(r.get("final_factors", 0) for r in all_results)
    total_wiki = sum(len(r.get("wiki_pages", [])) for r in all_results)

    summary = {
        "total_logics": len(logics),
        "total_factors": total_factors,
        "total_wiki_pages": total_wiki,
        "total_elapsed_seconds": total_elapsed,
        "results": all_results,
    }

    for r in all_results:
        if "error" in r:
            print(f"  ✗ {r['logic_name']}: {r['error']}")
        else:
            print(f"  ✓ {r['logic_name']}: "
                  f"{r['final_factors']} 因子, "
                  f"best_ir={r['best_ir']:.4f}, "
                  f"{r['elapsed_seconds']:.1f}s")

    print(f"\n总因子: {total_factors}")
    print(f"总 Wiki 页面: {total_wiki}")
    print(f"总耗时: {total_elapsed:.1f}s")

    # 保存汇总
    summary_file = OUTPUT_DIR / "mining_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总保存到: {summary_file}")


if __name__ == "__main__":
    main()
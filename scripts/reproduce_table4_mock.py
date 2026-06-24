#!/usr/bin/env python3.11
# coding=utf-8
"""
reproduce_table4_mock.py - Stage 1 mock Table 4 复现 CLI

用法::

    python3.11 scripts/reproduce_table4_mock.py --n-stocks 100 --n-days 200
    python3.11 scripts/reproduce_table4_mock.py --quick   # 50 票 × 100 日 (smoke test)
    python3.11 scripts/reproduce_table4_mock.py --full    # 500 票 × 500 日 (~5 分钟)

Stage 2 替换 loader/baselines 即可，runner 无需修改。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 把项目根加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QuantNodes.research.quant_alpha.evaluation import (
    MockDataLoader,
    MockTable4Runner,
    PolarsAlphaCalculatorEvaluator,
)
from QuantNodes.research.quant_alpha.evaluation.baselines import (
    G1Handcrafted,
    G2LlmOnly,
    G3AlphaGpt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 mock Table 4 复现 (G1/G2/G3 对比)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="快速 smoke test (50 票 × 100 日)"
    )
    parser.add_argument(
        "--full", action="store_true", help="完整 mock (500 票 × 500 日)"
    )
    parser.add_argument("--n-stocks", type=int, default=100)
    parser.add_argument("--n-days", type=int, default=200)
    parser.add_argument("--g1-n", type=int, default=100, help="G1 公式数量")
    parser.add_argument("--g2-n", type=int, default=50, help="G2 公式数量")
    parser.add_argument("--g3-n", type=int, default=30, help="G3 公式数量")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "output" / "table4_mock",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 配置 logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("table4_mock")

    # 参数归一化
    if args.quick:
        args.n_stocks, args.n_days = 50, 100
        args.g1_n, args.g2_n, args.g3_n = 10, 5, 5
    elif args.full:
        args.n_stocks, args.n_days = 500, 500
        args.g1_n, args.g2_n, args.g3_n = 100, 50, 30

    logger.info(
        "Stage 1 mock Table 4: %d 票 × %d 日, G1=%d / G2=%d / G3=%d",
        args.n_stocks,
        args.n_days,
        args.g1_n,
        args.g2_n,
        args.g3_n,
    )

    # 构造 pipeline
    loader = MockDataLoader(
        n_stocks=args.n_stocks,
        n_days=args.n_days,
        seed=args.seed,
    )
    evaluator = PolarsAlphaCalculatorEvaluator()

    baselines = [
        G1Handcrafted(n=args.g1_n, seed=args.seed),
        G2LlmOnly(n=args.g2_n, seed=args.seed + 1),
        G3AlphaGpt(
            n=args.g3_n,
            iterations=3 if not args.quick else 1,
            pool_size=5 if args.quick else 10,
            seed=args.seed + 2,
        ),
    ]

    runner = MockTable4Runner(
        loader=loader,
        evaluator=evaluator,
        baselines=baselines,
        output_dir=args.output_dir,
        stage="mock",
        notes=[
            f"数据规模: {args.n_stocks} stocks × {args.n_days} days",
            f"Baseline 数量: {len(baselines)} (G1/G2/G3)",
            "Stage 1 mock, 无真实数据 / 真实 LLM",
        ],
    )

    try:
        report = runner.run()
    except Exception as e:
        logger.error("Pipeline 失败: %s", e, exc_info=True)
        return 1

    # 输出汇总
    ranked = report.rank_groups_by_ir()
    print("\n" + "=" * 80)
    print("Stage 1 Mock Table 4 — 3 组 baseline 对比")
    print("=" * 80)
    for i, g in enumerate(ranked, 1):
        print(
            f"{i}. {g.group_name}: avg_IR={g.avg_ir:.4f} | "
            f"avg_IC={g.avg_ic:.4f} | "
            f"success={g.success_count}/{len(g.factors)} | "
            f"elapsed={g.elapsed_sec:.1f}s"
        )
    print("=" * 80)

    # 期望趋势：G3 ≥ G1 > G2
    g3_rank = next(i for i, g in enumerate(ranked, 1) if g.group_name == "G3_AlphaGpt")
    g1_rank = next(i for i, g in enumerate(ranked, 1) if g.group_name == "G1_Handcrafted")
    g2_rank = next(i for i, g in enumerate(ranked, 1) if g.group_name == "G2_LlmOnly")
    print(
        f"排名: G3={g3_rank}, G1={g1_rank}, G2={g2_rank} | "
        f"期望 G3 ≥ G1 > G2 (mock 阶段)"
    )

    print(f"\n详细报告已保存到: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
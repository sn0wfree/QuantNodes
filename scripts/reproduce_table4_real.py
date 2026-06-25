#!/usr/bin/env python3.11
# coding=utf-8
"""
reproduce_table4_real.py - Stage 2 real Table 4 复现 CLI

从 ClickHouse 加载真实 A 股数据，用 MiniMax LLM 生成公式，
跑 G1/G2/G3 三组对比，验证 Table 4 趋势 (G3 > G1 > G2)。

用法::

    python3.11 scripts/reproduce_table4_real.py
    python3.11 scripts/reproduce_table4_real.py --table quote.stock_quote
    python3.11 scripts/reproduce_table4_real.py --g1-n 100 --g2-n 50 --g3-n 30
    python3.11 scripts/reproduce_table4_real.py --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QuantNodes.research.quant_alpha.evaluation import (
    ClickHouseDataLoader,
    PolarsAlphaCalculatorEvaluator,
    RealTable4Runner,
)
from QuantNodes.research.quant_alpha.evaluation.baselines import (
    G1Handcrafted,
    G2LlmOnly,
    G3AlphaGpt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 real Table 4 复现 (ClickHouse + MiniMax LLM)"
    )
    parser.add_argument(
        "--table", type=str, default="quote.stock_quote",
        help="ClickHouse 表名 (default: quote.stock_quote)",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--user", type=str, default="data")
    parser.add_argument("--password", type=str, default="123456")
    parser.add_argument("--database", type=str, default="quote")
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--g1-n", type=int, default=100)
    parser.add_argument("--g2-n", type=int, default=50)
    parser.add_argument("--g3-n", type=int, default=30)
    parser.add_argument("--g3-iterations", type=int, default=3)
    parser.add_argument("--g3-pool-size", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=str, default="data/output/table4_real",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="不使用本地 parquet 缓存",
    )
    parser.add_argument(
        "--cache-parquet", type=str, default="data/cache/full_a_2019_2024.parquet",
        help="本地 parquet 缓存路径",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # 1. DataLoader (ClickHouse)
    loader = ClickHouseDataLoader(
        table=args.table,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        start_date=args.start,
        end_date=args.end,
        cache_parquet=None if args.no_cache else args.cache_parquet,
    )

    # 2. Evaluator (复用 Stage 1)
    evaluator = PolarsAlphaCalculatorEvaluator()

    # 3. Baselines
    # G2/G3 默认使用 LLMGateway (自动路由到 MiniMax)
    g1 = G1Handcrafted(n=args.g1_n)
    g2 = G2LlmOnly(n=args.g2_n)
    g3 = G3AlphaGpt(
        n=args.g3_n,
        iterations=args.g3_iterations,
        pool_size=args.g3_pool_size,
    )

    # 4. Runner
    runner = RealTable4Runner(
        loader=loader,
        evaluator=evaluator,
        baselines=[g1, g2, g3],
        output_dir=Path(args.output_dir),
    )

    # 5. 执行
    print(f"\n{'='*60}")
    print(f"Stage 2 Real Table 4 复现")
    print(f"数据源: ClickHouse {args.table} ({args.start} ~ {args.end})")
    print(f"LLM: MiniMax (via LLMGateway)")
    print(f"G1={args.g1_n} 公式, G2={args.g2_n} 公式, G3={args.g3_n} 公式")
    print(f"{'='*60}\n")

    report = runner.run()

    # 6. 输出摘要
    print(f"\n{'='*60}")
    print(f"Stage 2 Real Table 4 结果")
    print(f"{'='*60}")
    for g in report.groups:
        print(
            f"  {g.group_name:20s} | "
            f"N={len(g.factors):4d} | "
            f"Success={g.success_count:4d} | "
            f"avg_IC={g.avg_ic:.4f} | "
            f"avg_IR={g.avg_ir:.4f} | "
            f"best_IR={g.best_ir:.4f}"
        )

    ranked = report.rank_groups_by_ir()
    print(f"\n排名 (avg_IR):")
    for i, g in enumerate(ranked, 1):
        print(f"  {i}. {g.group_name} (avg_IR={g.avg_ir:.4f})")

    # 验证趋势
    if len(ranked) >= 3:
        g3_ir = ranked[0].avg_ir
        g1_ir = ranked[1].avg_ir
        g2_ir = ranked[2].avg_ir
        if g3_ir > g1_ir > g2_ir:
            print(f"\n✅ 趋势验证: G3 ({g3_ir:.4f}) > G1 ({g1_ir:.4f}) > G2 ({g2_ir:.4f})")
        else:
            print(f"\n⚠️  趋势未达预期: G3={g3_ir:.4f}, G1={g1_ir:.4f}, G2={g2_ir:.4f}")

    print(f"\n报告已保存到: {args.output_dir}/")


if __name__ == "__main__":
    main()

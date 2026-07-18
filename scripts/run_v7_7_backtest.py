#!/usr/bin/env python3
# coding=utf-8
"""v7.7 Phase 1: PyCaret compare_models 模型筛选.

用法:
  python scripts/run_v7_7_backtest.py --compare              # 默认 raw 标签
  python scripts/run_v7_7_backtest.py --compare --target rank # rank 标签
  python scripts/run_v7_7_backtest.py --compare --sample 5000 # 采样加速
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.pycaret_estimator import (
    phase1_compare_models,
    phase1_get_top_models,
    CANDIDATE_MODELS,
)


def main():
    parser = argparse.ArgumentParser(description="v7.7 Phase 1: PyCaret 模型筛选")
    parser.add_argument("--compare", action="store_true", help="运行 compare_models")
    parser.add_argument("--target", default="raw", choices=["raw", "rank"], help="标签类型")
    parser.add_argument("--sample", type=int, default=None, help="采样大小 (None=全量)")
    parser.add_argument("--n-select", type=int, default=5, help="选择前 K 个模型")
    args = parser.parse_args()

    if not args.compare:
        print("请加 --compare 参数")
        return

    target_col = f"target_{args.target}"

    print("=" * 60)
    print("v7.7 Phase 1: PyCaret compare_models 模型筛选")
    print("=" * 60)
    print(f"标签: {target_col}")
    print(f"候选模型: {CANDIDATE_MODELS}")
    if args.sample:
        print(f"采样: {args.sample}")
    print()

    t0 = time.time()
    results_df = phase1_compare_models(
        target_col=target_col,
        n_select=args.n_select,
        sample_size=args.sample,
        verbose=True,
    )
    elapsed = time.time() - t0

    # 提取 top-N 模型 ID
    top_ids = phase1_get_top_models(results_df, n=args.n_select)
    print(f"\nTop-{args.n_select} 模型 ID: {top_ids}")
    print(f"耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

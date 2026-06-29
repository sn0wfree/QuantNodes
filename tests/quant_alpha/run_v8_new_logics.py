#!/usr/bin/env python3
# coding=utf-8
"""
run_v8_new_logics.py - 只跑 V8 剩余 2 个新 logic (trend_breakout, intraday_reversal)

V8 主脚本 6 logic 全跑 ~15min, 在 trend_breakout idea-gen 后被中断。
已完成 4 个老 logic (mr/mom/pvd/vol), 2 个新 logic 待跑。

设置与 run_6_logic_v8.py 完全一致, 确保可比。
"""

import json
import os
import time
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from run_6_logic_v8 import (
    define_logics, run_for_logic, DATA_PATH, OUTPUT_DIR,
)

# 只跑新 logic
NEW_LOGIC_NAMES = {"trend_breakout", "intraday_reversal"}


def main():
    print("=" * 60)
    print("V8 (continued) - 2 new logics (trend_breakout, intraday_reversal)")
    print("=" * 60)

    if not os.environ.get("QUANTNODES__LLM__API_KEY"):
        print("ERROR: QUANTNODES__LLM__API_KEY not set")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

    print(f"\n加载数据: {DATA_PATH}")
    data = pl.read_parquet(DATA_PATH)
    print(f"  Rows: {len(data)}, Stocks: {data['code'].n_unique()}")

    logics = [lg for lg in define_logics() if lg["name"] in NEW_LOGIC_NAMES]
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
    print("V8 (continued) 汇总")
    print("=" * 60)

    for r in all_results:
        if "error" in r:
            print(f"  ✗ {r['logic_name']}: {r['error']}")
        else:
            print(f"  ✓ {r['logic_name']}: "
                  f"{r['final_factors']} 因子, "
                  f"best_|IR|={r['best_abs_ir']:.4f}, "
                  f"{r['elapsed_seconds']:.1f}s")

    print(f"\n总耗时: {total_elapsed:.1f}s")

    # 追加到 v8_summary.json (如果存在)
    summary_file = OUTPUT_DIR / "v8_summary.json"
    if summary_file.exists():
        existing = json.loads(summary_file.read_text(encoding="utf-8"))
        existing_runs = {r["logic_name"]: r for r in existing.get("results", [])}
        for r in all_results:
            existing_runs[r["logic_name"]] = r
        existing["results"] = list(existing_runs.values())
        existing["total_factors"] = sum(
            r.get("final_factors", 0) for r in existing["results"]
        )
        existing["best_abs_ir_overall"] = max(
            (r.get("best_abs_ir", 0.0) for r in existing["results"]), default=0.0
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n汇总已追加到: {summary_file}")
        print(f"  总因子 (all 6 logic): {existing['total_factors']}")
        print(f"  最佳 |IR| (all 6): {existing['best_abs_ir_overall']:.4f}")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
mine_logics_demo.py - 离线演示脚本 (v3.0.2)

演示 mine_logic_library_v2 的完整流程:
1. 构造 NullLLMClient (离线, 无 API 调用)
2. 并发挖掘 alpha101 + alpha191
3. 构建 FactorPool 并保存 JSON
4. 生成 MetricsReportBuilder 并输出 MD + JSON

运行:
    python examples/mine_logics_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from QuantNodes.research.quant_alpha.logic_mining.batch import (
    ThreadSafeMetrics,
    mine_logic_library_v2,
)
from QuantNodes.research.quant_alpha.logic_mining.report import MetricsReportBuilder


def main() -> int:
    print("=" * 60)
    print("  QuantNodes Automated Factor Mining Demo (Offline)")
    print("=" * 60)
    print()

    # 1. 运行批量挖掘 (NullLLMClient, 无 API 调用)
    print("[1/3] Running mine_logic_library_v2 (offline, 2 libs, 2 workers)...")
    batch = mine_logic_library_v2(
        source_libs=["alpha101", "alpha191"],
        max_per_lib=3,
        workers=2,
        wiki_path="/tmp/wiki_demo",
        skip_existing=True,
    )

    print(f"  Done in {batch.wall_clock_s:.2f}s")
    print(f"  Attempted: {len(batch.attempted_ids)}")
    print(f"  Mined:     {batch.n_mined}")
    print(f"  Skipped:   {batch.n_skipped}")
    print(f"  Failed:    {batch.n_failed}")
    print()

    if batch.n_mined == 0:
        print("  No logics mined. Check if LLM client is available.")
        print("  (Offline mode uses NullLLMClient which returns mock data)")
        print()

    # 2. 池操作
    print("[2/3] Pool operations...")
    pool = batch.pool
    print(f"  Pool size: {len(pool)}")
    top = pool.select(top_n=3, by="ir")
    print(f"  Top 3 by IR:")
    for e in top:
        print(f"    {e.formula_id}: ir={e.ir:.4f}, lib={e.source_lib}")
    print()

    # 保存池到 JSON
    pool_json = Path("/tmp/mine_demo_pool.json")
    pool.save_json(pool_json)
    print(f"  Pool saved to {pool_json}")

    # 3. 生成报告
    print("[3/3] Generating report...")
    report = MetricsReportBuilder.from_batch(batch)
    report.to_json(Path("/tmp/mine_demo_report.json"))
    md = report.to_markdown()
    Path("/tmp/mine_demo_report.md").write_text(md, encoding="utf-8")
    print(f"  JSON: /tmp/mine_demo_report.json")
    print(f"  MD:   /tmp/mine_demo_report.md")
    print()

    # 打印 Markdown 报告预览
    print("-" * 60)
    print("Report preview (first 30 lines):")
    print("-" * 60)
    for line in md.split("\n")[:30]:
        print(line)
    if len(md.split("\n")) > 30:
        print("...")
    print()

    print("Demo complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
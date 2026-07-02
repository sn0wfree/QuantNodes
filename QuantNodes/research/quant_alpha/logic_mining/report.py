# coding=utf-8
"""
report.py - 离线指标报告生成器 (v3.0.2 Step 3)

提供:
- MetricsReportBuilder: 将 PipelineMetrics + batch 结果汇总为 JSON / Markdown
- 输出: metrics-{ts}.json + metrics-{ts}.md

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.report import MetricsReportBuilder
    from QuantNodes.research.quant_alpha.logic_mining.batch import (
        mine_logic_library_v2, ThreadSafeMetrics,
    )
    batch = mine_logic_library_v2(source_libs=["alpha101"], max_per_lib=5, ...)
    report = MetricsReportBuilder.from_batch(batch)
    report.to_json(path / "metrics.json")
    report.to_markdown(path / "metrics.md")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = ["MetricsReportBuilder"]


@dataclass
class MetricsReportBuilder:
    """从 batch 结果生成 JSON + Markdown 报告

    Attributes:
        total_attempted:  尝试挖掘总数
        total_mined:      成功提取 (structured_logic is not None)
        total_skipped:    跳过 (wiki 已存在)
        total_failed:     失败 (LLM/parse/structured 异常)
        wall_clock_s:     总耗时
        warnings:         警告列表
        agent_stats:      per-agent 统计 (call_failures / parse_failures / parse_layer_reached)
        failed_ids:       失败的 formula_id 列表
        source_lib_breakdown: 按 source_lib 分组的成功/尝试计数
        generated_at:     报告生成时间 (ISO 8601)
    """
    total_attempted: int = 0
    total_mined: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    wall_clock_s: float = 0.0
    warnings: List[str] = field(default_factory=list)
    agent_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed_ids: List[Dict[str, str]] = field(default_factory=list)
    source_lib_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @classmethod
    def from_batch(cls, batch: Any) -> "MetricsReportBuilder":
        """从 LogicMiningBatchResult 构建报告"""
        metrics_dict = batch.metrics.to_dict() if batch.metrics else {}
        pool_summary = batch.pool.summary() if batch.pool else {}

        # 按 source_lib 统计
        lib_breakdown: Dict[str, Dict[str, int]] = {}
        if batch.pool:
            for entry in batch.pool.values():
                lib = entry.source_lib
                if lib not in lib_breakdown:
                    lib_breakdown[lib] = {"mined": 0, "attempted": 0}
                lib_breakdown[lib]["mined"] += 1
            for fid in batch.attempted_ids:
                lib = fid.split("-")[0] if "-" in fid else fid
                if lib not in lib_breakdown:
                    lib_breakdown[lib] = {"mined": 0, "attempted": 0}
                lib_breakdown[lib]["attempted"] += 1

        agent_stats = _extract_agent_stats(metrics_dict)

        return cls(
            total_attempted=len(batch.attempted_ids),
            total_mined=batch.n_mined,
            total_skipped=batch.n_skipped,
            total_failed=batch.n_failed,
            wall_clock_s=batch.wall_clock_s,
            warnings=list(batch.warnings),
            agent_stats=agent_stats,
            failed_ids=[{"formula_id": fid, "error": err} for fid, err in batch.failed_ids],
            source_lib_breakdown=lib_breakdown,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 的字典"""
        return {
            "report": {
                "generated_at": self.generated_at,
                "wall_clock_s": round(self.wall_clock_s, 3),
            },
            "summary": {
                "total_attempted": self.total_attempted,
                "total_mined": self.total_mined,
                "total_skipped": self.total_skipped,
                "total_failed": self.total_failed,
                "success_rate": (
                    round(self.total_mined / self.total_attempted, 4)
                    if self.total_attempted > 0 else 0.0
                ),
            },
            "source_lib_breakdown": self.source_lib_breakdown,
            "agent_stats": self.agent_stats,
            "failed_ids": self.failed_ids,
            "warnings": self.warnings,
        }

    def to_json(self, path: Path) -> None:
        """写 JSON 报告"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Report written to %s", path)

    def to_markdown(self) -> str:
        """生成 Markdown 格式报告"""
        lines: List[str] = []
        lines.append("# Logic Mining Run Report")
        lines.append(f"\n**Generated**: {self.generated_at}")
        lines.append(f"**Wall clock**: {self.wall_clock_s:.2f}s\n")

        # Summary table
        lines.append("## Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Total attempted | {self.total_attempted} |")
        lines.append(f"| Successfully mined | {self.total_mined} |")
        lines.append(f"| Skipped (existing) | {self.total_skipped} |")
        lines.append(f"| Failed | {self.total_failed} |")
        if self.total_attempted > 0:
            rate = self.total_mined / self.total_attempted * 100
            lines.append(f"| Success rate | {rate:.1f}% |")
        lines.append("")

        # Source breakdown
        if self.source_lib_breakdown:
            lines.append("## Source Library Breakdown\n")
            lines.append("| Source Lib | Attempted | Mined |")
            lines.append("|---|---|---|")
            for lib, counts in sorted(self.source_lib_breakdown.items()):
                lines.append(f"| {lib} | {counts.get('attempted', 0)} | {counts.get('mined', 0)} |")
            lines.append("")

        # Agent stats
        if self.agent_stats:
            lines.append("## Agent Statistics\n")
            lines.append("| Agent | Call Failures | Parse Failures | Max Layer | Structured Failures |")
            lines.append("|---|---|---|---|---|")
            for agent_id, stats in sorted(self.agent_stats.items()):
                lines.append(
                    f"| {agent_id} "
                    f"| {stats.get('call_failures', 0)} "
                    f"| {stats.get('parse_failures', 0)} "
                    f"| {stats.get('parse_layer_reached', 0)} "
                    f"| {stats.get('structured_failures', 0)} |"
                )
            lines.append("")

        # Failed IDs
        if self.failed_ids:
            lines.append("## Failed Formulas\n")
            for item in self.failed_ids[:20]:  # 最多显示 20 条
                lines.append(f"- `{item['formula_id']}` — {item['error'][:80]}")
            if len(self.failed_ids) > 20:
                lines.append(f"- ... and {len(self.failed_ids) - 20} more")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings\n")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.append("---")
        lines.append("*Report generated by QuantNodes Logic Mining v3.0.2*")
        return "\n".join(lines)


def _extract_agent_stats(metrics_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """从 metrics.to_dict() 提取 per-agent 统计"""
    agent_ids: Set[str] = set()
    agent_ids.update(metrics_dict.get("call_failures", {}).keys())
    agent_ids.update(metrics_dict.get("parse_failures", {}).keys())
    agent_ids.update(metrics_dict.get("structured_failures", {}).keys())

    stats: Dict[str, Dict[str, Any]] = {}
    for agent_id in sorted(agent_ids):
        stats[agent_id] = {
            "call_failures": metrics_dict.get("call_failures", {}).get(agent_id, 0),
            "parse_failures": metrics_dict.get("parse_failures", {}).get(agent_id, 0),
            "parse_layer_reached": metrics_dict.get("parse_layer_reached", {}).get(agent_id, 0),
            "structured_failures": metrics_dict.get("structured_failures", {}).get(agent_id, 0),
        }
    return stats
# coding=utf-8
"""
runner.py - Table 4 复现主入口

串联 DataLoader + Baseline × 3 + Evaluator，输出 Table4Report。
支持 Stage 1（mock）与 Stage 2（real，仅替换 loader / llm_client）。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import (
    Baseline,
    DataLoader,
    Evaluator,
    Table4GroupResult,
    Table4Report,
    Table4Runner,
)

logger = logging.getLogger(__name__)

__all__ = ["MockTable4Runner", "RealTable4Runner"]


class MockTable4Runner(Table4Runner):
    """Stage 1 mock 主入口

    用法::

        from QuantNodes.research.quant_alpha.evaluation import (
            MockDataLoader, PolarsAlphaCalculatorEvaluator,
            G1Handcrafted, G2LlmOnly, G3AlphaGpt,
        )
        from QuantNodes.research.quant_alpha.evaluation.runner import MockTable4Runner

        runner = MockTable4Runner(
            loader=MockDataLoader(n_stocks=500, n_days=500),
            evaluator=PolarsAlphaCalculatorEvaluator(),
            baselines=[G1Handcrafted(n=100), G2LlmOnly(n=50), G3AlphaGpt(n=30)],
            output_dir=Path("data/output/table4_mock"),
        )
        report = runner.run()

    Stage 2 替换 loader/baselines 即可，runner 无需修改。
    """

    def __init__(
        self,
        loader: DataLoader,
        evaluator: Evaluator,
        baselines: Sequence[Baseline],
        output_dir: Optional[Path] = None,
        forward_returns: Optional[List[int]] = None,
        stage: str = "mock",
        notes: Optional[List[str]] = None,
    ) -> None:
        self.loader = loader
        self.evaluator = evaluator
        self.baselines = list(baselines)
        self.output_dir = Path(output_dir) if output_dir else None
        self.forward_returns = forward_returns or [1]
        self.stage = stage
        self.notes = list(notes or [])

    def run(self) -> Table4Report:
        """执行完整 pipeline"""
        logger.info("[Table4Runner:%s] 开始执行", self.stage)
        started_at = time.time()

        # 1. 加载数据
        t0 = time.time()
        data = self.loader.load()
        load_elapsed = time.time() - t0
        logger.info(
            "[Table4Runner] 数据加载完成: %s rows, %.2fs",
            f"{data.height:,}" if hasattr(data, "height") else len(data),
            load_elapsed,
        )

        # 2. 构造 report
        report = Table4Report(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            stage=self.stage,
            notes=self.notes,
        )

        # 3. 遍历 baselines
        for baseline in self.baselines:
            group_name = baseline.group_name
            logger.info("[Table4Runner] === %s 开始 ===", group_name)

            t0 = time.time()
            factors = baseline.generate_factors()
            gen_elapsed = time.time() - t0
            logger.info(
                "[Table4Runner] %s 生成 %d 个因子 (%.2fs)",
                group_name,
                len(factors),
                gen_elapsed,
            )

            t0 = time.time()
            metrics = self.evaluator.evaluate(
                factors, data, forward_returns=self.forward_returns
            )
            eval_elapsed = time.time() - t0
            logger.info(
                "[Table4Runner] %s 评估完成: %d success / %d failed (%.2fs)",
                group_name,
                sum(1 for m in metrics if m.status == "success"),
                sum(1 for m in metrics if m.status == "failed"),
                eval_elapsed,
            )

            group = Table4GroupResult(
                group_name=group_name,
                factors=factors,
                metrics=metrics,
                elapsed_sec=gen_elapsed + eval_elapsed,
            )
            report.add_group(group)

        total_elapsed = time.time() - started_at
        logger.info("[Table4Runner] 全部完成: %.2fs", total_elapsed)

        # 4. 输出
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.save_json(report, self.output_dir / "table4_report.json")
            self.save_markdown(report, self.output_dir / "table4_report.md")

        return report

    def save_json(self, report: Table4Report, path: Path) -> None:
        """保存为 JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("[Table4Runner] JSON saved to %s", path)

    def save_markdown(self, report: Table4Report, path: Path) -> None:
        """保存为 Markdown 报告"""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append(f"# Table 4 复现报告（{report.stage}）\n")
        lines.append(f"**生成时间**: {report.timestamp}\n")
        lines.append("\n## 汇总\n")
        lines.append("| Group | N | Success | Failed | avg IC | avg IR | best IR | elapsed(s) |\n")
        lines.append("|------|--:|--------:|-------:|-------:|-------:|--------:|-----------:|\n")
        for g in report.groups:
            lines.append(
                f"| {g.group_name} | {len(g.factors)} | {g.success_count} | "
                f"{g.failed_count} | {g.avg_ic:.4f} | {g.avg_ir:.4f} | "
                f"{g.best_ir:.4f} | {g.elapsed_sec:.2f} |\n"
            )

        # 排名
        ranked = report.rank_groups_by_ir()
        lines.append("\n## 按 avg_IR 排名\n")
        for i, g in enumerate(ranked, 1):
            lines.append(f"{i}. **{g.group_name}** — avg_IR = {g.avg_ir:.4f}\n")

        # 论文对比（Stage 2 填）
        if report.paper_comparison:
            lines.append("\n## 论文 Table 4 对比\n")
            lines.append("| Group | Ours avg_IR | Paper avg_IR | Diff |\n")
            lines.append("|------|------------:|-------------:|-----:|\n")
            for row in report.paper_comparison.get("rows", []):
                lines.append(
                    f"| {row['group']} | {row['ours']:.4f} | {row['paper']:.4f} | "
                    f"{row['diff']:.4f} |\n"
                )

        if report.notes:
            lines.append("\n## 备注\n")
            for n in report.notes:
                lines.append(f"- {n}\n")

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info("[Table4Runner] Markdown saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .mock_data_loader import MockDataLoader

    loader = MockDataLoader(n_stocks=100, n_days=200)
    print("Mock loader summary:", loader.load_summary())


class RealTable4Runner(MockTable4Runner):
    """Stage 2 real 主入口

    与 MockTable4Runner 完全相同的流程，
    仅 stage="real" 和默认输出目录不同。

    用法::

        from QuantNodes.research.quant_alpha.evaluation import (
            ClickHouseDataLoader, PolarsAlphaCalculatorEvaluator,
            G1Handcrafted, G2LlmOnly, G3AlphaGpt,
        )
        from QuantNodes.research.quant_alpha.evaluation.runner import RealTable4Runner

        runner = RealTable4Runner(
            loader=ClickHouseDataLoader(table="quote.stock_quote"),
            evaluator=PolarsAlphaCalculatorEvaluator(),
            baselines=[
                G1Handcrafted(n=100),
                G2LlmOnly(n=50),       # 默认使用 LLMGateway → MiniMax
                G3AlphaGpt(n=30),      # 默认使用 LLMGateway → MiniMax
            ],
            output_dir=Path("data/output/table4_real"),
        )
        report = runner.run()
    """

    def __init__(
        self,
        loader: DataLoader,
        evaluator: Evaluator,
        baselines: Sequence[Baseline],
        output_dir: Optional[Path] = None,
        forward_returns: Optional[List[int]] = None,
        notes: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            loader=loader,
            evaluator=evaluator,
            baselines=baselines,
            output_dir=output_dir or Path("data/output/table4_real"),
            forward_returns=forward_returns,
            stage="real",
            notes=notes,
        )
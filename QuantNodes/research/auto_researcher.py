# coding=utf-8
"""
AutoResearcher - 自动因子研究系统

编排因子挖掘、评估、去重、存储的完整流程。
支持3个阶段: 模板枚举 → MCTS搜索 → LLM增强
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import polars as pl

from QuantNodes.research.factor_miner import FactorMiner
from QuantNodes.research.factor_evaluator import (
    EvalConfig,
    FactorEvaluationResult,
    FactorEvaluator,
)
from QuantNodes.research.mcts_search import MCTSSearch
from QuantNodes.research.wiki import (
    FactorCategory,
    FactorSource,
    WikiFactor,
    WikiFactorProxy,
)


@dataclass
class AutoResearchResult:
    """自动挖掘结果"""
    valid_factors: List[FactorEvaluationResult] = field(default_factory=list)
    all_evaluated: List[FactorEvaluationResult] = field(default_factory=list)
    rejected_count: int = 0
    deduplicated_count: int = 0
    elapsed_seconds: float = 0.0
    report_markdown: str = ""


class AutoResearcher:
    """自动因子研究系统

    阶段1: 模板枚举 + 6维度评估 + 相关性去重
    阶段2: MCTS 搜索 (见 mcts_search.py)
    """

    def __init__(self, wiki_path: str):
        self.wiki_path = wiki_path
        self.proxy = WikiFactorProxy(wiki_path)
        self.miner = FactorMiner()
        self.evaluator = FactorEvaluator()

    def run(
        self,
        data: pl.DataFrame,
        eval_config: EvalConfig = None,
        date_column: str = "date",
        code_column: str = "code",
        forward_return_column: str = "forward_return",
        max_factors: int = 100,
        store_to_wiki: bool = True,
        use_mcts: bool = False,
        mcts_iterations: int = 50,
    ) -> AutoResearchResult:
        """执行完整挖掘流程

        Args:
            data: 行情数据 (需包含 date, code, close, vol, forward_return 等列)
            eval_config: 评估配置
            date_column: 日期列名
            code_column: 股票代码列名
            forward_return_column: 前瞻收益率列名
            max_factors: 最大候选因子数
            store_to_wiki: 是否存入 Wiki
            use_mcts: 是否使用 MCTS 搜索 (阶段2)
            mcts_iterations: MCTS 迭代次数

        Returns:
            AutoResearchResult
        """
        start = time.time()
        config = eval_config or EvalConfig()

        # 1. 获取可用列
        available_cols = [
            c for c in data.columns
            if c not in (date_column, code_column, forward_return_column)
        ]

        # 2. 阶段1: 模板枚举
        candidates = self.miner.generate(
            available_columns=available_cols,
            config=type("Cfg", (), {
                "max_factors": max_factors,
                "windows": [5, 10, 20, 60],
            })(),
        )

        evaluated: List[FactorEvaluationResult] = []
        existing_factors: List[pl.Series] = []

        for candidate in candidates:
            result = self.evaluator.evaluate(
                candidate=candidate,
                data=data,
                date_column=date_column,
                code_column=code_column,
                forward_return_column=forward_return_column,
                existing_factors=existing_factors if existing_factors else None,
            )
            evaluated.append(result)
            if result.is_valid and result.factor_values is not None:
                existing_factors.append(result.factor_values)

        # 3. 阶段2: MCTS 搜索 (可选)
        if use_mcts:
            mcts = MCTSSearch(evaluator=self.evaluator, eval_config=config)
            seed_formulas = [
                r.candidate.formula for r in evaluated
                if r.is_valid
            ][:10]
            mcts_results = mcts.search(
                data=data,
                seed_formulas=seed_formulas,
                iterations=mcts_iterations,
                date_column=date_column,
                code_column=code_column,
                forward_return_column=forward_return_column,
            )
            evaluated.extend(mcts_results)

        # 4. 筛选有效因子
        valid = [r for r in evaluated if r.is_valid]

        # 5. 相关性去重
        deduplicated = self.evaluator.deduplicate(valid, config.corr_threshold)

        # 6. 存入 Wiki
        if store_to_wiki:
            for result in deduplicated:
                self._store_to_wiki(result)

        # 7. 生成报告
        elapsed = time.time() - start
        report = self._generate_report(deduplicated, evaluated, elapsed)

        return AutoResearchResult(
            valid_factors=deduplicated,
            all_evaluated=evaluated,
            rejected_count=len(candidates) - len(valid),
            deduplicated_count=len(valid) - len(deduplicated),
            elapsed_seconds=elapsed,
            report_markdown=report,
        )

    def mine_single_factor(
        self,
        formula: str,
        data: pl.DataFrame,
        description: str = "",
        date_column: str = "date",
        code_column: str = "code",
        forward_return_column: str = "forward_return",
        store_to_wiki: bool = False,
    ) -> FactorEvaluationResult:
        """验证单个因子公式"""
        candidate = self.miner.generate_single(
            formula=formula,
            description=description,
            category=FactorCategory.OTHER,
        )

        result = self.evaluator.evaluate(
            candidate=candidate,
            data=data,
            date_column=date_column,
            code_column=code_column,
            forward_return_column=forward_return_column,
        )

        if result.is_valid and store_to_wiki:
            self._store_to_wiki(result)

        return result

    def _store_to_wiki(self, result: FactorEvaluationResult):
        """将验证通过的因子存入 Wiki"""
        factor = WikiFactor(
            name=result.candidate.name,
            formula=result.candidate.formula,
            source=FactorSource.AUTO_RESEARCH,
            category=result.candidate.category,
            tags=[result.candidate.template_name],
            ic_mean=result.ic_mean,
            ic_std=result.ic_std,
            icir=result.icir,
            rank_ic_mean=result.rank_ic_mean,
            turnover=result.turnover,
            metadata={
                "stability_score": result.stability_score,
                "monotonicity_score": result.monotonicity_score,
                "coverage": result.coverage,
                "overall_score": result.overall_score,
                "group_returns": result.group_returns,
            },
        )
        self.proxy.store_factor(factor)

    def _generate_report(
        self,
        valid: List[FactorEvaluationResult],
        all_evaluated: List[FactorEvaluationResult],
        elapsed: float,
    ) -> str:
        """生成 Markdown 挖掘报告"""
        lines = [
            "# AutoResearch 因子挖掘报告",
            "",
            f"**总耗时**: {elapsed:.1f}s",
            f"**候选因子数**: {len(all_evaluated)}",
            f"**通过验证**: {len(valid)}",
            f"**通过率**: {len(valid) / max(len(all_evaluated), 1) * 100:.1f}%",
            "",
            "## 通过验证的因子",
            "",
        ]

        if not valid:
            lines.append("无因子通过验证。")
        else:
            lines.append("| 因子 | IC | ICIR | 稳定性 | 单调性 | 换手率 | 评分 |")
            lines.append("|------|-----|------|--------|--------|--------|------|")
            for r in sorted(valid, key=lambda x: x.overall_score, reverse=True):
                lines.append(
                    f"| {r.candidate.name} "
                    f"| {r.ic_mean:.4f} "
                    f"| {r.icir:.4f} "
                    f"| {r.stability_score:.2f} "
                    f"| {r.monotonicity_score:.2f} "
                    f"| {r.turnover:.3f} "
                    f"| {r.overall_score:.3f} |"
                )

        lines.append("")
        lines.append("## 详细信息")
        lines.append("")

        for i, r in enumerate(valid[:10], 1):  # 最多显示10个
            lines.append(f"### {i}. {r.candidate.name}")
            lines.append(f"- **公式**: `{r.candidate.formula}`")
            lines.append(f"- **描述**: {r.candidate.description}")
            lines.append(f"- **模板**: {r.candidate.template_name}")
            lines.append(f"- **IC Mean**: {r.ic_mean:.4f}")
            lines.append(f"- **IC IR**: {r.icir:.4f}")
            lines.append(f"- **Rank IC**: {r.rank_ic_mean:.4f}")
            lines.append(f"- **稳定性**: {r.stability_score:.2f}")
            lines.append(f"- **单调性**: {r.monotonicity_score:.2f}")
            lines.append(f"- **换手率**: {r.turnover:.3f}")
            lines.append(f"- **覆盖率**: {r.coverage:.2f}")
            if r.group_returns:
                lines.append(f"- **分组收益**: {[f'{x:.4f}' for x in r.group_returns]}")
            lines.append("")

        return "\n".join(lines)

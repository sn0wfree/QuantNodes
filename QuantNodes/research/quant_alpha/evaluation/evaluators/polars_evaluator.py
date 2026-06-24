# coding=utf-8
"""
polars_evaluator.py - Stage 1/2 通用 Polars 评估器

内部包 alpha_evaluate tool（M5），把 polars 公式批量评估为 FactorMetrics 列表。
Stage 1 + Stage 2 共用此实现（不依赖 mock / real 数据）。

复用：
- QuantNodes.agent.tools.alpha_evaluate.AlphaEvaluateTool：M5 已实现评估逻辑
- contracts.FactorMetrics：统一输出 schema
- contracts.FactorSpec：输入因子列表
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..contracts import Evaluator, FactorMetrics, FactorSpec

logger = logging.getLogger(__name__)

__all__ = ["PolarsAlphaCalculatorEvaluator"]


class PolarsAlphaCalculatorEvaluator(Evaluator):
    """Stage 1/2 通用 Polars 评估器

    内部包 alpha_evaluate tool（M5），把 FactorSpec 列表转换为
    polars 公式字符串列表，调用 tool.execute()，再转换为 FactorMetrics。

    字段映射：
        FactorSpec.formula → tool.execute(formulas=[...])
        FactorSpec.formula_id → FactorMetrics.formula_id
        tool 返回 metrics dict → FactorMetrics.from_alpha_evaluate()
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._tool = None  # lazy init（避免 import 期 nanobot 影响）

    def _get_tool(self):
        """懒加载 alpha_evaluate tool（避免 nanobot import 副作用）"""
        if self._tool is None:
            try:
                from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool

                self._tool = AlphaEvaluateTool()
            except ImportError as e:
                logger.error(
                    "alpha_evaluate tool 不可用: %s；请安装 nanobot (pip install 'quantnodes[agent]')",
                    e,
                )
                raise
        return self._tool

    def evaluate(
        self,
        factors: List[FactorSpec],
        data: Any,
        forward_returns: Optional[List[int]] = None,
    ) -> List[FactorMetrics]:
        """批量评估因子

        Args:
            factors: FactorSpec 列表
            data: polars.DataFrame（含 date / code / OHLCV / industry 等列）
            forward_returns: 前瞻期列表（默认 [1]）

        Returns:
            FactorMetrics 列表（顺序与 factors 一一对应）
        """
        if not factors:
            return []

        tool = self._get_tool()
        formulas = [f.formula for f in factors]
        fr = forward_returns or [1]

        logger.info(
            "[PolarsAlphaCalculatorEvaluator] 评估 %d 个公式 (forward_returns=%s)",
            len(formulas),
            fr,
        )

        try:
            import asyncio

            coro = tool.execute(
                formulas=formulas,
                data=data,
                forward_returns=fr,
                max_workers=self.max_workers,
            )
            # 在 sync 上下文中运行 async tool（nanobot tool 本身就是 async）
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        result = ex.submit(asyncio.run, coro).result()
                else:
                    result = loop.run_until_complete(coro)
            except RuntimeError:
                # 无 event loop → 新建一个
                result = asyncio.run(coro)
        except Exception as e:
            logger.error("[PolarsAlphaCalculatorEvaluator] tool 执行失败: %s", e)
            return [
                FactorMetrics(
                    formula_id=f.formula_id,
                    status="failed",
                    error_msg=f"tool.execute failed: {e}",
                )
                for f in factors
            ]

        if not isinstance(result, dict):
            logger.error(
                "[PolarsAlphaCalculatorEvaluator] tool 返回非 dict: %s",
                type(result).__name__,
            )
            return [
                FactorMetrics(
                    formula_id=f.formula_id,
                    status="failed",
                    error_msg="tool.execute returned non-dict",
                )
                for f in factors
            ]

        # alpha_evaluate tool 返回结构：
        # {
        #   "status": "success" | "failed",
        #   "evaluations": [
        #       {"formula": "...", "status": "...", "metrics": {...}, "error_msg": ...},
        #       ...
        #   ],
        #   "summary": {...}
        # }
        evaluations = result.get("evaluations", [])
        if len(evaluations) != len(factors):
            logger.warning(
                "[PolarsAlphaCalculatorEvaluator] 评估数量不匹配: factors=%d, evaluations=%d",
                len(factors),
                len(evaluations),
            )

        out: List[FactorMetrics] = []
        for i, factor in enumerate(factors):
            if i < len(evaluations):
                eval_dict = evaluations[i]
                out.append(FactorMetrics.from_alpha_evaluate(factor.formula_id, eval_dict))
            else:
                out.append(
                    FactorMetrics(
                        formula_id=factor.formula_id,
                        status="failed",
                        error_msg="missing evaluation result",
                    )
                )

        n_success = sum(1 for m in out if m.status == "success")
        logger.info(
            "[PolarsAlphaCalculatorEvaluator] 完成: %d/%d success",
            n_success,
            len(out),
        )

        return out
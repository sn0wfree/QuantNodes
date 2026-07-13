# coding=utf-8
"""策略抗过拟合检验工具 (quant-validation skill 的 nanobot 入口).

调用方式: 通过 nanobot tool 协议, 传入策略参数 + 净值数据, 返回 markdown 报告.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from QuantNodes.agent.tools.base import Tool, ToolExecutionResult
from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    ValidationConfig,
    run_full_validation,
    performance_metrics,
    run_rotation_backtest,
)

logger = logging.getLogger(__name__)


class ValidationTool(Tool):
    """策略抗过拟合检验工具.

    4 个 action: validate_starting_points / validate_rebalance_offsets /
    validate_parameter_perturbation / ablation. 默认跑全 4 项并出报告.

    Parameters (JSON schema):
        etf_nav: list[dict]  # 每行 {date, code, close}  (or DataFrame.to_dict())
        lookback: int       # 默认 144
        top_n: int          # 默认 10
        actions: list[str]  # 子集 [start, rebal, perturb, ablation, all] (默认 ["all"])
        start_points: list[str]  # 起点列表 (默认 ["2018-01-01", "2020-01-01", "2022-01-01"])
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "quant_validation"

    @property
    def description(self) -> str:
        return (
            "对动量 ETF 轮动策略做抗过拟合检验: 起点依赖 / 调仓日偏移 / "
            "参数扰动 / 消融实验. 返回 markdown 报告与红黄绿结论."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "etf_nav": {
                    "type": "array",
                    "description": "ETF 净值数据, 形如 [{date, code, close}, ...]",
                    "items": {"type": "object"},
                },
                "lookback": {"type": "integer", "default": 144},
                "top_n": {"type": "integer", "default": 10},
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["all"],
                },
                "start_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["2018-01-01", "2020-01-01", "2022-01-01"],
                },
            },
            "required": ["etf_nav"],
        }

    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        try:
            etf_records: List[Dict[str, Any]] = kwargs.get("etf_nav", [])
            if not etf_records:
                return ToolExecutionResult(
                    tool_name=self.name, success=False,
                    content={}, error="etf_nav 为空",
                )
            # 还原 DataFrame
            df = pd.DataFrame(etf_records)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.pivot_table(index="date", columns="code", values="close").sort_index()
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.pivot_table(index="datetime", columns="code", values="close").sort_index()
            else:
                return ToolExecutionResult(
                    tool_name=self.name, success=False,
                    content={}, error="etf_nav 缺 date 列",
                )

            lookback = int(kwargs.get("lookback", 144))
            top_n = int(kwargs.get("top_n", 10))
            actions_req = kwargs.get("actions", ["all"])
            start_points = tuple(kwargs.get("start_points", ["2018-01-01", "2020-01-01", "2022-01-01"]))

            cfg = RotationConfig(
                lookback=lookback, top_n=top_n, min_history=lookback, corr_threshold=0.9,
                diversification=DiversificationCaps(
                    a_share_broad=2, a_share_sector=2, hk=1, a_share=3,
                    require_commodity=True, require_overseas=True,
                ),
            )
            vcfg = ValidationConfig(start_points=start_points)

            if "all" in actions_req:
                report = run_full_validation(df, DEFAULT_POOL, cfg, vcfg=vcfg)
                md = report.to_markdown()
                return ToolExecutionResult(
                    tool_name=self.name, success=True,
                    content={
                        "report_markdown": md,
                        "passed": report.passed,
                        "failed": report.failed,
                        "actions": [
                            {"name": a.name, "passed": bool(a.passed), "summary": a.summary}
                            for a in report.actions
                        ],
                    },
                )

            # 单个 action
            from QuantNodes.strategy.momentum_etf_rotation import (
                validate_starting_points,
                validate_rebalance_offsets,
                validate_parameter_perturbation,
                ablation,
            )
            action_map = {
                "start": validate_starting_points,
                "rebal": validate_rebalance_offsets,
                "perturb": validate_parameter_perturbation,
                "ablation": ablation,
            }
            results = []
            for a in actions_req:
                fn = action_map.get(a)
                if fn is None:
                    continue
                r = fn(df, DEFAULT_POOL, cfg, vcfg)
                results.append({"name": r.name, "passed": bool(r.passed), "summary": r.summary})
            return ToolExecutionResult(
                tool_name=self.name, success=True,
                content={"results": results},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("quant_validation 失败")
            return ToolExecutionResult(
                tool_name=self.name, success=False,
                content={}, error=str(exc),
            )

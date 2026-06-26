# coding=utf-8
"""WorkflowTool — nanobot Tool 子类，暴露 run_workflow 给 LLM。

Usage::

    from QuantNodes.agent.workflows.tool import WorkflowTool

    wt = WorkflowTool(llm_client=provider, model="minimax-M3")
    # 然后 registry.register(wt)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from QuantNodes.agent.tools.base import Tool
from .registry import REGISTRY
from .step_agent import StepAgent

logger = logging.getLogger(__name__)


def _update_state(state: Any, step_spec: Any, records: list[Any]) -> None:
    """根据 step_spec.state_output 更新 state。

    - 如果 state 字段是 list → extend (追加)
    - 否则 → setattr (覆盖)
    """
    if not step_spec.state_output:
        return
    target = getattr(state, step_spec.state_output, None)
    if target is None:
        return
    if isinstance(target, list):
        target.extend(records)
    else:
        setattr(state, step_spec.state_output, records)


class WorkflowTool(Tool):
    """nanobot Tool 子类，暴露 run_workflow 给 LLM。

    执行逻辑:
    1. 从 REGISTRY 查找 WorkflowSpec
    2. 构造 state
    3. 多轮循环执行 steps (支持 skip_on_last + prev_output 链式传递)
    4. 执行 final_steps
    5. result_builder 构建结果
    6. 存 JSON 文件 + 返回摘要
    """

    _scopes = {"core", "subagent"}

    def __init__(
        self,
        llm_client: Any = None,
        model: Optional[str] = None,
        results_dir: Optional[Path] = None,
    ) -> None:
        self._llm_client = llm_client
        self._model = model
        self._results_dir = results_dir or Path(".agent/results")

    @property
    def name(self) -> str:
        return "run_workflow"

    @property
    def description(self) -> str:
        return REGISTRY.build_llm_description()

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "The workflow name to execute.",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Workflow-specific configuration. "
                        "See workflow description for available parameters."
                    ),
                },
            },
            "required": ["workflow"],
        }

    async def execute(
        self,
        workflow: str,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """执行 workflow。

        Args:
            workflow: workflow 名称 (从 REGISTRY 查找)。
            config: workflow 配置参数。

        Returns:
            摘要 JSON 字符串。
        """
        spec = REGISTRY.get(workflow)
        if spec is None:
            available = [s["name"] for s in REGISTRY.list_all()]
            return json.dumps(
                {"status": "error", "message": f"Unknown workflow: {workflow!r}. Available: {available}"},
                ensure_ascii=False,
            )

        config = config or {}
        client = self._llm_client
        iterations = config.get("iterations", spec.iterations)

        logger.info("WorkflowTool: starting %r (%d iterations)", workflow, iterations)

        # 构造 state
        state = spec.state_factory()

        # 多轮循环
        for round_idx in range(1, iterations + 1):
            is_last = (round_idx == iterations)
            prev_output: Optional[list[Any]] = None

            for step_spec in spec.steps:
                if step_spec.skip_on_last and is_last:
                    logger.info("WorkflowTool: skipping %s on last round", step_spec.agent_id)
                    continue

                logger.info(
                    "WorkflowTool: round %d/%d step %s",
                    round_idx, iterations, step_spec.agent_id,
                )
                step = StepAgent(step_spec, llm_client=client)
                records = await asyncio.to_thread(
                    step.run,
                    state=state,
                    round_idx=round_idx,
                    prev_output=prev_output,
                    **config,
                )
                _update_state(state, step_spec, records)
                prev_output = records

        # 最终步骤
        for step_spec in spec.final_steps:
            logger.info("WorkflowTool: final step %s", step_spec.agent_id)
            step = StepAgent(step_spec, llm_client=client)
            records = await asyncio.to_thread(
                step.run,
                state=state,
                **config,
            )
            _update_state(state, step_spec, records)

        # 构建结果
        result = spec.result_builder(state, config)

        # 存完整 JSON
        try:
            self._results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_file = self._results_dir / f"{workflow}-{ts}.json"
            result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            result_file_str = str(result_file)
            logger.info("WorkflowTool: result saved to %s", result_file_str)
        except Exception as exc:
            logger.warning("WorkflowTool: failed to save result file: %s", exc)
            result_file_str = ""

        # 返回摘要
        summary = result.get("summary", {})
        top_formulas = result.get("final_pool", [])[:5]
        return json.dumps(
            {
                "status": "completed",
                "summary": summary,
                "result_file": result_file_str,
                "top_formulas": top_formulas,
            },
            ensure_ascii=False,
            default=str,
        )


__all__ = [
    "WorkflowTool",
    "_update_state",
]

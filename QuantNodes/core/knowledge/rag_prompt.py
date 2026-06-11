"""RAG Prompt — 从检索结果构造带上下文的 prompt。

用途: Hypothesizer / Mutator 在生成新因子前, 检索 top-k 历史因子作为 in-context 示例。

Week 8 升级: 谱系 RAG — 每个示例附 ancestors/descendants 上下文, 树状展开。
"""
from __future__ import annotations

from typing import Any, Optional

from ..trajectory import TrajectoryEntry, TrajectoryPool
from .knowledge_base import KnowledgeBase
from .lineage_expand import expand_lineage, expand_lineage_batch


_RAG_HEADER = """你是一个量化研究员, 负责基于历史经验和当前研究假设生成新 alpha 因子。
下方是历史表现良好的 {n_examples} 个因子作为参考, 含完整演化谱系 (ancestors / descendants):
"""

_RAG_EXAMPLE_TEMPLATE = """---
示例 {idx}: {name}
表达式: {expression}
描述: {description}
指标: sharpe={sharpe}, arr={arr}, ic_mean={ic_mean}
"""

_LINEAGE_RELATION_TEMPLATE = """{relation} (depth={depth}): {name} | sharpe={sharpe} | {expression}"""

_RAG_TASK_TEMPLATE = """
现在, 请基于以下研究假设生成新因子:
研究假设: {hypothesis}
补充描述: {description}

请综合参考示例的设计思路, 生成一个与历史不同但经济意义清晰的因子。
返回 JSON: {{"name": "因子名", "expression": "代码表达式", "description": "因子描述"}}
"""


def build_rag_prompt(
    direction: str,
    description: str,
    kb: Optional[KnowledgeBase] = None,
    top_k: int = 3,
    min_score: float = 0.01,
    include_lineage: bool = True,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 2,
) -> str:
    """构造带 RAG 上下文的 prompt (Week 8 升级版: 含谱系)。

    Args:
        direction: 研究方向 (hypothesis)
        description: 补充描述
        kb: KnowledgeBase (None=不附 RAG 上下文)
        top_k: 检索 top-k 数量
        min_score: 最小相似度阈值
        include_lineage: 是否附加祖先/后裔 (默认 True)
        max_ancestor_depth: 祖先展开深度
        max_descendant_depth: 后裔展开深度

    Returns:
        str: 完整 prompt (含 RAG 段 + 谱系 + 任务段)
    """
    rag_section = ""
    if kb is not None and len(kb) > 0:
        query = f"{direction} {description}"
        results = kb.query(query, top_k=top_k, min_score=min_score)
        if results:
            example_parts = []
            for i, (entry, score) in enumerate(results, 1):
                example_parts.append(_format_example(i, entry, score))
                if include_lineage and kb.pool is not None and entry is not None:
                    lineage_str = _format_lineage(
                        kb.pool, entry.entry_id,
                        max_ancestor_depth=max_ancestor_depth,
                        max_descendant_depth=max_descendant_depth,
                    )
                    if lineage_str:
                        example_parts.append(lineage_str)
            rag_section = (
                _RAG_HEADER.format(n_examples=len(results))
                + "\n".join(example_parts)
            )
    return rag_section + _RAG_TASK_TEMPLATE.format(
        hypothesis=direction, description=description,
    )


def _format_example(idx: int, entry: TrajectoryEntry | None, score: float) -> str:
    """格式化单个示例。"""
    if entry is None:
        return _RAG_EXAMPLE_TEMPLATE.format(
            idx=idx, name="<unknown>", expression="", description="",
            sharpe="?", arr="?", ic_mean="?",
        )
    cfg = entry.config_snapshot or {}
    factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
    metrics = entry.metrics or {}
    return _RAG_EXAMPLE_TEMPLATE.format(
        idx=idx,
        name=factor_cfg.get("name", entry.entry_id[:8]),
        expression=factor_cfg.get("expression", ""),
        description=factor_cfg.get("description", ""),
        sharpe=metrics.get("sharpe", "?"),
        arr=metrics.get("arr", "?"),
        ic_mean=metrics.get("ic_mean", "?"),
    )


def _format_lineage(
    pool: TrajectoryPool,
    root_id: str,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 2,
) -> str:
    """格式化谱系上下文, 嵌入示例后。"""
    expanded = expand_lineage(
        pool, root_id,
        max_ancestor_depth=max_ancestor_depth,
        max_descendant_depth=max_descendant_depth,
    )
    if not expanded["ancestors"] and not expanded["descendants"]:
        return ""

    parts: list[str] = []
    parts.append("  谱系上下文:")
    for depth, entry in expanded["ancestors"]:
        cfg = entry.config_snapshot or {}
        factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        parts.append("  " + _LINEAGE_RELATION_TEMPLATE.format(
            relation="↑ ancestor",
            depth=depth,
            name=factor_cfg.get("name", entry.entry_id[:8]),
            sharpe=(entry.metrics or {}).get("sharpe", "?"),
            expression=factor_cfg.get("expression", "")[:40],
        ))
    for depth, entry in expanded["descendants"]:
        cfg = entry.config_snapshot or {}
        factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        parts.append("  " + _LINEAGE_RELATION_TEMPLATE.format(
            relation="↓ descendant",
            depth=depth,
            name=factor_cfg.get("name", entry.entry_id[:8]),
            sharpe=(entry.metrics or {}).get("sharpe", "?"),
            expression=factor_cfg.get("expression", "")[:40],
        ))
    return "\n".join(parts)

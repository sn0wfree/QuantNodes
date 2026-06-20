"""RAG Prompt — 从检索结果构造带上下文的 prompt。

用途: Hypothesizer / Mutator 在生成新因子前, 检索 top-k 历史因子作为 in-context 示例。

Week 8 升级: 谱系 RAG — 每个示例附 ancestors/descendants 上下文, 树状展开。
Week 9 升级: 谱系压缩 — use_compress=True 时, 祖先/后裔段先用 LLM/启发式
            总结为 1 段简短描述, 减少 token 消耗。
"""
from __future__ import annotations

from typing import Optional

from ..trajectory import TrajectoryEntry, TrajectoryPool
from .knowledge_base import KnowledgeBase
from .lineage_compress import Compressor
from .lineage_expand import expand_lineage


_RAG_HEADER = """你是一个量化研究员, 负责基于历史经验和当前研究假设生成新 alpha 因子。
下方是历史表现良好的 {n_examples} 个因子作为参考, 含完整演化谱系 (ancestors / descendants):
"""

_RAG_EXAMPLE_TEMPLATE = """---
示例 {idx}: {name}
表达式: {expression}
描述: {description}
指标: sharpe={sharpe}, arr={arr}, ic_mean={ic_mean}
"""

_LINEAGE_RELATION_TEMPLATE = (
    "{relation} (depth={depth}): {name} | sharpe={sharpe} | {expression}"
)

_LINEAGE_COMPRESSED_TEMPLATE = """{relation} ({n} entries): {summary}"""

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
    use_compress: bool = False,
    compressor: Optional[Compressor] = None,
) -> str:
    """构造带 RAG 上下文的 prompt (Week 9 升级版: 含谱系 + 压缩)。

    Args:
        direction: 研究方向 (hypothesis)
        description: 补充描述
        kb: KnowledgeBase (None=不附 RAG 上下文)
        top_k: 检索 top-k 数量
        min_score: 最小相似度阈值
        include_lineage: 是否附加祖先/后裔
        max_ancestor_depth: 祖先展开深度
        max_descendant_depth: 后裔展开深度
        use_compress: 是否压缩谱系段 (默认 False, 沿用 Week 8 多行格式)
        compressor: Compressor 实例 (use_compress=True 时必填,
                    内部未传时自动构造 mock Compressor)

    Returns:
        str: 完整 prompt
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
                        use_compress=use_compress,
                        compressor=compressor,
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
    use_compress: bool = False,
    compressor: Optional[Compressor] = None,
) -> str:
    """格式化谱系上下文, 嵌入示例后。"""
    expanded = expand_lineage(
        pool, root_id,
        max_ancestor_depth=max_ancestor_depth,
        max_descendant_depth=max_descendant_depth,
    )
    ancestors = expanded["ancestors"]
    descendants = expanded["descendants"]

    if not ancestors and not descendants:
        return ""

    if use_compress:
        # Week 9: 压缩模式
        comp = compressor or Compressor(model="mock")
        return _format_lineage_compressed(ancestors, descendants, comp)

    # Week 8: 多行展开模式
    return _format_lineage_expanded(ancestors, descendants)


def _format_lineage_expanded(ancestors, descendants) -> str:
    """Week 8 风格: 多行展开。"""
    parts: list[str] = ["  谱系上下文:"]
    for depth, entry in ancestors:
        cfg = entry.config_snapshot or {}
        factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        parts.append("  " + _LINEAGE_RELATION_TEMPLATE.format(
            relation="↑ ancestor",
            depth=depth,
            name=factor_cfg.get("name", entry.entry_id[:8]),
            sharpe=(entry.metrics or {}).get("sharpe", "?"),
            expression=factor_cfg.get("expression", "")[:40],
        ))
    for depth, entry in descendants:
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


def _format_lineage_compressed(ancestors, descendants, compressor: Compressor) -> str:
    """Week 9 风格: 1 行总结 ancestors + 1 行总结 descendants。"""
    parts: list[str] = ["  谱系上下文 (压缩):"]
    if ancestors:
        c_anc = compressor.compress(ancestors, relation="ancestors")
        parts.append("  " + _LINEAGE_COMPRESSED_TEMPLATE.format(
            relation="↑ ancestors",
            n=c_anc.original_count,
            summary=c_anc.summary,
        ))
    if descendants:
        c_desc = compressor.compress(descendants, relation="descendants")
        parts.append("  " + _LINEAGE_COMPRESSED_TEMPLATE.format(
            relation="↓ descendants",
            n=c_desc.original_count,
            summary=c_desc.summary,
        ))
    return "\n".join(parts)

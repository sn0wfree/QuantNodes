"""谱系压缩 — LLM 总结祖先/后裔链为 1 段简短描述, 减少 token。

设计:
    - Compressor 类 (类似 LLMJudge 的协议)
    - compress_lineage(entries, relation) -> str
    - 默认 mock 模式: 启发式拼接 (1 行 / entry, 含 name + operation + sharpe)
    - 支持自定义 llm_callable (真实 LLM)
    - 限制输出 token: max_tokens (默认 200 chars)

用途:
    build_rag_prompt(..., use_compress=True) 时,
    每个示例的 ancestors / descendants 段先压缩为 1 行, 再注入 prompt。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional


# 单个 entry 的压缩模板 (heuristic 模式)
_HEURISTIC_ENTRY_TEMPLATE = "{name} ({op} r{round}, sharpe={sharpe:.2f})"


@dataclass
class CompressedLineage:
    """压缩结果。"""
    summary: str           # 1 段简短总结
    original_count: int    # 压缩前 entry 数
    compressed_chars: int  # 压缩后字符数
    method: str            # "llm" / "heuristic"


# ============================================================================
# Compressor
# ============================================================================

class Compressor:
    """谱系压缩器。

    Args:
        model: "mock" (默认启发式) / "deepseek-v3" / 其他 (需 llm_callable)
        max_tokens: 输出最大字符数 (默认 200)
        llm_callable: 自定义 LLM 调用函数, 接受 prompt 返回 string
    """

    def __init__(
        self,
        model: str = "mock",
        max_tokens: int = 200,
        llm_callable: Optional[Callable] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._llm_callable = llm_callable

    def compress(
        self,
        entries: list,  # list of (depth, TrajectoryEntry) 或 TrajectoryEntry
        relation: str = "ancestors",  # "ancestors" / "descendants"
    ) -> CompressedLineage:
        """压缩一组 entry 为 1 段简短总结。

        Args:
            entries: lineage expand 后的 (depth, entry) 列表
            relation: 关系类型 (ancestors / descendants), 仅用于 prompt 上下文

        Returns:
            CompressedLineage
        """
        # 规整化: (depth, entry)
        normalized: list[tuple[int, object]] = []
        for item in entries:
            if isinstance(item, tuple) and len(item) == 2:
                normalized.append(item)
            else:
                normalized.append((0, item))

        if not normalized:
            return CompressedLineage(
                summary="", original_count=0, compressed_chars=0, method="heuristic"
            )

        if self._llm_callable is not None:
            summary, method = self._llm_summarize(normalized, relation)
        elif self.model == "mock":
            summary = self._heuristic_summary(normalized, relation)
            method = "heuristic"
        else:
            raise NotImplementedError(
                f"真实 LLM '{self.model}' 未实现, 请提供 llm_callable 或使用 model='mock'"
            )

        # 截断到 max_tokens
        if len(summary) > self.max_tokens:
            summary = summary[: self.max_tokens - 3] + "..."

        return CompressedLineage(
            summary=summary,
            original_count=len(normalized),
            compressed_chars=len(summary),
            method=method,
        )

    # ------------------------------------------------------------------
    # 启发式 (无 LLM 依赖)
    # ------------------------------------------------------------------

    def _heuristic_summary(
        self,
        entries: list[tuple[int, object]],
        relation: str,
    ) -> str:
        """启发式: 每 entry 1 行, name + operation + sharpe。"""
        relation_arrow = "↑" if relation == "ancestors" else "↓"
        parts: list[str] = []
        for depth, entry in entries:
            cfg = (entry.config_snapshot or {}).get("factor", {})
            name = cfg.get("name", entry.entry_id[:8])
            sharpe = (entry.metrics or {}).get("sharpe", 0)
            entry_str = _HEURISTIC_ENTRY_TEMPLATE.format(
                name=name, op=entry.operation,
                round=entry.round_idx, sharpe=sharpe,
            )
            parts.append(f"{relation_arrow}d{depth} {entry_str}")
        return " ; ".join(parts)

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _llm_summarize(
        self,
        entries: list[tuple[int, object]],
        relation: str,
    ) -> tuple[str, str]:
        """真实 LLM 总结。Returns (summary, effective_method)."""
        prompt = self._build_prompt(entries, relation)
        raw = self._llm_callable(prompt)
        try:
            data = json.loads(raw)
            return str(data.get("summary", "")), "llm"
        except (json.JSONDecodeError, TypeError, KeyError):
            # 解析失败, fallback 到启发式
            return self._heuristic_summary(entries, relation), "heuristic"

    def _build_prompt(
        self,
        entries: list[tuple[int, object]],
        relation: str,
    ) -> str:
        """构造 LLM prompt。"""
        lines: list[str] = []
        for depth, entry in entries:
            cfg = (entry.config_snapshot or {}).get("factor", {})
            sharpe = (entry.metrics or {}).get("sharpe", 0)
            lines.append(
                f"  depth={depth}, name={cfg.get('name', entry.entry_id[:8])}, "
                f"op={entry.operation}, sharpe={sharpe}, "
                f"desc={cfg.get('description', '')}"
            )
        entries_text = "\n".join(lines)
        return (
            f"请总结以下 {len(entries)} 个 {relation} 的核心设计思路 (限 {self.max_tokens} 字):\n"
            f"{entries_text}\n\n"
            f"返回 JSON: {{\"summary\": \"你的总结\"}}"
        )


# ============================================================================
# 模块级便捷函数
# ============================================================================

def compress_lineage(
    entries,
    relation: str = "ancestors",
    model: str = "mock",
    max_tokens: int = 200,
    llm_callable: Optional[Callable] = None,
) -> CompressedLineage:
    """便捷函数: 一次性压缩。

    Args:
        entries: (depth, entry) 列表
        relation: "ancestors" / "descendants"
        model: 同 Compressor
        max_tokens: 同 Compressor
        llm_callable: 同 Compressor
    """
    c = Compressor(model=model, max_tokens=max_tokens, llm_callable=llm_callable)
    return c.compress(entries, relation=relation)

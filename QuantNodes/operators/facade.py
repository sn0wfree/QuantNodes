# coding=utf-8
"""
operators.facade - 算子统一查询门面 (Facade, Phase 3.2, 2026-06-22)

QuantNodes 有 3 个并列的算子注册表 (docs/26 §3.3):

    L0 内置 (_OPERATOR_REGISTRY)        factor_node/factor_functions
    L1 复合 DAG (_COMPOSITE_REGISTRY)   operators/composite_dag.py
    L2 自定义 (_CustomOperatorRegistry) operators/registry.py

调用方 (agent/config/executor.py, loader.py) 此前需分别 import:
    - get_operator / list_operators / operator_info  (L0, 内部已级联 L2)
    - is_composite_op / get_composite_spec / list_composite_ops  (L1, 完全隔离)
    - CustomOperator.get / _CustomOperatorRegistry.list  (L2)

本 Facade 提供**单一只读入口**统一这 3 层查询, 调用方只需:

    from QuantNodes.operators import operator_facade as ops
    fn = ops.resolve("ts_mean")          # L0/L2 callable
    spec = ops.get_composite(name)        # L1 CompositeSpec
    if ops.exists(name): ...
    ops.kind(name)  # -> "custom" / "builtin" / "composite" / None
    ops.list_all()  # 三层去重合并

设计要点 (Facade 模式):
  - **只委托, 不改行为**: 全部转调既有函数, 与旧 API bitwise 一致 (向后兼容)。
  - **只读**: 注册仍走各自的 @register_operator / @composite_operator /
    CustomOperator (写路径不收敛, 避免破坏隔离语义)。
  - L0 的 ``get_operator`` 本身已级联 L2 (custom→builtin), Facade 借此保持
    与旧查询完全一致的优先级; L1 因严格隔离 (见 composite_dag.py:137-146)
    单独暴露 ``get_composite`` / ``is_composite``。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from QuantNodes.operators.composite_dag import CompositeSpec


class OperatorFacade:
    """3 层算子注册表的统一只读门面.

    所有方法均委托给既有查询函数, 不缓存、不持有状态, 因此对运行时新增的
    算子注册 (custom / composite) 实时可见。
    """

    # ---- L0 + L2 (callable 级联查询) ----

    def resolve(self, name: str, category: Optional[str] = None) -> Optional[Callable]:
        """获取算子 callable (级联: 自定义 L2 → 内置 L0)。

        与 ``factor_functions.get_operator`` 行为完全一致。composite (L1) 不在
        此返回 (它们不是直接 callable), 请用 :meth:`get_composite`。
        """
        from QuantNodes.factor_node.factor_functions import get_operator

        return get_operator(name, category)

    def info(self, name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取算子详细信息 dict (级联: L2 → L0)。"""
        from QuantNodes.factor_node.factor_functions import operator_info

        return operator_info(name, category)

    # ---- L1 (composite, 隔离查询) ----

    def get_composite(self, name: str) -> Optional["CompositeSpec"]:
        """获取 composite DAG 的 ``CompositeSpec`` (L1, 与 L0/L2 隔离)。"""
        from QuantNodes.operators.composite_dag import get_composite_spec

        return get_composite_spec(name)

    def is_composite(self, name: str) -> bool:
        """判断 name 是否是 composite 算子 (L1)。"""
        from QuantNodes.operators.composite_dag import is_composite_op

        return is_composite_op(name)

    # ---- 跨层统一查询 ----

    def exists(self, name: str) -> bool:
        """name 是否在任一层注册 (callable 或 composite)。"""
        if self.is_composite(name):
            return True
        return self.resolve(name) is not None

    def kind(self, name: str) -> Optional[str]:
        """返回 name 所属层: ``"custom"`` / ``"builtin"`` / ``"composite"`` / ``None``.

        优先级与查询级联一致 (custom → builtin → composite)。同名时返回最先
        命中的层 (与 :meth:`resolve` 实际返回的实现保持一致)。
        """
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        if _CustomOperatorRegistry.get(name) is not None:
            return "custom"
        if self.resolve(name) is not None:
            return "builtin"
        if self.is_composite(name):
            return "composite"
        return None

    def list_all(
        self,
        category: Optional[str] = None,
        include_custom: bool = True,
        include_composite: bool = True,
    ) -> List[str]:
        """列出三层算子名 (去重, 保持 custom → builtin → composite 顺序)。

        Args:
            category: 限定类别 (point/time/section/multi_section/talib)。
            include_custom: 是否含 L2 自定义 (默认 True)。
            include_composite: 是否含 L1 composite (默认 True)。
        """
        from QuantNodes.factor_node.factor_functions import list_operators

        # list_operators 已做 custom(L2) + builtin(L0) 去重合并
        names = list(list_operators(category=category, include_custom=include_custom))

        if include_composite:
            from QuantNodes.operators.composite_dag import list_composite_ops

            seen = set(names)
            for name in list_composite_ops(category=category):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def documentation(
        self, output_format: str = "markdown", category: Optional[str] = None
    ) -> str:
        """生成 L0 内置算子文档 (委托 ``generate_documentation``)。

        composite (L1) 的 LLM 文档另有专用入口 :meth:`composite_doc_for_llm`。
        """
        from QuantNodes.factor_node.factor_functions import generate_documentation

        return generate_documentation(output_format=output_format, category=category)

    def composite_doc_for_llm(self) -> str:
        """生成 composite (L1) 给 LLM 的 markdown 文档。"""
        from QuantNodes.operators.composite_dag import get_composite_doc_for_llm

        return get_composite_doc_for_llm()

    def __repr__(self) -> str:
        return "<OperatorFacade L0+L1+L2 unified read-only>"


# 模块级单例 — 调用方统一从这里查询
operator_facade = OperatorFacade()

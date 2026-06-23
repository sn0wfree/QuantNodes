# coding=utf-8
"""OperatorFacade 测试 (Phase 3.2, 2026-06-22).

覆盖 3 层注册表 (L0 内置 / L1 composite / L2 自定义) 的统一只读门面:
    - resolve / info  (L0+L2 级联 callable)
    - get_composite / is_composite  (L1 隔离)
    - exists / kind / list_all  (跨层)
    - documentation / composite_doc_for_llm
    - 与既有 API bitwise 一致 (向后兼容: facade 只委托不改行为)
    - 自定义算子注册后实时可见 (无状态缓存)
"""
from __future__ import annotations

import pytest


@pytest.fixture
def ops():
    from QuantNodes.operators import operator_facade

    return operator_facade


@pytest.fixture
def clean_custom():
    """每个用例前后清空自定义注册表, 避免污染依赖 builtin 计数的测试。"""
    from QuantNodes.operators.registry import _CustomOperatorRegistry

    _CustomOperatorRegistry.unregister_all()
    yield
    _CustomOperatorRegistry.unregister_all()


# ============================================================================
# 1. import / 单例 (3 tests)
# ============================================================================

class TestFacadeImport:
    def test_singleton_importable(self):
        from QuantNodes.operators import operator_facade, OperatorFacade

        assert isinstance(operator_facade, OperatorFacade)

    def test_repr(self, ops):
        assert "OperatorFacade" in repr(ops)

    def test_in_all(self):
        import QuantNodes.operators as opmod

        assert "operator_facade" in opmod.__all__
        assert "OperatorFacade" in opmod.__all__


# ============================================================================
# 2. resolve / info — L0 builtin (4 tests)
# ============================================================================

class TestResolveBuiltin:
    def test_resolve_builtin(self, ops):
        assert ops.resolve("ts_mean") is not None

    def test_resolve_matches_get_operator(self, ops):
        from QuantNodes.factor_node.factor_functions import get_operator

        assert ops.resolve("ts_mean") is get_operator("ts_mean")

    def test_resolve_missing_returns_none(self, ops):
        assert ops.resolve("__no_such_op__") is None

    def test_info_matches_operator_info(self, ops):
        from QuantNodes.factor_node.factor_functions import operator_info

        assert ops.info("ts_mean") == operator_info("ts_mean")


# ============================================================================
# 3. composite — L1 (4 tests)
# ============================================================================

class TestComposite:
    def test_is_composite_true(self, ops):
        from QuantNodes.operators import list_composite_ops

        name = list_composite_ops()[0]
        assert ops.is_composite(name) is True

    def test_is_composite_false_for_builtin(self, ops):
        assert ops.is_composite("ts_mean") is False

    def test_get_composite_returns_spec(self, ops):
        from QuantNodes.operators import list_composite_ops, get_composite_spec

        name = list_composite_ops()[0]
        assert ops.get_composite(name) is get_composite_spec(name)

    def test_get_composite_none_for_builtin(self, ops):
        assert ops.get_composite("ts_mean") is None


# ============================================================================
# 4. exists / kind — 跨层 (6 tests)
# ============================================================================

class TestExistsKind:
    def test_exists_builtin(self, ops):
        assert ops.exists("ts_mean") is True

    def test_exists_composite(self, ops):
        from QuantNodes.operators import list_composite_ops

        assert ops.exists(list_composite_ops()[0]) is True

    def test_exists_missing(self, ops):
        assert ops.exists("__no_such_op__") is False

    def test_kind_builtin(self, ops):
        assert ops.kind("ts_mean") == "builtin"

    def test_kind_composite(self, ops):
        from QuantNodes.operators import list_composite_ops

        assert ops.kind(list_composite_ops()[0]) == "composite"

    def test_kind_missing(self, ops):
        assert ops.kind("__no_such_op__") is None


# ============================================================================
# 5. list_all — 三层合并 (4 tests)
# ============================================================================

class TestListAll:
    def test_list_all_includes_builtin(self, ops):
        assert "ts_mean" in ops.list_all()

    def test_list_all_includes_composite_by_default(self, ops):
        from QuantNodes.operators import list_composite_ops

        names = ops.list_all()
        for c in list_composite_ops():
            assert c in names

    def test_list_all_exclude_composite(self, ops):
        from QuantNodes.factor_node.factor_functions import list_operators

        assert ops.list_all(include_composite=False) == list(list_operators())

    def test_list_all_no_duplicates(self, ops):
        names = ops.list_all()
        assert len(names) == len(set(names))


# ============================================================================
# 6. documentation (2 tests)
# ============================================================================

class TestDocumentation:
    def test_documentation_matches(self, ops):
        from QuantNodes.factor_node.factor_functions import generate_documentation

        assert ops.documentation() == generate_documentation()

    def test_composite_doc_matches(self, ops):
        from QuantNodes.operators import get_composite_doc_for_llm

        assert ops.composite_doc_for_llm() == get_composite_doc_for_llm()


# ============================================================================
# 7. 自定义算子实时可见 + kind="custom" (3 tests)
# ============================================================================

class TestCustomVisibility:
    def test_custom_resolve_after_register(self, ops, clean_custom):
        from QuantNodes.operators import CustomOperator

        @CustomOperator.point("facade_custom_op")
        def _op(f):
            return f

        assert ops.resolve("facade_custom_op") is not None
        assert ops.exists("facade_custom_op") is True

    def test_custom_kind(self, ops, clean_custom):
        from QuantNodes.operators import CustomOperator

        @CustomOperator.point("facade_custom_kind_op")
        def _op(f):
            return f

        assert ops.kind("facade_custom_kind_op") == "custom"

    def test_custom_in_list_all(self, ops, clean_custom):
        from QuantNodes.operators import CustomOperator

        @CustomOperator.time("facade_custom_listed")
        def _op(f):
            return f

        assert "facade_custom_listed" in ops.list_all()

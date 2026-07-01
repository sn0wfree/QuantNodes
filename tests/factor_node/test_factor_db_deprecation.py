# coding=utf-8
"""
test_factor_db_deprecation.py

P2.12c.2: factor_node/factor_db.py 已标 DEPRECATED (v3.0+)。
本测试验证 deprecation 行为符合预期。

详见 `docs/33-data-access-architecture.md` 第 5 节。
"""

from __future__ import annotations

import warnings

import pytest


class TestFactorDBDeprecation:
    """factor_node.factor_db 模块 DEPRECATED 状态验证。"""

    def test_module_docstring_marks_deprecated(self):
        """模块 docstring 包含 DEPRECATED 标记。"""
        import QuantNodes.factor_node.factor_db as module

        doc = module.__doc__ or ""
        assert "DEPRECATED" in doc, (
            "factor_db module 应在 docstring 中标记 DEPRECATED (v3.0+)"
        )

    def test_factor_db_class_docstring_marks_deprecated(self):
        """FactorDB 类 docstring 标记 DEPRECATED。"""
        from QuantNodes.factor_node.factor_db import FactorDB

        doc = FactorDB.__doc__ or ""
        assert "DEPRECATED" in doc or "deprecated" in doc.lower(), (
            "FactorDB 类应在 docstring 中标记 deprecated"
        )

    def test_factor_db_still_creatable_backward_compat(self):
        """虽然 DEPRECATED, FactorDB 仍可创建 (向后兼容)。"""
        from QuantNodes.factor_node.factor_db import FactorDB

        db = FactorDB(name="legacy_db")
        assert db.name == "legacy_db"

    def test_writable_factor_db_still_creatable_backward_compat(self):
        """WritableFactorDB 也保持向后兼容。"""
        from QuantNodes.factor_node.factor_db import WritableFactorDB

        db = WritableFactorDB(name="legacy_writable")
        assert db.name == "legacy_writable"

    def test_legacy_methods_return_zero_or_none(self):
        """DEPRECATED 状态: 所有抽象方法返回 0 或 None (占位)。"""
        from QuantNodes.factor_node.factor_db import FactorDB

        db = FactorDB()
        # 这些是 v2.x 历史占位, v3.0 全部返回空值
        assert db.connect() == 0
        assert db.disconnect() == 0
        assert db.TableNames == []
        assert db.getTable("any") is None
        assert db.getID() == []
        assert db.getDateTime() == []


class TestFactorDBImportPath:
    """验证 import 路径稳定。"""

    def test_import_via_factor_node(self):
        """从 factor_node.__init__ re-export 路径仍可用。"""
        from QuantNodes.factor_node import FactorDB as FDB_v1

        assert FDB_v1.__name__ == "FactorDB"

    def test_import_direct_module(self):
        """从 factor_node.factor_db 直接 import 仍可用。"""
        from QuantNodes.factor_node.factor_db import FactorDB as FDB_v2

        assert FDB_v2.__name__ == "FactorDB"

    def test_both_paths_return_same_class(self):
        """两条 import 路径是同一个类 (向后兼容保证)。"""
        from QuantNodes.factor_node import FactorDB as FDB_v1
        from QuantNodes.factor_node.factor_db import FactorDB as FDB_v2

        assert FDB_v1 is FDB_v2


class TestFactorDBReplacementHints:
    """文档中提供的替代方案应可在代码中使用。"""

    def test_database_node_alternative_available(self):
        """替代方案 1: database_node.BaseDBNode 可用。"""
        from QuantNodes.database_node.base import BaseDBNode

        assert hasattr(BaseDBNode, "query")
        assert hasattr(BaseDBNode, "connect")

    def test_wiki_alternative_available(self):
        """替代方案 2: WikiFactorProxy 可用。"""
        from QuantNodes.research.wiki import WikiFactorProxy

        assert hasattr(WikiFactorProxy, "store_factor")
        assert hasattr(WikiFactorProxy, "get_factor")

    def test_evaluation_dataloader_alternative_available(self):
        """替代方案 3: contracts.DataLoader 可用。"""
        from QuantNodes.research.quant_alpha.evaluation.contracts import DataLoader

        assert hasattr(DataLoader, "load")
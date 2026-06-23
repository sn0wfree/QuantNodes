# coding: utf-8
"""DataSource 顶层 ABC 测试 (Phase 3.3)。"""
from __future__ import annotations

import pytest

from QuantNodes.core.data_source import DataSource


class TestDataSourceABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DataSource()

    def test_subclass_must_implement_close(self):
        class Incomplete(DataSource):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_context_manager_calls_close(self):
        calls = []

        class Concrete(DataSource):
            def close(self):
                calls.append("closed")

        with Concrete() as ds:
            assert isinstance(ds, DataSource)
        assert calls == ["closed"]

    def test_exit_returns_false(self):
        class Concrete(DataSource):
            def close(self):
                pass

        ds = Concrete()
        assert ds.__exit__(None, None, None) is False


class TestSubclassRelationships:
    def test_basedbnode_is_datasource(self):
        from QuantNodes.database_node import BaseDBNode

        assert issubclass(BaseDBNode, DataSource)

    def test_db_node_instance_is_datasource(self):
        from QuantNodes.database_node import SQLiteNode

        node = SQLiteNode(":memory:")
        assert isinstance(node, DataSource)

    def test_db_close_delegates_disconnect(self):
        from QuantNodes.database_node import SQLiteNode

        node = SQLiteNode(":memory:")
        calls = []
        node.disconnect = lambda: calls.append("disc")
        node.close()
        assert calls == ["disc"]

    def test_fileformatloader_is_datasource(self):
        from QuantNodes.research.factor_test.utils.file_loaders import (
            FileFormatLoader,
        )

        assert issubclass(FileFormatLoader, DataSource)

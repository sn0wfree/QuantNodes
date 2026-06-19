# coding: utf-8
"""R4 微清理回归: load_data 永真表达式修复 + e2e rename alias."""

import pandas as pd
import pytest


def test_load_data_loads_price_regardless_of_load_keys(tmp_path, monkeypatch):
    """A5: 删除 ``if 'cp' in keys or 'cp' not in keys`` 永真表达式后,
    price 仍然无条件加载 (与原行为一致)."""
    from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode

    class FakeLoader:
        def __init__(self, *a, **kw):
            self.calls = []
        def load_factor(self, *a, **kw):
            return pd.DataFrame({"x": [1, 2]})
        def load_h5(self, fname, key):
            self.calls.append((fname, key))
            if key == "cp":
                return pd.DataFrame({"a": [1.0, 2.0]})
            if key == "index_cp":
                return pd.DataFrame({"i": [3.0, 4.0]})
            raise KeyError(key)
        def add_index(self, df, axis_type="stock"):
            return df
        def get_stock_axis(self):
            return pd.DataFrame({"sl": ["a"]}), pd.DataFrame({"td": [20240101]})
        def valid_shape(self, df):
            return False

    monkeypatch.setattr(
        "QuantNodes.research.factor_test.nodes.load_data_node.DataLoader",
        FakeLoader,
    )
    node = LoadDataNode(config={
        "data_path": str(tmp_path),
        # 故意不含 'cp' — 修复前永真表达式仍会加载, 修复后无条件加载
        "load_keys": ["stklist", "trade_dt"],
    })
    out = node._execute()
    assert "price" in out, "price 应该总是被加载 (A5 修复)"


def test_inject_synthetic_data_alias_still_works():
    """B3: ``_inject_synthetic_data`` 重命名为 ``_inject_prepared_data``,
    旧名称作为 alias 保留 1 周期."""
    from QuantNodes.research.factor_test.e2e import run_evolution_e2e
    assert run_evolution_e2e._inject_synthetic_data is run_evolution_e2e._inject_prepared_data


def test_pipeline_runner_no_sys_path_pollution():
    """B1: pipeline_runner.py 顶层 sys.path.insert 已删除,
    导入不应破坏 sys.path."""
    import sys
    before = list(sys.path)
    from QuantNodes.research.factor_test import pipeline_runner  # noqa: F401
    after = list(sys.path)
    # 删 _PROJECT_ROOT 后, sys.path 不应被任意 prepend
    assert before == after or after[0] != str(
        __import__("pathlib").Path(pipeline_runner.__file__).resolve().parents[3]
    )

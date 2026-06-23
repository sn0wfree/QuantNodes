# coding=utf-8
"""Tests for PR-QN-4 dual-engine framework (engine detection + registry switching)."""
from __future__ import annotations

from QuantNodes.operators._engine import Engine, detect_engine
from QuantNodes.operators.composite_dag import (
    _COMPOSITE_REGISTRY_POLARS,
    _COMPOSITE_REGISTRY_PANDAS,
    is_composite_op,
    get_composite_spec,
    list_composite_ops,
)


class TestEngineEnum:
    def test_polars_value(self):
        assert Engine.POLARS.value == "polars"

    def test_pandas_value(self):
        assert Engine.PANDAS.value == "pandas"

    def test_auto_value(self):
        assert Engine.AUTO.value == "auto"


class TestDetectEngine:
    def test_polars_import_as(self):
        code = "import polars as pl\nresult = pl.col('close')"
        assert detect_engine(code) == Engine.POLARS

    def test_polars_from_import(self):
        code = "from polars import col, lit"
        assert detect_engine(code) == Engine.POLARS

    def test_pandas_import_as(self):
        code = "import pandas as pd\nresult = pd.DataFrame({'a': [1]})"
        assert detect_engine(code) == Engine.PANDAS

    def test_pandas_from_import(self):
        code = "from pandas import DataFrame"
        assert detect_engine(code) == Engine.PANDAS

    def test_mixed_prefers_polars(self):
        code = (
            "import polars as pl\n"
            "import pandas as pd\n"
            "df = pl.DataFrame({'a': [1]})"
        )
        assert detect_engine(code) == Engine.POLARS

    def test_empty_code_returns_polars(self):
        assert detect_engine("") == Engine.POLARS

    def test_syntax_error_returns_polars(self):
        assert detect_engine("def (") == Engine.POLARS

    def test_no_imports_returns_polars(self):
        code = "x = 1 + 2"
        assert detect_engine(code) == Engine.POLARS

    def test_polars_submodule(self):
        code = "from polars.exceptions import InvalidOperationError"
        assert detect_engine(code) == Engine.POLARS

    def test_pandas_submodule(self):
        code = "from pandas.core.dtypes.common import is_numeric_dtype"
        assert detect_engine(code) == Engine.PANDAS


class TestCompositeSpecEngineField:
    def test_default_engine_is_polars(self):
        from QuantNodes.operators.composite_dag import CompositeSpec

        spec = CompositeSpec(name="test", template=lambda x: x)
        assert spec.engine == "polars"

    def test_explicit_engine_pandas(self):
        from QuantNodes.operators.composite_dag import CompositeSpec

        spec = CompositeSpec(name="test", template=lambda x: x, engine="pandas")
        assert spec.engine == "pandas"


class TestDualRegistry:
    def test_polars_registry_has_existing_ops(self):
        assert "industry_neutralize" in _COMPOSITE_REGISTRY_POLARS.list()

    def test_pandas_registry_empty_by_default(self):
        assert len(_COMPOSITE_REGISTRY_PANDAS.list()) == 0

    def test_is_composite_op_any(self):
        assert is_composite_op("industry_neutralize", engine="any")

    def test_is_composite_op_polars(self):
        assert is_composite_op("industry_neutralize", engine="polars")

    def test_is_composite_op_pandas_not_found(self):
        assert not is_composite_op("industry_neutralize", engine="pandas")

    def test_get_composite_spec_polars(self):
        spec = get_composite_spec("industry_neutralize", engine="polars")
        assert spec is not None
        assert spec.engine == "polars"

    def test_get_composite_spec_pandas_returns_none(self):
        spec = get_composite_spec("industry_neutralize", engine="pandas")
        assert spec is None

    def test_list_composite_ops_polars(self):
        ops = list_composite_ops(engine="polars")
        assert "industry_neutralize" in ops

    def test_list_composite_ops_pandas_empty(self):
        ops = list_composite_ops(engine="pandas")
        assert len(ops) == 0

    def test_list_composite_ops_any_union(self):
        ops = list_composite_ops(engine="any")
        assert "industry_neutralize" in ops

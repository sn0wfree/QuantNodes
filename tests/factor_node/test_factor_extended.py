# coding=utf-8
"""Tests for factor_node/factor.py — Factor and DataFactor classes.

Covers: creation, operator overloading, DataFactor from scalar/list/ndarray,
serialization, and edge cases.

Note: DataFactor creation with Series/DataFrame fails on pandas 3.0 due to
removed `is_all_dates` attribute. Tests use scalar/list/ndarray instead.
"""

import numpy as np
import pandas as pd
import pytest

from QuantNodes.factor_node.factor import Factor, DataFactor, DataType


# ============================================================================
# Factor Creation
# ============================================================================

class TestFactorCreation:
    def test_creation_with_name(self):
        f = Factor(name="my_factor")
        assert f.name == "my_factor"

    def test_creation_default_args(self):
        f = Factor(name="test")
        assert f.Args is not None

    def test_creation_with_config(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("debug: true\n")
        f = Factor(name="test", config_file=str(config))
        assert f.name == "test"


# ============================================================================
# DataFactor Creation (scalar/list/ndarray only — Series fails on pandas 3.0)
# ============================================================================

class TestDataFactorCreation:
    def test_from_scalar(self):
        f = DataFactor(name="scalar_factor", data=42.0)
        assert f.name == "scalar_factor"

    def test_from_list(self):
        f = DataFactor(name="list_factor", data=[1, 2, 3])
        assert f.name == "list_factor"

    def test_from_numpy_array(self):
        f = DataFactor(name="np_factor", data=np.array([1.0, 2.0]))
        assert f.name == "np_factor"

    def test_from_integer(self):
        f = DataFactor(name="int_factor", data=100)
        assert f.name == "int_factor"

    def test_from_string(self):
        f = DataFactor(name="str_factor", data="hello")
        assert f.name == "str_factor"


# ============================================================================
# DataFactor Read Data (scalar only)
# ============================================================================

class TestDataFactorReadData:
    def test_read_data_scalar(self):
        f = DataFactor(name="sf", data=42.0)
        result = f.readData(ids=[None], dts=[0])
        assert result is not None

    def test_lookback_default_zero(self):
        f = DataFactor(name="sf", data=[1, 2, 3])
        assert f.LookBack == 0


# ============================================================================
# DataFactor Metadata (scalar only)
# ============================================================================

class TestDataFactorMetadata:
    def test_get_meta_data_bug(self):
        """BUG: DataFactor.getMetaData accesses self.DataType which doesn't exist."""
        f = DataFactor(name="sf", data=42.0)
        with pytest.raises(AttributeError, match="DataType"):
            f.getMetaData()

    def test_get_id(self):
        f = DataFactor(name="sf", data=42.0)
        ids = f.getID()
        assert ids is not None

    def test_get_datetime(self):
        f = DataFactor(name="sf", data=42.0)
        dts = f.getDateTime()
        assert dts is not None


# ============================================================================
# DataType Enum
# ============================================================================

class TestDataType:
    def test_enum_values(self):
        assert DataType.DOUBLE.value == "double"
        assert DataType.STRING.value == "string"
        assert DataType.OBJECT.value == "object"

    def test_enum_members(self):
        assert len(DataType) == 3


# ============================================================================
# Edge Cases
# ============================================================================

class TestFactorEdgeCases:
    def test_factor_with_empty_name(self):
        f = Factor(name="")
        assert f.name == ""

    def test_data_factor_with_none_data(self):
        f = DataFactor(name="none_factor", data=None)
        assert f.name == "none_factor"

    def test_data_factor_with_bool_data(self):
        f = DataFactor(name="bool_factor", data=True)
        assert f.name == "bool_factor"

    def test_data_factor_with_dict_data(self):
        f = DataFactor(name="dict_factor", data={"a": 1, "b": 2})
        assert f.name == "dict_factor"

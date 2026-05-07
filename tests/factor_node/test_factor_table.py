# coding=utf-8
"""QuantNodes.factor_node.factor_table 单元测试

Note: FactorTable and CustomFT have a known issue with sys_args in their
internal _ErgodicMode and _OperationMode classes. We test what we can without
hitting this issue.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from QuantNodes.factor_node.factor_table import ErgodicModeType


class TestErgodicModeType:
    def test_ergodic_mode_type_enum_values(self):
        assert ErgodicModeType.FACTOR.value == "因子"
        assert ErgodicModeType.ID.value == "ID"

    def test_ergodic_mode_type_is_enum(self):
        from enum import Enum
        assert issubclass(ErgodicModeType, Enum)


class TestFactorTableName:
    """Test FactorTable.name property without instantiating FactorTable"""

    def test_factor_table_name_property_access(self):
        from QuantNodes.factor_node.factor_table import FactorTable
        with patch.object(FactorTable, '__init__', return_value=None):
            ft = FactorTable.__new__(FactorTable)
            ft._Name = "test_table"
            assert ft.Name == "test_table"

    def test_factor_table_factor_db_property_access(self):
        from QuantNodes.factor_node.factor_table import FactorTable
        with patch.object(FactorTable, '__init__', return_value=None):
            ft = FactorTable.__new__(FactorTable)
            mock_fdb = MagicMock()
            mock_fdb.Name = "test_fdb"
            ft._FactorDB = mock_fdb
            assert ft.FactorDB == mock_fdb


class TestFactorTableMethods:
    """Test FactorTable methods that don't require full initialization"""

    def test_get_meta_data_returns_empty_dict(self):
        from QuantNodes.factor_node.factor_table import FactorTable
        with patch.object(FactorTable, '__init__', return_value=None):
            ft = FactorTable.__new__(FactorTable)
            ft._Name = "test"
            ft._FactorDB = None
            result = ft.getMetaData()
            assert result == {}

    def test_get_meta_data_key_returns_none(self):
        from QuantNodes.factor_node.factor_table import FactorTable
        with patch.object(FactorTable, '__init__', return_value=None):
            ft = FactorTable.__new__(FactorTable)
            ft._Name = "test"
            ft._FactorDB = None
            result = ft.getMetaData("key")
            assert result is None


class TestCustomFTFactory:
    """Test CustomFT factory methods"""

    def test_factor_names_from_empty_factors(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            assert sorted(cft.FactorNames) == []

    def test_set_datetime(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            dts = ["2024-01-01", "2024-01-02", "2024-01-03"]
            cft.setDateTime(dts)
            assert cft._DateTimes == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_set_id(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            ids = ["000001", "000002", "000003"]
            cft.setID(ids)
            assert cft._IDs == ["000001", "000002", "000003"]

    def test_get_id_returns_ids(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = ["id1", "id2", "id3"]
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            assert cft.getID() == ["id1", "id2", "id3"]

    def test_get_datetime_returns_all(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = ["2024-01-01", "2024-01-02", "2024-01-03"]
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            assert cft.getDateTime() == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_id_filter_str_property(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            assert cft.IDFilterStr is None

    def test_add_factors_raises_on_duplicate(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            mock_factor = MagicMock()
            mock_factor.Name = "factor1"
            cft.addFactors([mock_factor])
            with pytest.raises(Exception) as exc_info:
                cft.addFactors([mock_factor])
            assert "重名" in str(exc_info.value) or "FactorError" in str(exc_info.value)

    def test_delete_factors(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            mock_factor = MagicMock()
            mock_factor.Name = "factor1"
            cft.addFactors([mock_factor])
            assert "factor1" in cft._Factors
            cft.deleteFactors(["factor1"])
            assert "factor1" not in cft._Factors

    def test_delete_factors_with_none_removes_all(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            mock_factor1 = MagicMock()
            mock_factor1.Name = "factor1"
            mock_factor2 = MagicMock()
            mock_factor2.Name = "factor2"
            cft.addFactors([mock_factor1, mock_factor2])
            cft.deleteFactors()
            assert cft._Factors == {}

    def test_rename_factor(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            mock_factor = MagicMock()
            mock_factor.Name = "factor1"
            cft.addFactors([mock_factor])
            cft.renameFactor("factor1", "factor_renamed")
            assert "factor_renamed" in cft._Factors
            assert "factor1" not in cft._Factors

    def test_rename_factor_raises_on_nonexistent(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            with pytest.raises(Exception) as exc_info:
                cft.renameFactor("nonexistent", "new_name")
            assert "不存在" in str(exc_info.value) or "FactorError" in str(exc_info.value)

    def test_rename_factor_raises_on_duplicate(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = []
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            mock_factor1 = MagicMock()
            mock_factor1.Name = "factor1"
            mock_factor2 = MagicMock()
            mock_factor2.Name = "factor2"
            cft.addFactors([mock_factor1, mock_factor2])
            with pytest.raises(Exception) as exc_info:
                cft.renameFactor("factor1", "factor2")
            assert "重名" in str(exc_info.value) or "FactorError" in str(exc_info.value)

    def test_get_filtered_id_without_filter(self):
        from QuantNodes.factor_node.factor_table import CustomFT
        with patch.object(CustomFT, '__init__', return_value=None):
            cft = CustomFT.__new__(CustomFT)
            cft._Factors = {}
            cft._DateTimes = []
            cft._IDs = ["id1", "id2", "id3"]
            cft._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
            cft._TableArgDict = {}
            cft._IDFilterStr = None
            cft._CompiledIDFilter = {}
            cft._isStarted = False
            cft._Name = "custom"
            result = cft.getFilteredID("2024-01-01")
            assert result == ["id1", "id2", "id3"]

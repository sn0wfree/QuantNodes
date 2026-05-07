# coding=utf-8
"""QuantNodes.factor_node.factor_operation 单元测试"""
import numpy as np

from QuantNodes.factor_node.factor_operation import (
    DerivativeFactor, PointOperation, TimeOperation, SectionOperation, PanelOperation,
    DTModeType, IDModeType, OutputModeType, LookBackMode, DataOperationType,
    _DefaultOperator
)


class TestDataOperationType:
    def test_enum_values(self):
        assert DataOperationType.DOUBLE.value == "double"
        assert DataOperationType.STRING.value == "string"
        assert DataOperationType.OBJECT.value == "object"


class TestDTModeType:
    def test_enum_values(self):
        assert DTModeType.SINGLE.value == "单时点"
        assert DTModeType.MULTI.value == "多时点"


class TestIDModeType:
    def test_enum_values(self):
        assert IDModeType.SINGLE.value == "单ID"
        assert IDModeType.MULTI.value == "多ID"


class TestOutputModeType:
    def test_enum_values(self):
        assert OutputModeType.FULL_SECTION.value == "全截面"
        assert OutputModeType.SINGLE_ID.value == "单ID"


class TestLookBackMode:
    def test_enum_values(self):
        assert LookBackMode.ROLLING.value == "滚动窗口"
        assert LookBackMode.EXPANDING.value == "扩张窗口"


class TestDefaultOperator:
    def test_returns_nan(self):
        result = _DefaultOperator(None, None, None, [], {})
        assert np.isnan(result)


class TestDerivativeFactor:
    def test_creation_with_name(self):
        df = DerivativeFactor(name="TestFactor")
        assert df.name == "TestFactor"

    def test_creation_with_empty_descriptors(self):
        df = DerivativeFactor(name="Test")
        assert df.Descriptors == []

    def test_descriptors_defaults_to_empty_list(self):
        df = DerivativeFactor(name="Test")
        assert df.Descriptors == []

    def test_operator_defaults_to_default_operator(self):
        df = DerivativeFactor(name="Test")
        assert df.Operator == _DefaultOperator

    def test_model_args_defaults_to_empty_dict(self):
        df = DerivativeFactor(name="Test")
        assert df.ModelArgs == {}

    def test_get_metadata_returns_datatype_series(self):
        df = DerivativeFactor(name="Test")
        result = df.getMetaData()
        assert "DataType" in result.index

    def test_get_metadata_with_datatype_key(self):
        df = DerivativeFactor(name="Test")
        result = df.getMetaData(key="DataType")
        assert result == df.DataType

    def test_start_returns_zero(self):
        df = DerivativeFactor(name="Test")
        assert df.start(dts=[]) == 0

    def test_end_returns_zero(self):
        df = DerivativeFactor(name="Test")
        assert df.end() == 0


class TestPointOperation:
    def test_creation(self):
        op = PointOperation(name="TestOp")
        assert op.name == "TestOp"

    def test_dt_mode_defaults_to_single(self):
        op = PointOperation()
        assert op.DTMode == DTModeType.SINGLE

    def test_id_mode_defaults_to_single(self):
        op = PointOperation()
        assert op.IDMode == IDModeType.SINGLE

    def test_dispatch_map_has_all_modes(self):
        dispatch = PointOperation._DT_ID_DISPATCH
        assert (DTModeType.MULTI, IDModeType.MULTI) in dispatch
        assert (DTModeType.SINGLE, IDModeType.SINGLE) in dispatch
        assert (DTModeType.MULTI, IDModeType.SINGLE) in dispatch
        assert (DTModeType.SINGLE, IDModeType.MULTI) in dispatch


class TestTimeOperation:
    def test_creation(self):
        op = TimeOperation(name="TestOp")
        assert op.name == "TestOp"

    def test_dt_mode_defaults_to_single(self):
        op = TimeOperation()
        assert op.DTMode == DTModeType.SINGLE

    def test_id_mode_defaults_to_single(self):
        op = TimeOperation()
        assert op.IDMode == IDModeType.SINGLE

    def test_dispatch_map_has_all_modes(self):
        dispatch = TimeOperation._DT_ID_DISPATCH
        assert (DTModeType.SINGLE, IDModeType.SINGLE) in dispatch
        assert (DTModeType.SINGLE, IDModeType.MULTI) in dispatch
        assert (DTModeType.MULTI, IDModeType.SINGLE) in dispatch
        assert (DTModeType.MULTI, IDModeType.MULTI) in dispatch


class TestSectionOperation:
    def test_creation(self):
        op = SectionOperation(name="TestOp")
        assert op.name == "TestOp"

    def test_dt_mode_defaults_to_single(self):
        op = SectionOperation()
        assert op.DTMode == DTModeType.SINGLE

    def test_output_mode_defaults_to_full_section(self):
        op = SectionOperation()
        assert op.OutputMode == OutputModeType.FULL_SECTION

    def test_output_dispatch_map_has_all_modes(self):
        dispatch = SectionOperation._OUTPUT_DT_DISPATCH
        assert (OutputModeType.FULL_SECTION, DTModeType.SINGLE) in dispatch
        assert (OutputModeType.FULL_SECTION, DTModeType.MULTI) in dispatch
        assert (OutputModeType.SINGLE_ID, DTModeType.SINGLE) in dispatch
        assert (OutputModeType.SINGLE_ID, DTModeType.MULTI) in dispatch


class TestPanelOperation:
    def test_creation(self):
        op = PanelOperation(name="TestOp")
        assert op.name == "TestOp"

    def test_dt_mode_defaults_to_single(self):
        op = PanelOperation()
        assert op.DTMode == DTModeType.SINGLE

    def test_output_mode_defaults_to_full_section(self):
        op = PanelOperation()
        assert op.OutputMode == OutputModeType.FULL_SECTION

    def test_output_dispatch_map_has_all_modes(self):
        dispatch = PanelOperation._OUTPUT_DT_DISPATCH
        assert (OutputModeType.FULL_SECTION, DTModeType.SINGLE) in dispatch
        assert (OutputModeType.FULL_SECTION, DTModeType.MULTI) in dispatch
        assert (OutputModeType.SINGLE_ID, DTModeType.SINGLE) in dispatch
        assert (OutputModeType.SINGLE_ID, DTModeType.MULTI) in dispatch

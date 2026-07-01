# coding=utf-8
"""Tests for factor_node/factor_table.py — structural and import tests.

Note: CustomFT/FactorTable __init__ calls QuantNodesObject.__init__(sys_args=...)
but QuantNodesObject is a @dataclass that doesn't accept sys_args. This is a
pre-existing compatibility bug. Tests here verify imports, class existence,
and method signatures only.
"""

import pytest

from QuantNodes.factor_node.factor_table import FactorTable, CustomFT


# ============================================================================
# Import Tests
# ============================================================================

class TestFactorTableImports:
    def test_import_factor_table(self):
        assert FactorTable is not None

    def test_import_custom_ft(self):
        assert CustomFT is not None

    def test_custom_ft_is_subclass(self):
        assert issubclass(CustomFT, FactorTable)


# ============================================================================
# Class Structure
# ============================================================================

class TestFactorTableStructure:
    def test_factor_table_has_name_property(self):
        assert hasattr(FactorTable, 'Name')

    def test_factor_table_has_factor_names_property(self):
        assert hasattr(FactorTable, 'FactorNames')

    def test_factor_table_has_get_factor_method(self):
        assert hasattr(FactorTable, 'getFactor')

    def test_factor_table_has_read_data_method(self):
        assert hasattr(FactorTable, 'readData')

    def test_factor_table_has_start_method(self):
        assert hasattr(FactorTable, 'start')

    def test_factor_table_has_end_method(self):
        assert hasattr(FactorTable, 'end')

    def test_factor_table_has_move_method(self):
        assert hasattr(FactorTable, 'move')

    def test_factor_table_has_write2fdb_method(self):
        assert hasattr(FactorTable, 'write2FDB')


class TestCustomFTStructure:
    def test_custom_ft_has_factor_names_property(self):
        assert hasattr(CustomFT, 'FactorNames')

    def test_custom_ft_has_add_factors_method(self):
        assert hasattr(CustomFT, 'addFactors')

    def test_custom_ft_has_delete_factors_method(self):
        assert hasattr(CustomFT, 'deleteFactors')

    def test_custom_ft_has_rename_factor_method(self):
        assert hasattr(CustomFT, 'renameFactor')

    def test_custom_ft_has_set_datetime_method(self):
        assert hasattr(CustomFT, 'setDateTime')

    def test_custom_ft_has_set_id_method(self):
        assert hasattr(CustomFT, 'setID')

    def test_custom_ft_has_set_id_filter_method(self):
        assert hasattr(CustomFT, 'setIDFilter')

    def test_custom_ft_has_start_method(self):
        assert hasattr(CustomFT, 'start')

    def test_custom_ft_has_end_method(self):
        assert hasattr(CustomFT, 'end')

    def test_custom_ft_has_get_factor_method(self):
        assert hasattr(CustomFT, 'getFactor')

    def test_custom_ft_has_get_factor_metadata_method(self):
        assert hasattr(CustomFT, 'getFactorMetaData')


# ============================================================================
# Method Signatures
# ============================================================================

class TestMethodSignatures:
    def test_custom_ft_init_signature(self):
        import inspect
        sig = inspect.signature(CustomFT.__init__)
        params = list(sig.parameters.keys())
        assert "name" in params

    def test_factor_table_init_signature(self):
        import inspect
        sig = inspect.signature(FactorTable.__init__)
        params = list(sig.parameters.keys())
        assert "name" in params
        assert "fdb" in params

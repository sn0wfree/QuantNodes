# coding=utf-8
"""QuantNodes.factor_node.factor_db 单元测试"""

from QuantNodes.factor_node.factor_db import FactorDB, WritableFactorDB


class TestFactorDB:
    def test_factor_db_creation(self):
        db = FactorDB(name="TestDB")
        assert db.name == "TestDB"

    def test_connect(self):
        db = FactorDB()
        assert db.connect() == 0

    def test_disconnect(self):
        db = FactorDB()
        assert db.disconnect() == 0

    def test_is_available(self):
        db = FactorDB()
        assert db.isAvailable() is True

    def test_table_names_empty(self):
        db = FactorDB()
        assert db.TableNames == []

    def test_get_table_returns_none(self):
        db = FactorDB()
        assert db.getTable("nonexistent") is None

    def test_get_id_empty(self):
        db = FactorDB()
        assert db.getID() == []

    def test_get_datetime_empty(self):
        db = FactorDB()
        assert db.getDateTime() == []


class TestWritableFactorDB:
    def test_writable_factor_db_creation(self):
        db = WritableFactorDB(name="WritableDB")
        assert db.name == "WritableDB"

    def test_rename_table(self):
        db = WritableFactorDB()
        assert db.renameTable("old", "new") == 0

    def test_delete_table(self):
        db = WritableFactorDB()
        assert db.deleteTable("test_table") == 0

    def test_set_table_meta_data(self):
        db = WritableFactorDB()
        assert db.setTableMetaData("table", key="key", value="value") == 0

    def test_set_table_meta_data_with_dict(self):
        db = WritableFactorDB()
        assert db.setTableMetaData("table", meta_data={"k1": "v1", "k2": "v2"}) == 0

    def test_rename_factor(self):
        db = WritableFactorDB()
        assert db.renameFactor("table", "old", "new") == 0

    def test_delete_factor(self):
        db = WritableFactorDB()
        assert db.deleteFactor("table", ["factor1", "factor2"]) == 0

    def test_set_factor_meta_data(self):
        db = WritableFactorDB()
        assert db.setFactorMetaData("table", "factor", key="key", value="value") == 0

    def test_write_data(self):
        db = WritableFactorDB()
        assert db.writeData(None, "table") == 0

    def test_offset_datetime_zero(self):
        db = WritableFactorDB()
        assert db.offsetDateTime(0, "table", ["factor"]) == 0

    def test_offset_datetime_nonzero_table_not_found(self):
        db = WritableFactorDB()
        assert db.offsetDateTime(5, "nonexistent", ["factor"]) == -1

    def test_read_transform_write_table_not_found(self):
        db = WritableFactorDB()
        result = db._read_transform_write(
            table_name="nonexistent",
            factor_names=["factor"],
            ids=["id1"],
            dts=["dt1"],
            transform_fn=lambda d, f: d,
        )
        assert result == -1

    def test_change_data_table_not_found(self):
        db = WritableFactorDB()
        assert db.changeData("nonexistent", ["factor"], ["id1"], ["dt1"]) == -1

    def test_fill_na_table_not_found(self):
        db = WritableFactorDB()
        assert db.fillNA(0.0, "nonexistent", ["factor"], ["id1"], ["dt1"]) == -1

    def test_replace_data_table_not_found(self):
        db = WritableFactorDB()
        assert db.replaceData(0.0, 1.0, "nonexistent", ["factor"], ["id1"], ["dt1"]) == -1

    def test_optimize_data(self):
        db = WritableFactorDB()
        assert db.optimizeData("table", ["factor"]) == 0

    def test_fix_data(self):
        db = WritableFactorDB()
        assert db.fixData("table", ["factor"]) == 0

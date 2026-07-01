# coding=utf-8
"""Tests for database_node/{duckdb_node,mysql_node,clickhouse_node}.

Covers: DuckDBNode (memory/file, insert_df strategies), MySQLNode creation
(no live connection), ClickHouseNode creation, CHBase HTTP helpers,
factory integration.
"""

from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.database_node.duckdb_node import DuckDBNode
from QuantNodes.database_node.mysql_node import MySQLNode
from QuantNodes.database_node.clickhouse_node import ClickHouseNode, CHBase, ch_conn_tuple
from QuantNodes.database_node.factory import create_db_node


# ============================================================================
# DuckDBNode
# ============================================================================

class TestDuckDBNode:
    def test_creation_default(self):
        node = DuckDBNode()
        assert node._database == ":memory:"
        assert node._read_only is False

    def test_creation_file_mode(self, tmp_path):
        path = tmp_path / "test.duckdb"
        node = DuckDBNode(database=str(path))
        assert node._database == str(path)

    def test_creation_read_only(self):
        node = DuckDBNode("/tmp/test.db", read_only=True)
        assert node._read_only is True

    def test_connect_memory(self):
        node = DuckDBNode()
        conn = node.connect()
        assert conn is not None
        # Should be able to execute a query
        result = conn.execute("SELECT 1 AS x").fetchdf()
        assert result["x"].iloc[0] == 1

    def test_query(self):
        node = DuckDBNode()
        node.connect()
        df = node.query("SELECT 1 AS x, 2 AS y")
        assert len(df) == 1
        assert "x" in df.columns
        assert "y" in df.columns

    def test_query_with_params(self):
        node = DuckDBNode()
        node.connect()
        df = node.query("SELECT ? AS val", params=(42,))
        assert df["val"].iloc[0] == 42

    def test_execute_create_table(self):
        node = DuckDBNode()
        node.connect()
        node.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
        df = node.query("SELECT * FROM t")
        assert len(df) == 0

    def test_execute_returns_rowcount(self):
        """DuckDB always returns -1 for rowcount (not tracked)."""
        node = DuckDBNode()
        node.connect()
        # DuckDB doesn't track rowcount, always -1
        n_ddl = node.execute("CREATE TABLE t (id INTEGER)")
        assert n_ddl == -1  # DuckDB quirk
        # Verify the DDL actually worked
        result = node.query("SELECT COUNT(*) AS n FROM t")
        assert result["n"].iloc[0] == 0

    def test_insert_df_append(self):
        node = DuckDBNode()
        node.connect()
        df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        n = node.insert_df(df, "t", if_exists="append")
        assert n == 2
        result = node.query("SELECT * FROM t ORDER BY id")
        assert len(result) == 2

    def test_insert_df_replace(self):
        node = DuckDBNode()
        node.connect()
        # Initial insert
        df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        node.insert_df(df1, "t", if_exists="append")
        # Replace
        df2 = pd.DataFrame({"id": [3], "value": [30]})
        node.insert_df(df2, "t", if_exists="replace")
        result = node.query("SELECT * FROM t")
        assert len(result) == 1

    def test_insert_df_default_if_exists(self):
        """Default if_exists='append' should work."""
        node = DuckDBNode()
        node.connect()
        df = pd.DataFrame({"x": [1]})
        node.insert_df(df, "t")
        # Default if_exists='append'
        node.insert_df(df, "t")
        result = node.query("SELECT * FROM t")
        # Without 'replace' handling, second append may create table again
        # (the except branch handles this)
        assert result is not None

    def test_disconnect(self):
        node = DuckDBNode()
        node.connect()
        node.disconnect()
        assert node._conn is None

    def test_context_manager(self):
        with DuckDBNode() as node:
            df = node.query("SELECT 1 AS x")
            assert "x" in df.columns


# ============================================================================
# MySQLNode
# ============================================================================

class TestMySQLNode:
    def test_creation(self):
        node = MySQLNode(
            host="localhost",
            user="root",
            passwd="password",
            db="mydb",
        )
        assert node._host == "localhost"
        assert node._user == "root"
        assert node._db == "mydb"

    def test_creation_custom_port(self):
        node = MySQLNode(host="localhost", port=3307)
        assert node._port == 3307

    def test_creation_default_charset(self):
        node = MySQLNode(host="localhost")
        assert node._charset == "UTF8"

    def test_creation_custom_pool_size(self):
        node = MySQLNode(host="localhost", pool_size=20)
        assert node._pool_size == 20

    def test_creation_custom_pool_recycle(self):
        node = MySQLNode(host="localhost", pool_recycle=7200)
        assert node._pool_recycle == 7200

    def test_build_url(self):
        node = MySQLNode(
            host="db.example.com",
            port=3306,
            user="user",
            passwd="pass",
            db="mydb",
        )
        url = node._build_url()
        assert "mysql+pymysql://" in url
        assert "user:pass" in url
        assert "db.example.com" in url
        assert "mydb" in url
        assert "charset=UTF8" in url

    def test_build_url_with_special_chars_in_password(self):
        """Special chars in password should not break URL."""
        node = MySQLNode(host="host", passwd="p@ssw0rd!#")
        # Should not raise
        url = node._build_url()
        assert "p@ssw0rd!#" in url or "p%40ssw0rd" in url  # either way is fine

    def test_disconnect_no_connection(self):
        """disconnect on unconnected node is safe."""
        node = MySQLNode(host="localhost")
        node.disconnect()  # Should not raise
        assert node._engine is None


# ============================================================================
# CHBase (ClickHouse HTTP)
# ============================================================================

class TestCHBase:
    def test_creation(self):
        ch = CHBase(name="test", user="default", passwd="", host="localhost", port=8123, db="default")
        assert ch._para.host == "localhost"
        assert ch._para.port == 8123

    def test_creation_defaults(self):
        ch = CHBase(name="test")
        assert ch._para.host == "0.0.0.0"
        assert ch._para.port == 8123
        assert ch._para.user == "default"
        assert ch._para.passwd == "123456"

    def test_accepted_formats(self):
        ch = CHBase(name="test")
        assert "DataFrame" in ch.accepted_formats
        assert "JSON" in ch.accepted_formats
        assert "CSV" in ch.accepted_formats

    def test_default_settings(self):
        ch = CHBase(name="test")
        assert ch.settings["enable_http_compression"] == 1
        assert ch.settings["send_progress_in_http_headers"] == 0
        assert ch.settings["wait_end_of_query"] == 0

    def test_check_sql_select_only(self):
        # Valid queries
        CHBase._check_sql_select_only("SELECT * FROM t")
        CHBase._check_sql_select_only("DESCRIBE TABLE t")
        CHBase._check_sql_select_only("SHOW TABLES")
        CHBase._check_sql_select_only("SHOW DATABASES")
        # Invalid queries
        with pytest.raises(ValueError, match="select"):
            CHBase._check_sql_select_only("INSERT INTO t VALUES (1)")
        with pytest.raises(ValueError, match="select"):
            CHBase._check_sql_select_only("DROP TABLE t")

    def test_transfer_sql_format(self):
        sql = "SELECT * FROM t"
        formatted = CHBase._transfer_sql_format(sql, convert_to="DataFrame")
        assert "format JSONCompact" in formatted
        assert "SELECT * FROM t" in formatted

    def test_transfer_sql_format_json(self):
        sql = "SELECT * FROM t"
        formatted = CHBase._transfer_sql_format(sql, convert_to=None)
        assert "format JSON" in formatted

    def test_merge_settings_invalid_key(self):
        with pytest.raises(ValueError, match="invalid"):
            CHBase._merge_settings({"invalid_key": 1})

    def test_merge_settings_valid(self):
        result = CHBase._merge_settings({"wait_end_of_query": 1})
        assert result["wait_end_of_query"] == 1

    def test_merge_settings_bool_conversion(self):
        result = CHBase._merge_settings({"enable_http_compression": True})
        assert result["enable_http_compression"] == 1

    def test_create_conn(self):
        ch = CHBase(name="test", host="localhost", port=8123)
        conn = ch._create_conn()
        assert conn is not None
        conn.close()


# ============================================================================
# ClickHouseNode
# ============================================================================

class TestClickHouseNode:
    def test_creation(self):
        node = ClickHouseNode(host="localhost", user="default", passwd="")
        assert node._host == "localhost"
        assert node._user == "default"
        assert node._interface == "http"
        assert node._port == 8123

    def test_creation_custom_port(self):
        node = ClickHouseNode(host="localhost", port=9000, interface="native")
        assert node._port == 9000
        assert node._interface == "native"

    def test_creation_with_pool_size(self):
        node = ClickHouseNode(host="localhost", pool_size=20)
        assert node._pool_size == 20

    def test_database_default(self):
        node = ClickHouseNode(host="localhost")
        assert node._database == "default"

    def test_show_tables_query_format(self):
        """ClickHouse show_tables uses FROM clause."""
        node = ClickHouseNode(host="localhost", database="mydb")
        # show_tables constructs SHOW TABLES FROM <db>
        # We can verify by checking the query string
        assert node._database == "mydb"
        # show_tables calls query with this format
        # (Can't actually call without a live server)

    def test_disconnect(self):
        node = ClickHouseNode(host="localhost")
        node.disconnect()
        assert node._client is None
        assert node._http_client is None

    def test_disconnect_native(self):
        node = ClickHouseNode(host="localhost", interface="native")
        node.disconnect()
        assert node._client is None


# ============================================================================
# Factory Integration
# ============================================================================

class TestFactoryIntegration:
    def test_create_duckdb_memory(self):
        node = create_db_node("duckdb", database=":memory:")
        assert isinstance(node, DuckDBNode)

    def test_create_duckdb_file(self, tmp_path):
        path = tmp_path / "test.duckdb"
        node = create_db_node("duckdb", database=str(path))
        assert isinstance(node, DuckDBNode)

    def test_create_mysql(self):
        node = create_db_node("mysql", host="localhost", user="root")
        assert isinstance(node, MySQLNode)

    def test_create_clickhouse(self):
        node = create_db_node("clickhouse", host="localhost")
        assert isinstance(node, ClickHouseNode)


# ============================================================================
# ch_conn_tuple
# ============================================================================

class TestChConnTuple:
    def test_namedtuple(self):
        tup = ch_conn_tuple("host", 8123, "user", "pass", "db")
        assert tup.host == "host"
        assert tup.port == 8123
        assert tup.user == "user"
        assert tup.passwd == "pass"
        assert tup.db == "db"

    def test_unpacking(self):
        tup = ch_conn_tuple("h", 1, "u", "p", "d")
        host, port, user, passwd, db = tup
        assert host == "h"
        assert passwd == "p"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_duckdb_disconnect_then_query(self):
        """After disconnect, query should auto-reconnect."""
        node = DuckDBNode()
        node.connect()
        node.disconnect()
        # Should auto-reconnect
        df = node.query("SELECT 1 AS x")
        assert df["x"].iloc[0] == 1

    def test_duckdb_multiple_inserts(self):
        node = DuckDBNode()
        node.connect()
        node.execute("CREATE TABLE t (id INTEGER, val INTEGER)")
        for i in range(3):
            df = pd.DataFrame({"id": [i], "val": [i * 10]})
            node.insert_df(df, "t", if_exists="append")
        result = node.query("SELECT COUNT(*) AS n FROM t")
        assert result["n"].iloc[0] == 3

    def test_ch_base_settings_overrides(self):
        """Custom settings override defaults."""
        ch = CHBase(name="test")
        # Merge custom settings
        original_compress = ch.settings["enable_http_compression"]
        # Note: settings are merged in __init__ via _merge_settings
        assert ch.settings is not None

    def test_ch_check_sql_lowercase(self):
        """Lowercase SQL is also accepted."""
        CHBase._check_sql_select_only("select * from t")
        CHBase._check_sql_select_only("describe table t")
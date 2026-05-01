# -*- coding: utf-8 -*-
"""database_node 单元测试"""
import os
import configparser
import pytest
import pandas as pd
from pathlib import Path

from QuantNodes.database_node import (
    BaseDBNode,
    SQLiteNode,
    DuckDBNode,
    MySQLNode,
    ClickHouseNode,
    CSVNode,
    ParquetNode,
)


def _load_conn_ini(section: str, defaults: dict) -> dict:
    """从 conn.ini 读取连接配置，fallback 到 defaults"""
    conn_ini = os.getenv("TEST_CONN_INI", "conn.ini")
    ini_path = Path(conn_ini)
    if ini_path.exists():
        parser = configparser.ConfigParser()
        parser.read(str(ini_path), encoding="utf-8")
        if section in parser:
            result = dict(defaults)
            result.update(dict(parser.items(section)))
            return result
    return defaults


class TestBaseDBNode:
    """BaseDBNode 基类测试"""

    def test_抽象类不能实例化(self):
        """BaseDBNode 是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            BaseDBNode()

    def test_健康检查成功(self):
        """健康检查成功返回 True"""
        node = SQLiteNode(":memory:")
        node.connect()
        assert node.health_check() is True
        node.disconnect()

    def test_健康检查失败_无连接(self):
        """未连接且无有效数据库时健康检查返回 False"""
        node = SQLiteNode("/nonexistent/path/to/db.sqlite")
        assert node.health_check() is False

    def test_上下文管理器(self):
        """测试 __enter__/__exit__ 协议"""
        with SQLiteNode(":memory:") as node:
            result = node.query("SELECT 1")
            assert len(result) == 1
        assert node._conn is None


class TestSQLiteNode:
    """SQLiteNode 测试"""

    def test_内存模式_基础(self):
        """:memory: 模式基础操作"""
        node = SQLiteNode(":memory:")
        node.connect()

        node.execute("CREATE TABLE test (id INT, name TEXT)")
        node.execute("INSERT INTO test VALUES (1, 'Alice')")

        result = node.query("SELECT * FROM test")
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

        node.disconnect()

    def test_文件模式(self, temp_sqlite_db):
        """文件模式 SQLite 操作"""
        node = SQLiteNode(str(temp_sqlite_db))
        node.connect()

        node.execute("CREATE TABLE test (id INT, name TEXT)")
        node.execute("INSERT INTO test VALUES (1, 'Bob')")

        result = node.query("SELECT * FROM test")
        assert len(result) == 1

        node.disconnect()

    def test_参数化查询(self):
        """带参数的查询"""
        node = SQLiteNode(":memory:")
        node.connect()

        node.execute("CREATE TABLE test (id INT, name TEXT)")
        node.execute("INSERT INTO test VALUES (1, 'Alice')")
        node.execute("INSERT INTO test VALUES (2, 'Bob')")

        result = node.query("SELECT * FROM test WHERE id = ?", (1,))
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

        node.disconnect()

    def test_insert_df_append(self, sample_df):
        """insert_df append 模式"""
        node = SQLiteNode(":memory:")
        node.connect()

        node.insert_df(sample_df, "test", if_exists='append')
        result = node.query("SELECT * FROM test")
        assert len(result) == 5

        node.insert_df(sample_df, "test", if_exists='append')
        result = node.query("SELECT * FROM test")
        assert len(result) == 10

        node.disconnect()

    def test_insert_df_replace(self, sample_df):
        """insert_df replace 模式"""
        node = SQLiteNode(":memory:")
        node.connect()

        node.insert_df(sample_df, "test", if_exists='replace')
        assert len(node.query("SELECT * FROM test")) == 5

        new_df = pd.DataFrame({'id': [100], 'name': ['Zack'], 'age': [50], 'score': [95.0]})
        node.insert_df(new_df, "test", if_exists='replace')
        result = node.query("SELECT * FROM test")
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Zack'

        node.disconnect()

    def test_insert_df_fail(self, sample_df):
        """insert_df fail 模式 - 表存在时抛出异常"""
        node = SQLiteNode(":memory:")
        node.connect()

        node.insert_df(sample_df, "test", if_exists='fail')

        with pytest.raises(ValueError):
            node.insert_df(sample_df, "test", if_exists='fail')

        node.disconnect()

    def test_断开连接(self):
        """断开连接后连接为 None"""
        node = SQLiteNode(":memory:")
        node.connect()
        assert node._conn is not None

        node.disconnect()
        assert node._conn is None

    def test_健康检查(self):
        """健康检查"""
        node = SQLiteNode(":memory:")
        node.connect()
        assert node.health_check() is True
        node.disconnect()


class TestDuckDBNode:
    """DuckDBNode 测试"""

    def test_内存模式_基础(self):
        """:memory: 模式基础操作"""
        node = DuckDBNode(":memory:")
        node.connect()

        node.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        node.execute("INSERT INTO test VALUES (1, 'Alice')")

        result = node.query("SELECT * FROM test")
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

        node.disconnect()

    def test_文件模式(self, temp_duckdb_db):
        """文件模式 DuckDB 操作"""
        node = DuckDBNode(str(temp_duckdb_db))
        node.connect()

        node.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        node.execute("INSERT INTO test VALUES (1, 'Bob')")

        result = node.query("SELECT * FROM test")
        assert len(result) == 1

        node.disconnect()

    def test_只读模式_写入失败(self, temp_duckdb_db, sample_df):
        """read_only=True 时写入操作应失败"""
        node = DuckDBNode(str(temp_duckdb_db))
        node.connect()
        node.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        node.execute("INSERT INTO test VALUES (1, 'Init')")
        node.disconnect()

        read_only_node = DuckDBNode(str(temp_duckdb_db), read_only=True)
        read_only_node.connect()

        with pytest.raises(Exception):
            read_only_node.execute("INSERT INTO test VALUES (2, 'New')")

        with pytest.raises(Exception):
            read_only_node.insert_df(sample_df, "test2")

        read_only_node.disconnect()

    def test_insert_df(self, sample_df):
        """insert_df 插入"""
        node = DuckDBNode(":memory:")
        node.connect()

        node.insert_df(sample_df, "test")
        result = node.query("SELECT * FROM test")
        assert len(result) == 5

        node.disconnect()

    def test_insert_df_replace(self, sample_df):
        """insert_df replace 模式"""
        node = DuckDBNode(":memory:")
        node.connect()

        node.insert_df(sample_df, "test")
        assert len(node.query("SELECT * FROM test")) == 5

        new_df = pd.DataFrame({'id': [100], 'name': ['Zack'], 'age': [50], 'score': [95.0]})
        node.insert_df(new_df, "test", if_exists='replace')
        result = node.query("SELECT * FROM test")
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Zack'

        node.disconnect()

    def test_断开连接(self):
        """断开连接后连接为 None"""
        node = DuckDBNode(":memory:")
        node.connect()
        assert node._conn is not None

        node.disconnect()
        assert node._conn is None

    def test_健康检查(self):
        """健康检查"""
        node = DuckDBNode(":memory:")
        node.connect()
        assert node.health_check() is True
        node.disconnect()


class TestMySQLNode:
    """MySQLNode 测试"""

    def test_初始化参数(self):
        """所有初始化参数正确存储"""
        node = MySQLNode(
            host="localhost",
            port=3307,
            user="test_user",
            passwd="test_pass",
            db="test_db",
            charset="utf8mb4",
            pool_size=20,
            pool_recycle=1800
        )

        assert node._host == "localhost"
        assert node._port == 3307
        assert node._user == "test_user"
        assert node._passwd == "test_pass"
        assert node._db == "test_db"
        assert node._charset == "utf8mb4"
        assert node._pool_size == 20
        assert node._pool_recycle == 1800

    def test_构建URL(self):
        """SQLAlchemy URL 构建正确"""
        node = MySQLNode(
            host="localhost",
            port=3306,
            user="root",
            passwd="password",
            db="mydb",
            charset="UTF8"
        )

        url = node._build_url()
        assert "mysql+pymysql://" in url
        assert "root:password" in url
        assert "localhost:3306" in url
        assert "mydb" in url
        assert "charset=UTF8" in url

    def test_show_tables方法存在(self):
        """show_tables 方法存在"""
        node = MySQLNode(host="localhost")
        assert hasattr(node, 'show_tables')
        assert callable(node.show_tables)

    def test_show_databases方法存在(self):
        """show_databases 方法存在"""
        node = MySQLNode(host="localhost")
        assert hasattr(node, 'show_databases')
        assert callable(node.show_databases)

    def test_默认参数(self):
        """默认参数正确"""
        node = MySQLNode(host="localhost")

        assert node._port == 3306
        assert node._user == ''
        assert node._passwd == ''
        assert node._db == ''
        assert node._charset == 'UTF8'
        assert node._pool_size == 10
        assert node._pool_recycle == 3600


@pytest.mark.integration
class TestMySQLNodeIntegration:
    """MySQL 集成测试（需要真实 MySQL 连接）"""

    @pytest.fixture
    def mysql_node(self):
        """MySQL 测试节点 - 优先读 conn.ini，fallback 到环境变量"""
        conn_params = _load_conn_ini("MySQL", {
            "host": "localhost",
            "port": "3306",
            "user": "root",
            "passwd": "",
            "db": "test",
        })
        node = MySQLNode(
            host=conn_params["host"],
            port=int(conn_params["port"]),
            user=conn_params["user"],
            passwd=conn_params["passwd"],
            db=conn_params["db"],
        )
        node.connect()
        yield node
        node.disconnect()

    def test_真实连接(self, mysql_node):
        """测试真实 MySQL 连接"""
        result = mysql_node.query("SELECT 1 as col")
        assert len(result) == 1
        assert result.iloc[0]['col'] == 1


class TestClickHouseNode:
    """ClickHouseNode 测试"""

    def test_初始化参数(self):
        """所有初始化参数正确存储"""
        node = ClickHouseNode(
            host="localhost",
            port=8123,
            user="test_user",
            passwd="test_pass",
            database="test_db",
            interface="http",
            pool_size=15
        )

        assert node._host == "localhost"
        assert node._port == 8123
        assert node._user == "test_user"
        assert node._passwd == "test_pass"
        assert node._database == "test_db"
        assert node._interface == "http"
        assert node._pool_size == 15

    def test_HTTP接口初始化(self):
        """HTTP 接口使用 CHBase"""
        node = ClickHouseNode(
            host="localhost",
            interface="http"
        )
        node.connect()

        assert node._http_client is not None
        assert node._client is None
        from QuantNodes.database_node.clickhouse_node import CHBase
        assert isinstance(node._http_client, CHBase)

        node.disconnect()

    def test_Native接口初始化(self):
        """Native 接口参数设置正确"""
        node = ClickHouseNode(
            host="localhost",
            port=9000,
            interface="native"
        )

        assert node._interface == "native"
        assert node._port == 9000

    def test_show_tables方法存在(self):
        """show_tables 方法存在"""
        node = ClickHouseNode(host="localhost")
        assert hasattr(node, 'show_tables')
        assert callable(node.show_tables)

    def test_show_databases方法存在(self):
        """show_databases 方法存在"""
        node = ClickHouseNode(host="localhost")
        assert hasattr(node, 'show_databases')
        assert callable(node.show_databases)

    def test_默认参数(self):
        """默认参数正确"""
        node = ClickHouseNode(host="localhost")

        assert node._port == 8123
        assert node._user == 'default'
        assert node._passwd == ''
        assert node._database == 'default'
        assert node._interface == 'http'
        assert node._pool_size == 10


@pytest.mark.integration
class TestClickHouseNodeIntegration:
    """ClickHouse 集成测试（需要真实 ClickHouse 连接）"""

    @pytest.fixture
    def clickhouse_node(self):
        """ClickHouse 测试节点 - 优先读 conn.ini，fallback 到环境变量"""
        conn_params = _load_conn_ini("ClickHouse", {
            "host": "localhost",
            "port": "8123",
            "user": "default",
            "passwd": "",
            "db": "default",
        })
        node = ClickHouseNode(
            host=conn_params["host"],
            port=int(conn_params["port"]),
            user=conn_params["user"],
            passwd=conn_params["passwd"],
            database=conn_params["db"],
        )
        node.connect()
        yield node
        node.disconnect()

    def test_真实连接_HTTP(self, clickhouse_node):
        """测试真实 ClickHouse HTTP 连接"""
        result = clickhouse_node.query("SELECT 1")
        assert len(result) == 1


class TestCSVNode:
    """CSVNode 测试"""

    def test_无过滤查询(self, temp_csv_file):
        """无 WHERE 子句时返回完整 DataFrame"""
        node = CSVNode(str(temp_csv_file))
        result = node.query()

        assert len(result) == 5
        assert list(result.columns) == ['id', 'name', 'age', 'score']

    def test_无过滤查询_带SQL参数(self, temp_csv_file):
        """带 SQL 参数但无 WHERE 子句"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * FROM nonexistent")

        assert len(result) == 5

    def test_WHERE过滤_大于(self, temp_csv_file):
        """WHERE age > 30"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE age > 30")

        assert len(result) == 3
        assert all(result['age'] > 30)

    def test_WHERE过滤_等于(self, temp_csv_file):
        """WHERE name == 'Alice'"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE name == 'Alice'")

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

    def test_WHERE过滤_多AND条件(self, temp_csv_file):
        """WHERE age > 25 and score >= 90"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE age > 25 and score >= 90")

        assert len(result) == 2
        assert all((result['age'] > 25) & (result['score'] >= 90))

    def test_WHERE过滤_多OR条件(self, temp_csv_file):
        """WHERE age < 30 or score > 90"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE age < 30 or score > 90")

        assert len(result) == 2
        assert set(result['name']) == {'Alice', 'David'}

    def test_WHERE过滤_LIKE表达式(self, temp_csv_file):
        """WHERE name.str.contains('li')"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE name.str.contains('li')")

        assert len(result) == 2
        assert set(result['name']) == {'Alice', 'Charlie'}

    def test_WHERE过滤_空结果(self, temp_csv_file):
        """WHERE 条件无匹配"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE age > 100")

        assert len(result) == 0

    def test_WHERE过滤_无空格(self, temp_csv_file):
        """WHERE 后无空格"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHEREage>30")

        assert len(result) == 3

    def test_WHERE过滤_浮点比较(self, temp_csv_file):
        """WHERE score == 85.5"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE score == 85.5")

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

    def test_WHERE过滤_不等于(self, temp_csv_file):
        """WHERE name != 'Alice'"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE name != 'Alice'")

        assert len(result) == 4

    def test_WHERE过滤_小于等于(self, temp_csv_file):
        """WHERE age <= 30"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE age <= 30")

        assert len(result) == 2
        assert all(result['age'] <= 30)

    def test_WHERE过滤_大于等于(self, temp_csv_file):
        """WHERE age >= 35"""
        node = CSVNode(str(temp_csv_file))
        result = node.query("SELECT * WHERE age >= 35")

        assert len(result) == 3
        assert all(result['age'] >= 35)

    def test_execute_抛出异常(self, temp_csv_file):
        """CSVNode 不支持 execute"""
        node = CSVNode(str(temp_csv_file))
        with pytest.raises(NotImplementedError):
            node.execute("DROP TABLE test")

    def test_insert_df_抛出异常(self, temp_csv_file, sample_df):
        """CSVNode 不支持 insert_df"""
        node = CSVNode(str(temp_csv_file))
        with pytest.raises(NotImplementedError):
            node.insert_df(sample_df, "test")

    def test_健康检查_文件存在(self, temp_csv_file):
        """文件存在时健康检查返回 True"""
        node = CSVNode(str(temp_csv_file))
        assert node.health_check() is True

    def test_健康检查_文件不存在(self):
        """文件不存在时健康检查返回 False"""
        node = CSVNode("/nonexistent/path/file.csv")
        assert node.health_check() is False

    def test_disconnect_释放内存(self, temp_csv_file):
        """disconnect 后数据为 None"""
        node = CSVNode(str(temp_csv_file))
        node.connect()
        assert node._data is not None

        node.disconnect()
        assert node._data is None


class TestParquetNode:
    """ParquetNode 测试"""

    def test_无过滤查询(self, temp_parquet_file):
        """无 WHERE 子句时返回完整 DataFrame"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query()

        assert len(result) == 5
        assert list(result.columns) == ['id', 'name', 'age', 'score']

    def test_无过滤查询_带SQL参数(self, temp_parquet_file):
        """带 SQL 参数但无 WHERE 子句"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * FROM nonexistent")

        assert len(result) == 5

    def test_WHERE过滤_大于(self, temp_parquet_file):
        """WHERE age > 30"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE age > 30")

        assert len(result) == 3
        assert all(result['age'] > 30)

    def test_WHERE过滤_等于(self, temp_parquet_file):
        """WHERE name == 'Alice'"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE name == 'Alice'")

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

    def test_WHERE过滤_多AND条件(self, temp_parquet_file):
        """WHERE age > 25 and score >= 90"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE age > 25 and score >= 90")

        assert len(result) == 2
        assert all((result['age'] > 25) & (result['score'] >= 90))

    def test_WHERE过滤_多OR条件(self, temp_parquet_file):
        """WHERE age < 30 or score > 90"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE age < 30 or score > 90")

        assert len(result) == 2
        assert set(result['name']) == {'Alice', 'David'}

    def test_WHERE过滤_LIKE表达式(self, temp_parquet_file):
        """WHERE name.str.contains('li')"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE name.str.contains('li')")

        assert len(result) == 2
        assert set(result['name']) == {'Alice', 'Charlie'}

    def test_WHERE过滤_空结果(self, temp_parquet_file):
        """WHERE 条件无匹配"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE age > 100")

        assert len(result) == 0

    def test_WHERE过滤_无空格(self, temp_parquet_file):
        """WHERE 后无空格"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHEREage>30")

        assert len(result) == 3

    def test_WHERE过滤_浮点比较(self, temp_parquet_file):
        """WHERE score == 85.5"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE score == 85.5")

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'

    def test_WHERE过滤_不等于(self, temp_parquet_file):
        """WHERE name != 'Alice'"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE name != 'Alice'")

        assert len(result) == 4

    def test_WHERE过滤_小于等于(self, temp_parquet_file):
        """WHERE age <= 30"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE age <= 30")

        assert len(result) == 2
        assert all(result['age'] <= 30)

    def test_WHERE过滤_大于等于(self, temp_parquet_file):
        """WHERE age >= 35"""
        node = ParquetNode(str(temp_parquet_file))
        result = node.query("SELECT * WHERE age >= 35")

        assert len(result) == 3
        assert all(result['age'] >= 35)

    def test_execute_抛出异常(self, temp_parquet_file):
        """ParquetNode 不支持 execute"""
        node = ParquetNode(str(temp_parquet_file))
        with pytest.raises(NotImplementedError):
            node.execute("DROP TABLE test")

    def test_insert_df_抛出异常(self, temp_parquet_file, sample_df):
        """ParquetNode 不支持 insert_df"""
        node = ParquetNode(str(temp_parquet_file))
        with pytest.raises(NotImplementedError):
            node.insert_df(sample_df, "test")

    def test_健康检查_文件存在(self, temp_parquet_file):
        """文件存在时健康检查返回 True"""
        node = ParquetNode(str(temp_parquet_file))
        assert node.health_check() is True

    def test_健康检查_文件不存在(self):
        """文件不存在时健康检查返回 False"""
        node = ParquetNode("/nonexistent/path/file.parquet")
        assert node.health_check() is False

    def test_disconnect_释放内存(self, temp_parquet_file):
        """disconnect 后数据为 None"""
        node = ParquetNode(str(temp_parquet_file))
        node.connect()
        assert node._data is not None

        node.disconnect()
        assert node._data is None

# -*- coding: utf-8 -*-
"""ClickHouse 节点

支持 HTTP 接口和官方 driver 双接口
"""
import gzip
import http.client
import json
import urllib.parse
from collections import namedtuple
from typing import Optional, List

import pandas as pd

from QuantNodes.database_node.base import BaseDBNode


ch_conn_tuple = namedtuple('ch_conn_tuple', ['host', 'port', 'user', 'passwd', 'db'])


class CHBase:
    """ClickHouse HTTP 基础实现"""

    def __init__(self, name: str, user: str = 'default', passwd: str = '123456',
                 host: str = '0.0.0.0', port: int = 8123, db: str = 'default'):
        self.name = name
        self._para = ch_conn_tuple(host, port, user, passwd, db)
        self.accepted_formats = [
            'DataFrame', 'TabSeparated', 'TabSeparatedRaw', 'TabSeparatedWithNames',
            'TabSeparatedWithNamesAndTypes', 'CSV', 'CSVWithNames', 'Values', 'Vertical', 'JSON',
            'JSONCompact', 'JSONEachRow', 'TSKV', 'Pretty', 'PrettyCompact',
            'PrettyCompactMonoBlock', 'PrettyNoEscapes', 'PrettySpace', 'XML'
        ]
        self.settings = self._merge_settings(None)
        http_get_params = {'user': self._para.user, 'password': self._para.passwd}
        http_get_params.update(self.settings)
        self.http_get_params = http_get_params

    def SHOWTABLES(self):
        """显示所有表"""
        res = self.get(f'SHOW TABLES FROM {self._para.db}').values
        return res

    def _create_conn(self):
        """创建 HTTP 连接"""
        url_str = (f"http://{self._para.user}:{self._para.passwd}@{self._para.host}:{self._para.port}")
        components = urllib.parse.urlparse(url_str)
        return http.client.HTTPConnection(components.hostname, port=components.port)

    @staticmethod
    def _check_sql_select_only(sql: str) -> None:
        """检查 SQL 是否为查询语句"""
        if sql.strip(' \n\t').lower()[:6] not in ['select', 'descri', 'show t', 'show d']:
            first_word = sql.strip(' \n\t').split(' ')[0]
            raise ValueError(
                f'"query" should start with "select" or "describe" or "show", but got "{first_word}"'
            )

    @staticmethod
    def _transfer_sql_format(sql: str, convert_to: str) -> str:
        """转换 SQL 格式"""
        clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
        query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
        return query_with_format

    def _compression_switched_request(self, query_with_format: str, conn, updated_settings, http_get_params):
        """根据设置决定是否压缩请求"""
        if updated_settings['enable_http_compression'] == 1:
            conn.request('POST', '/?' + urllib.parse.urlencode(http_get_params),
                         body=gzip.compress(query_with_format.encode()),
                         headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        else:
            conn.request('POST', '/?' + urllib.parse.urlencode(http_get_params), body=query_with_format.encode())
        return conn

    def _get_data(self, conn, updated_settings, auto_close=True):
        """获取响应数据"""
        resp = conn.getresponse()

        if resp.status == 404:
            error_message = (gzip.decompress(resp.read()).decode() if updated_settings['enable_http_compression'] == 1
                           else resp.read().decode())
            if auto_close:
                conn.close()
            raise ValueError(error_message)
        elif resp.status == 401:
            if auto_close:
                conn.close()
            raise ConnectionRefusedError(resp.reason + '. The username or password is incorrect.')
        else:
            if resp.status != 200:
                error_message = (gzip.decompress(resp.read()).decode() if updated_settings['enable_http_compression'] == 1
                               else resp.read().decode())
                if auto_close:
                    conn.close()
                raise NotImplementedError(f'Unknown Error: status: {resp.status}, reason: {resp.reason}, message: {error_message}')

        total = bytes()
        while not resp.isclosed():
            total += resp.read(300 * 1024)
        if auto_close:
            conn.close()
        return gzip.decompress(total).decode() if updated_settings['enable_http_compression'] == 1 else total.decode()

    @staticmethod
    def _load_into_pd(ret_value: str, convert_to: str) -> pd.DataFrame:
        """将返回值转换为 DataFrame"""
        if convert_to.lower() == 'dataframe':
            result_dict = json.loads(ret_value, strict=False)
            dataframe = pd.DataFrame.from_records(
                result_dict['data'],
                columns=[i['name'] for i in result_dict['meta']]
            )
            for i in result_dict['meta']:
                if i['type'] in ['DateTime', 'Nullable(DateTime)']:
                    dataframe[i['name']] = pd.to_datetime(dataframe[i['name']])
            return dataframe
        return ret_value

    def get(self, sql: str, convert_to: str = 'DataFrame', auto_close: bool = True):
        """执行查询并返回结果"""
        conn = self._create_conn()
        self._check_sql_select_only(sql)
        updated_settings = self.settings
        query_with_format = self._transfer_sql_format(sql, convert_to)
        conn = self._compression_switched_request(query_with_format, conn, updated_settings, self.http_get_params)
        ret_value = self._get_data(conn, updated_settings, auto_close=auto_close)
        return self._load_into_pd(ret_value, convert_to)

    @staticmethod
    def _merge_settings(settings):
        """合并设置"""
        updated_settings = {
            'enable_http_compression': 1,
            'send_progress_in_http_headers': 0,
            'log_queries': 1,
            'connect_timeout': 10,
            'receive_timeout': 300,
            'send_timeout': 300,
            'output_format_json_quote_64bit_integers': 0,
            'wait_end_of_query': 0
        }
        if settings is not None:
            invalid_keys = list(set(settings.keys()) - set(updated_settings.keys()))
            if invalid_keys:
                raise ValueError(f'setting "{invalid_keys[0]}" is invalid')
            updated_settings.update(settings)

        for i in updated_settings:
            updated_settings[i] = 1 if updated_settings[i] is True else 0 if updated_settings[i] is False else updated_settings[i]
        return updated_settings

    def get_describe_table(self, db: str, table: str) -> pd.DataFrame:
        """获取表结构"""
        return self.get(f'DESCRIBE TABLE {db}.{table}', auto_close=True)

    def query(self, sql: str) -> pd.DataFrame:
        """执行查询"""
        return self.get(sql, convert_to='DataFrame')


class ClickHouseNode(BaseDBNode):
    """ClickHouse 数据库节点

    支持 HTTP 接口和官方 driver 双接口

    Args:
        host: 主机地址
        port: 端口 (HTTP 默认 8123，Native 默认 9000)
        user: 用户名 (默认 default)
        passwd: 密码
        database: 数据库名 (默认 default)
        interface: 接口类型 ('http' 或 'native'，默认 'http')
        pool_size: 连接池大小 (默认 10，可配置)

    Example:
        >>> # HTTP 接口
        >>> node = ClickHouseNode(
        ...     host="localhost",
        ...     user="default",
        ...     passwd="",
        ...     database="default"
        ... )

        >>> # Native 接口
        >>> node = ClickHouseNode(
        ...     host="localhost",
        ...     port=9000,
        ...     interface="native"
        ... )
    """

    def __init__(self, host: str, port: int = 8123,
                 user: str = 'default', passwd: str = '',
                 database: str = 'default',
                 interface: str = 'http',
                 pool_size: int = 10):
        self._host = host
        self._port = port
        self._user = user
        self._passwd = passwd
        self._database = database
        self._interface = interface
        self._pool_size = pool_size
        self._client = None
        self._http_client = None

    def connect(self):
        """建立连接"""
        if self._interface == 'native':
            import clickhouse_connect
            self._client = clickhouse_connect.get_client(
                host=self._host,
                port=self._port,
                username=self._user,
                password=self._passwd,
                database=self._database,
            )
        else:
            self._http_client = CHBase(
                name=self._database,
                user=self._user,
                passwd=self._passwd,
                host=self._host,
                port=self._port,
                db=self._database,
            )
        return self._client or self._http_client

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行查询"""
        client = self._client or self._http_client or self.connect()
        if self._interface == 'native':
            result = client.query(sql)
            if hasattr(result, 'result_rows'):
                return pd.DataFrame(result.result_rows, columns=result.column_names)
            return result
        else:
            return client.get(sql, convert_to='DataFrame')

    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行 DDL/DML"""
        client = self._client or self._http_client or self.connect()
        if self._interface == 'native':
            client.command(sql)
            return 0
        else:
            client.query(sql)
            return 0

    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = 'append') -> int:
        """插入 DataFrame"""
        client = self._client or self._http_client or self.connect()
        if self._interface == 'native':
            client.insert_df(table, df)
        else:
            client.insert(df, self._database, table)
        return len(df)

    def disconnect(self) -> None:
        """关闭连接"""
        if self._client:
            if self._interface == 'native':
                self._client.close()
            self._client = None
        self._http_client = None

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.query("SELECT 1")
            return True
        except Exception:
            return False

    def show_tables(self) -> List[str]:
        """列出所有表"""
        result = self.query(f"SHOW TABLES FROM {self._database}")
        return result.iloc[:, 0].tolist()

    def show_databases(self) -> List[str]:
        """列出所有数据库"""
        result = self.query("SHOW DATABASES")
        return result.iloc[:, 0].tolist()

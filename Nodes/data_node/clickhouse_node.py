# coding=utf-8


"""
Created on Sun Aug 18 12:17:40 2019

@author: lee1984
"""

import gzip
import json
import urllib
import warnings
from collections import ChainMap
from collections import namedtuple
from functools import lru_cache

import grequests
import numpy as np
import pandas as pd
import requests

from Nodes.data_node._ConnectionParser import ConnectionParser
from Nodes.utils_node.lazy_load import LazyInit

# from Nodes.utils_node.timer import timer

node = namedtuple('clickhouse', ['host', 'port', 'user', 'password', 'db'])


class ClickHouseDBNode(LazyInit):  # lazy load to improve loading speed
    def __init__(self, settings: (str, dict) = None):
        super(ClickHouseDBNode, self).__init__()
        if isinstance(settings, str):
            db_type, settings = ConnectionParser.parser(settings)
            if db_type.lower() != 'clickhouse':
                raise ValueError('settings is not for clickhouse!')
        elif isinstance(settings, dict):
            pass
        else:
            raise ValueError('settings must be str or dict')

        self.db_name = settings['db']
        # settings, conn = ConnectionParser.checker_multi_and_create(db, settings, target_db_type='ClickHouse')
        self._settings = settings
        self._conn = ClickHouseBaseNode(settings['db'], **settings)
        self.is_closed = False
        # for table in self.tables:
        #     if table != 'tables':
        #         try:
        #             setattr(self, table, ClickHouseTableNode(table, self._conn))
        #         except Exception as e:
        #             print(str(e))
        #             pass
        self._setup_()

        self._query = self._conn.query

    def close(self):
        self._conn.close()
        self.is_closed = True

    def __exit__(self):
        self.close()

    @property
    def tables(self):
        # tables = self._conn.tables
        return self._conn.tables

    def _setup_(self):
        for table in self.tables:
            if table not in dir(self):
                try:
                    table_node = ClickHouseTableNode(table, **self._settings)

                    self.__setitem__(table, table_node)

                    # setattr(self, table, ClickHouseTableNode(table, **self._settings))
                except Exception as e:
                    print(str(e))
                    pass
            else:
                raise ValueError(f'found a table named {table} which is a method or attribute, please rename it')

    def __getitem__(self, table: str):
        return getattr(self, table)

    def __setitem__(self, table: str, table_obj):
        setattr(self, table, table_obj)


class ClickHouseDBPool(LazyInit):  # lazy load to improve loading speed
    def __init__(self, settings: (str, dict) = None):
        super(ClickHouseDBPool, self).__init__()
        if isinstance(settings, str):
            db_type, settings = ConnectionParser.parser(settings)
            if db_type.lower() != 'clickhouse':
                raise ValueError('settings is not for clickhouse!')
        elif isinstance(settings, dict):
            pass
        else:
            raise ValueError('settings must be str or dict')
        # settings, conn = ConnectionParser.checker_multi_and_create(db, settings, target_db_type='ClickHouse')
        self._settings = settings
        self._conn = ClickHouseBaseNode(settings['db'], **settings)
        self.is_closed = False
        # for table in self.tables:
        #     if table != 'tables':
        #         try:
        #             setattr(self, table, ClickHouseTableNode(table, self._conn))
        #         except Exception as e:
        #             print(str(e))
        #             pass
        self._setup_()
        self._query = self._conn.query

    def close(self):
        self._conn.close()
        self.is_closed = True

    def __exit__(self):
        self.close()

    @property
    def databases(self):
        # tables = self._conn.tables
        return self._conn.databases

    def _setup_(self):
        for database in self.databases:
            if database not in dir(self):
                try:
                    db_node = ClickHouseDBNode(self._settings)

                    self.__setitem__(database, db_node)

                    # setattr(self, table, ClickHouseTableNode(table, **self._settings))
                except Exception as e:
                    print(str(e))
                    pass
            else:
                raise ValueError(f'found a table named {database} which is a method or attribute, please rename it')

    def __getitem__(self, table: str):
        return getattr(self, table)

    def __setitem__(self, table: str, table_obj):
        setattr(self, table, table_obj)


class _ClickHouseNodeBaseTool(object):

    @staticmethod
    def _check_df_and_dump(df, describe_table):
        non_nullable_columns = list(describe_table[~describe_table['type'].str.startswith('Nullable')]['name'])
        integer_columns = list(describe_table[describe_table['type'].str.contains('Int', regex=False)]['name'])
        missing_in_df = {i: np.where(df[i].isnull(), 1, 0).sum() for i in non_nullable_columns}

        df_columns = list(df.columns)
        each_row = df.to_dict(orient='records')
        for i in missing_in_df:
            if missing_in_df[i] > 0:
                raise ValueError('"{0}" is not a nullable column, missing values are not allowed.'.format(i))

        for row in each_row:
            for col in df_columns:
                if pd.isnull(row[col]):
                    row[col] = None
                else:
                    if col in integer_columns:
                        try:
                            row[col] = int(row[col])
                        except Exception as e:
                            print(str(e))
                            raise ValueError('Column "{0}" is {1}, while value "{2}"'.format(col,
                                                                                             describe_table[
                                                                                                 describe_table[
                                                                                                     'name'] == col].iloc[
                                                                                                 0]['type'], row[col]) + \
                                             ' in the dataframe column cannot be converted to Integer.')
            yield json.dumps(row, ensure_ascii=False)

    @staticmethod
    def _check_settings(settings, updated_settings):
        if settings is not None:
            invalid_setting_keys = list(set(settings.keys()) - set(updated_settings.keys()))
            if len(invalid_setting_keys) > 0:
                raise ValueError('setting "{0}" is invalid, valid settings are: {1}'.format(
                    invalid_setting_keys[0], ', '.join(updated_settings.keys())))
            else:
                pass

    @classmethod
    def _merge_settings(cls, settings,
                        updated_settings={'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
                                          'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
                                          'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
                                          'wait_end_of_query': 0}):

        if settings is not None:
            cls._check_settings(settings, updated_settings)
            updated_settings.update(settings)
        else:
            pass

        return {k: v * 1 if isinstance(v, bool) else v for k, v in updated_settings.items()}

    @staticmethod
    def _check_sql_select_only(sql):
        if sql.strip(' \n\t').lower()[:4] not in ['sele', 'desc', 'show', 'opti', 'crea']:
            raise ValueError('"query" should start with "select" or "describe" or "show", ' + \
                             'while the provided "query" starts with "{0}"'.format(sql.strip(' \n\t').split(' ')[0]))

    @staticmethod
    def _transfer_sql_format(sql, convert_to, transfer_sql_format=True):
        if transfer_sql_format:
            clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
            query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
            return query_with_format
        else:
            return sql

    @staticmethod
    def _load_into_pd(ret_value, convert_to):
        if convert_to.lower() == 'dataframe':
            result_dict = json.loads(ret_value, strict=False)
            dataframe = pd.DataFrame.from_records(result_dict['data'], columns=[i['name'] for i in result_dict['meta']])

            for i in result_dict['meta']:
                if i['type'] in ['DateTime', 'Nullable(DateTime)']:
                    dataframe[i['name']] = pd.to_datetime(dataframe[i['name']])
            ret_value = dataframe
        return ret_value


class _CreateTableTools(object):
    @staticmethod
    def _check_table_exists(db, table, obj):
        """

        :param db:
        :param table:
        :param obj:
        :return: exists =true not exists = False
        """
        sql = f"show tables from {db} like '{table}' "
        res = obj.query(sql)
        if res.empty:
            return False
        else:
            return True

    @staticmethod
    def _translate_dtype(sdf):
        if 'type' in sdf.columns and 'name' in sdf.columns:
            dtypes_series = sdf.set_index('name')['type']
            # df_type = 'type'
            return dtypes_series.replace('object', 'String').replace('datetime64[ns]', 'Datetime').map(lambda x: str(x))
        else:
            dtypes_series = sdf.dtypes
            # df_type ='data'

            return dtypes_series.replace('object', 'String').replace('datetime64[ns]', 'Datetime').map(
                lambda x: str(x).capitalize())

    @classmethod
    def _create_table_dtype(cls, db, table, sdf, key_cols, engine_type='ReplacingMergeTree', extra_format_dict=None):
        if extra_format_dict is None:
            extra_format_dict = {}
        dtypes_df = cls._translate_dtype(sdf)
        cols_def = ','.join([f"{name} {dtype}" if name not in extra_format_dict.keys() else "{}.{}".format(name, str(
            extra_format_dict.get(name))) for name, dtype in dtypes_df.to_dict().items()])
        order_by_cols = ','.join(key_cols)

        DB_TABLE = f"{db}.{table}"

        ORDER_BY_CLAUSE = f"ORDER BY ( {order_by_cols} )"

        base = SQLBuilder.raw_create_ReplacingMergeTree_table_sql(DB_TABLE, cols_def, ORDER_BY_CLAUSE,
                                                                  ENGINE_TYPE=engine_type)

        # base = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type} ORDER BY ({order_by_cols}) SETTINGS index_granularity = 8192 "
        return base

    @classmethod
    def _create_table(cls, obj: object, db: str, table: str, sql: str, key_cols: list,
                      engine_type: str = 'ReplacingMergeTree',
                      extra_format_dict: (dict, None) = None) -> bool:
        """

        :param obj:
        :param db:
        :param table:
        :param sql:
        :param key_cols:
        :param engine_type:
        :param extra_format_dict:
        :return:
        """

        if extra_format_dict is None:
            extra_format_dict = {}

        if isinstance(obj, ClickHouseDBNode):
            if len(obj.tables) == 0:
                warnings.warn('new database! it is no safe to create table!')
            query_func = obj.query
        elif isinstance(obj, ClickHouseDBPool):
            query_func = obj.query
        else:
            query_func = obj.query

        describe_sql = f'describe ( {sql} limit 1)'

        exist_status: bool = cls._check_table_exists(db, table, obj)
        if exist_status:
            print('table:{table} already exists!')
        else:
            print('will create {table} at {db}')
            dtypes_df = query_func(describe_sql)
            cls._create_table_dtype(db, table, dtypes_df, key_cols, engine_type=engine_type,
                                    extra_format_dict=extra_format_dict)
        return exist_status


# class _ClickHouseBaseNodeAsyncExt(object):
#
#
#     def _async_request(self, sql_list, convert_to='dataframe', todf=True, transfer_sql_format=True, return_sql=True,
#                        auto_switch=True,max_async_query_once=5):
#
#         def exception_handler(request, exception):
#             print("Request failed")
#
#         if not self._async:
#             raise ValueError("must manually switch to async mode")
#         tasks = (self._request_unit_(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format) for sql in
#                  sql_list)
#         if return_sql:
#             res_list = zip(sql_list,
#                            grequests.map(tasks, exception_handler=exception_handler),
#                            size=max_async_query_once)
#             for sql, resp in res_list:
#                 if todf:
#                     d = self._load_into_pd(resp.content, convert_to)
#                     yield sql, d
#                 else:
#                     yield sql, resp.content
#         else:
#             for resp in grequests.map(tasks, exception_handler=exception_handler, size=max_async_query_once):
#                 if todf:
#                     d = self._load_into_pd(resp.content, convert_to)
#                     yield d
#                 else:
#                     yield resp.content
#         if auto_switch:
#             self._async = False
#
#     # @timer
#     def _async_query(self, sql_list: (str, list)):
#         get_query = all(map(
#             lambda sql: sql.lower().startswith('select') or sql.lower().startswith('show') or sql.lower().startswith(
#                 'desc'), sql_list))
#         insert_query = all(map(lambda sql: sql.lower().startswith('insert') or sql.lower().startswith(
#             'optimize') or sql.lower().startswith('create'), sql_list))
#
#         if get_query:
#             todf = True
#         elif insert_query:
#             todf = False
#         else:
#             raise ValueError('Unknown sql! current only accept select, insert, show, optimize')
#         res = list(self._async_request(sql_list, convert_to='dataframe', todf=todf, return_sql=False))
#         return res
#

class _ClickHouseBaseNode(_ClickHouseNodeBaseTool):
    def __init__(self, name, **db_settings):
        """

        :param name:
        :param db_settings:
        """

        self._db_settings = db_settings

        self.db_name: str = db_settings['db']

        self._para = node(db_settings['host'], db_settings['port'], db_settings['user'],
                          db_settings['password'], db_settings['db'])

        self._base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port))

        self._default_settings = {'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
                                  'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
                                  'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
                                  'wait_end_of_query': 0}

        http_settings = self._merge_settings(None, updated_settings=self._default_settings)
        http_settings.update({'user': db_settings['user'], 'password': db_settings['password']})
        self.http_settings = http_settings
        # clickhouse_node = namedtuple('clickhouse_node', ['host', 'port', 'user', 'passwd', 'db'])

        self.accepted_formats = ['DataFrame', 'TabSeparated', 'TabSeparatedRaw', 'TabSeparatedWithNames',
                                 'TabSeparatedWithNamesAndTypes', 'CSV', 'CSVWithNames', 'Values', 'Vertical', 'JSON',
                                 'JSONCompact', 'JSONEachRow', 'TSKV', 'Pretty', 'PrettyCompact',
                                 'PrettyCompactMonoBlock', 'PrettyNoEscapes', 'PrettySpace', 'XML']
        self._async = False
        self._session = None
        self.max_async_query_once = 5
        self.is_closed = False
        self._test_connection_()

    def _test_connection_(self):
        ret_value = self.session.get(self._base_url)

        print('test_connection: ', ret_value.text.strip())

    # @property
    # def __coroutine_session__(self):
    #     return grequests

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        else:
            if self.is_closed:
                raise ValueError('session is closed!')
            else:
                pass
        return self._session  # grequests.AsyncRequest

    def __getitem__(self, sql: str):
        return self.query(sql)

    def __setitem__(self, db_table: str, df):
        if '.' in db_table:
            pass
            db, table = db_table.split('.')
        else:
            raise ValueError(f'get unknown db_table : {db_table}')
        self._df_insert_(db, table, df)

    def close(self):
        self.session.close()
        self.is_closed = True  # self._session.is_closed()

    def __exit__(self):
        self.close()

    def _describe_(self, db: str, table: str):
        describe_sql = 'describe table {}.{}'.format(db, table)
        describe_table = self._request(describe_sql)
        # non_nullable_columns = list(describe_table[~describe_table['type'].str.startswith('Nullable')]['name'])
        # integer_columns = list(describe_table[describe_table['type'].str.contains('Int', regex=False)]['name'])
        # missing_in_df = {i: np.where(df[i].isnull(), 1, 0).sum() for i in non_nullable_columns}
        #
        # df_columns = list(df.columns)
        # each_row = df.to_dict(orient='records')
        # del df
        return describe_table  # , integer_columns, non_nullable_columns

    def _request_unit_(self, sql: str, convert_to: str = 'dataframe', transfer_sql_format: bool = True):
        sql2 = self._transfer_sql_format(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)

        if self._async:
            session = grequests
        else:
            session = self.session

        if self.http_settings['enable_http_compression'] == 1:
            url = self._base_url + urllib.parse.urlencode(self.http_settings)
            resp = session.post(url,
                                data=gzip.compress(sql2.encode()),
                                headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        else:
            # settings = self.http_settings.copy()
            # settings.update({'query': sql2})
            url = self._base_url + urllib.parse.urlencode(ChainMap(self.http_settings, {'query': sql2}))
            resp = session.post(url)
        return resp

    # @timer
    @lru_cache(maxsize=10)
    def _request(self, sql: str, convert_to: str = 'dataframe', todf: bool = True, transfer_sql_format: bool = True):
        if self._async:
            self._async = False

        #     raise ValueError('current async mode will disable sync mode!')
        # sql2 = self._transfer_sql_format(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)
        #
        # if self.http_settings['enable_http_compression'] == 1:
        #     url = self._base_url + urllib.parse.urlencode(self.http_settings)
        #     resp = self.session.post(url,
        #                              data=gzip.compress(sql2.encode()),
        #                              headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        # else:
        #     settings = self.http_settings.copy()
        #     settings.update({'query': sql2})
        #     url = self._base_url + urllib.parse.urlencode(settings)
        #     resp = self.session.post(url)
        resp = self._request_unit_(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)
        if todf:
            d = self._load_into_pd(resp.content, convert_to)
            return d
        else:
            return resp.content

    def _async_request(self, sql_list, convert_to='dataframe', todf=True, transfer_sql_format=True, auto_switch=True):

        def exception_handler(request, exception):
            print("Request failed")

        if not self._async:
            if auto_switch:
                self._async = True
            else:
                raise ValueError("must manually switch to async mode")
        tasks = (self._request_unit_(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format) for sql in
                 sql_list)

        for resp in grequests.map(tasks, exception_handler=exception_handler, size=self.max_async_query_once):
            if todf:
                d = self._load_into_pd(resp.content, convert_to)
                yield d
            else:
                yield resp.content
        if auto_switch:
            self._async = False

    # @timer
    def _async_query(self, sql_list: (list,)):
        get_query = all(map(
            lambda sql: sql.lower().startswith('select') or sql.lower().startswith('show') or sql.lower().startswith(
                'desc'), sql_list))
        insert_query = all(map(lambda sql: sql.lower().startswith('insert') or sql.lower().startswith(
            'optimize') or sql.lower().startswith('create'), sql_list))

        if get_query:
            todf = True
            # res = self._request(sql, convert_to='dataframe', todf=True)
        elif insert_query:
            todf = False
            # res = self._request(sql, convert_to='dataframe', todf=False)
        else:
            raise ValueError('Unknown sql! current only accept select, insert, show, optimize')
        res = list(self._async_request(sql_list, convert_to='dataframe', todf=todf))
        return res

    def _df_insert_(self, db: str, table: str, df: pd.DataFrame):
        describe_table = self._describe_(db, table)
        query_with_format = 'insert into {0} format JSONEachRow \n{1}'.format('{}.{}'.format(db, table), '\n'.join(
            self._check_df_and_dump(df, describe_table)))
        self._request(query_with_format, convert_to='dataframe', todf=False, transfer_sql_format=False)

    def execute(self, sql: str):
        if isinstance(sql, str):
            return self._execute(sql)
        elif isinstance(sql, (list, tuple)):
            max_queries = self.max_async_query_once * 2
            if len(sql) > max_queries:
                raise ValueError(f'too many queries,please reduce to less than {max_queries}!')
            return self._async_query(sql)
        else:
            raise ValueError('sql must be str or list or tuple')

    def _execute(self, sql: str):

        if sql.lower().startswith('select') or sql.lower().startswith('show') or sql.lower().startswith('desc'):
            todf = True
            transfer_sql_format = True

            # res = self._request(sql, convert_to='dataframe', todf=True)
        elif sql.lower().startswith('insert') or sql.lower().startswith('optimize') or sql.lower().startswith('create'):
            todf = False
            transfer_sql_format = False
            # res = self._request(sql, convert_to='dataframe', todf=False)
        else:
            raise ValueError('Unknown sql! current only accept select, insert, show, optimize')
        res = self._request(sql, convert_to='dataframe', todf=todf, transfer_sql_format=transfer_sql_format)
        return res

    def query(self, sql: str, optimize: bool = False):
        """

        :param sql:
        :param optimize:
        :return:
        """
        if isinstance(sql, str) and sql.lower().startswith('insert into'):
            db_table = sql.lower().split('insert into ')[-1].split(' ')[0]
        else:
            db_table = 'no'

        res = self.execute(sql)
        try:
            if optimize and db_table != 'no':
                self.execute('optimize table {db_table}')
            else:
                pass
        except Exception as e:
            print(f'auto optimize process failure, please manual optimize on {db_table}')

        finally:
            return res


class ClickHouseBaseNode(_ClickHouseBaseNode, _CreateTableTools):
    def __init__(self, name, **settings):
        super(ClickHouseBaseNode, self).__init__(name, **settings)

    @property
    def tables(self):
        sql = 'SHOW TABLES FROM {db}'.format(db=self._para.db)
        res = self.execute(sql).values.ravel().tolist()
        return res

    @property
    def databases(self):
        sql = 'SHOW DATABASES'
        res = self.execute(sql).values.ravel().tolist()
        return res

    @property
    def _database_exist_status(self):
        return self.db_name in self.databases

    def create_table(self, db: str, table: str, select_sql: str, keys_cols: list,
                     table_engine: str = 'ReplacingMergeTree', extra_format_dict: bool = None,
                     return_status: bool = True):
        status = self._create_table(obj=self, db=db, table=table, sql=select_sql,
                                    keys_cols=keys_cols,
                                    table_engine=table_engine, extra_format_dict=extra_format_dict)
        if return_status:
            return status

    def check_table_exists(self, db: str, table: str):
        """

        :param db:
        :param table:
        :param obj:
        :return: exists =true not exists = False
        """
        sql = f"show tables from {db} like '{table}' "
        res = self.query(sql)
        if res.empty:
            return False
        else:
            return True


class ClickHouseTableNode(ClickHouseBaseNode):
    def __init__(self, table_name, **settings):
        super(ClickHouseTableNode, self).__init__(table_name, **settings)
        self.table_name = table_name

    @property
    def _table_exist_status(self):
        return self.table_name in self.tables

    @property
    def table_structure(self):
        if self._table_exist_status:
            return self._describe_(self.db_name, self.table_name)
        else:
            raise ValueError(f'table: {self.table_name} may not be exists!')

    @property
    def columns(self):
        if self._table_exist_status:
            return self.table_structure['name'].values.tolist()
        else:
            raise ValueError(f'table: {self.table_name} may not be exists!')

    def head(self, top=10):
        sql = f'select * from {self.db_name}.{self.table_name} limit {top}'
        return self.execute(sql)


class TableEngineCreator(object):
    @staticmethod
    def _assemble_cols_2_clause(prefix, cols, default=''):
        if cols is None:
            return default
        else:
            cols_str = ','.join(cols)
            return f"{prefix} ( {cols_str} ) "

    @classmethod
    def ReplacingMergeTree_creator(cls, DB_TABLE, cols_def, order_by_cols,
                                   sample_by_cols=None,
                                   ON_CLUSTER='', partition_by_cols=None, primary_by_cols=None):

        order_by_cols_str = ','.join(order_by_cols)
        ORDER_BY_CLAUSE = f'ORDER BY ( {order_by_cols_str} )'

        SAMPLE_CLAUSE = cls._assemble_cols_2_clause('SAMPLE BY', sample_by_cols, default='')

        PRIMARY_BY_CLAUSE = cls._assemble_cols_2_clause('PRIMARY BY', primary_by_cols, default='')
        # if primary_by_cols is not None:
        #     primary_by_cols_str = ','.join(primary_by_cols)
        #     PRIMARY_BY_CLAUSE = f'PRIMARY BY ( {primary_by_cols_str} )'
        # else:
        #     PRIMARY_BY_CLAUSE = ''

        PARTITION_by_CLAUSE = cls._assemble_cols_2_clause('PARTITION BY', partition_by_cols, default='')

        # if partition_by_cols is not None:
        #     partition_by_cols_str = ','.join(partition_by_cols)
        #     PARTITION_by_CLAUSE = f'PARTITION BY ( {partition_by_cols_str} )'
        # else:
        #     PARTITION_by_CLAUSE = ''

        return cls.raw_create_ReplacingMergeTree_table_sql(DB_TABLE, cols_def, ORDER_BY_CLAUSE,
                                                           PRIMARY_BY_CLAUSE=PRIMARY_BY_CLAUSE,
                                                           SAMPLE_CLAUSE=SAMPLE_CLAUSE,
                                                           ENGINE_TYPE='ReplacingMergeTree', ON_CLUSTER=ON_CLUSTER,
                                                           PARTITION_by_CLAUSE=PARTITION_by_CLAUSE)

    @staticmethod
    def raw_create_ReplacingMergeTree_table_sql(DB_TABLE, cols_def, ORDER_BY_CLAUSE,
                                                PRIMARY_BY_CLAUSE='', SAMPLE_CLAUSE='',
                                                ENGINE_TYPE='ReplacingMergeTree', ON_CLUSTER='', PARTITION_by_CLAUSE='',
                                                TTL=''
                                                ):
        ## TODO add ttl expr at future
        """

        :param ON_CLUSTER:
        :param SAMPLE_CLAUSE:
        :param PRIMARY_BY_CLAUSE:
        :param PARTITION_by_CLAUSE:
        :param DB_TABLE:
        :param cols_def:
        :param ORDER_BY_CLAUSE:
        :param ENGINE_TYPE:
        :return:
        """
        """CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
            (
                name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
                name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
                ...
            ) ENGINE = ReplacingMergeTree([ver])
            [PARTITION BY expr]
            [ORDER BY expr]
            [PRIMARY KEY expr]
            [SAMPLE BY expr]
            [SETTINGS name=value, ...]"""

        maid_body = f"CREATE TABLE IF NOT EXISTS {DB_TABLE} {ON_CLUSTER} ( {cols_def} ) ENGINE = {ENGINE_TYPE}"

        settings = "SETTINGS index_granularity = 8192"
        conds = f"{PARTITION_by_CLAUSE} {ORDER_BY_CLAUSE} {PRIMARY_BY_CLAUSE} {SAMPLE_CLAUSE}"

        base = f"{maid_body} {conds}  {settings}"
        return base


class SQLBuilder(TableEngineCreator):
    @staticmethod
    def _assemble_sample(sample=None):
        if sample is None:
            SAMPLE_CLAUSE = ''
        else:
            SAMPLE_CLAUSE = f'SAMPLE {sample}'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_array_join(array_join_list=None):
        if array_join_list is None:
            ARRAY_JOIN_CLAUSE = ''
        else:
            array_join = ','.join(array_join_list)
            ARRAY_JOIN_CLAUSE = f'ARRAY JOIN {array_join}'
        return ARRAY_JOIN_CLAUSE

    @staticmethod
    def _assemble_join(join_info_dict=None):

        if join_info_dict is None:
            JOIN_CLAUSE = ''
        else:
            join_type = join_info_dict.get('type')
            on_ = join_info_dict.get('ON')
            using_ = join_info_dict.get('USING')

            if join_type is None:
                raise ValueError('join_info_dict cannot locate join_type condition')

            if on_ is None:
                if using_ is None:
                    raise ValueError('join_info_dict cannot locate ON or USING condition')
                else:
                    JOIN_CLAUSE = f'{join_type} USING ({using_})'
            else:
                JOIN_CLAUSE = f'{join_type} ON {on_}'
        return JOIN_CLAUSE

    @staticmethod
    def _assemble_where_like(a_list, prefix='WHERE'):
        if a_list is None:
            SAMPLE_CLAUSE = ''
        else:
            a_list_str = 'and'.join(a_list)
            SAMPLE_CLAUSE = f'{prefix} {a_list_str}'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_group_by(group_by_cols=None):
        if group_by_cols is None:
            SAMPLE_CLAUSE = ''
        else:
            group_by_cols_str = ','.join(group_by_cols)
            SAMPLE_CLAUSE = f'GROUP BY ({group_by_cols_str})'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_order_by(order_by_cols=None):
        if order_by_cols is None:
            SAMPLE_CLAUSE = ''
        else:
            order_by_cols_str = ','.join(order_by_cols)
            SAMPLE_CLAUSE = f'ORDER BY ({order_by_cols_str})'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_limit_by(limit_n_by_dict=None):

        if limit_n_by_dict is None:
            SAMPLE_CLAUSE = ''
        else:
            N = limit_n_by_dict['N']
            order_by_cols_str = ','.join(limit_n_by_dict['limit_by_cols'])
            SAMPLE_CLAUSE = f'LIMIT {N} BY ({order_by_cols_str})'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_limit(limit_n=None):

        if limit_n is None:
            SAMPLE_CLAUSE = ''
        else:
            SAMPLE_CLAUSE = f'LIMIT {limit_n} '
        return SAMPLE_CLAUSE

    @classmethod
    def create_select_sql(cls, DB_TABLE: str, cols: list,
                          sample: (int, float, None) = None,
                          array_join: (list, None) = None, join: (dict, None) = None,
                          prewhere: (list, None) = None, where: (list, None) = None, having: (list, None) = None,
                          group_by: (list, None) = None,
                          order_by: (list, None) = None, limit_by: (dict, None) = None,
                          limit: (int, None) = None) -> str:
        """

        :param having: str ["r1 >1 and r2 <2"]
        :param DB_TABLE: str default.test
        :param cols: list [ r1,r2,r3 ]
        :param sample: str 0.1 or 1000
        :param array_join: list ['arrayA as a','arrayB as b']
        :param join: dict {'type':'all left join','USING' : "r1,r2"}
        :param prewhere: str ["r1 >1 and r2 <2"]
        :param where: str ["r1 >1.5 and r2 <1.3"]
        :param group_by: list ['r1','r2']
        :param order_by: list ['r1 desc','r2 desc']
        :param limit_by: dict {'N':10,'limit_by_cols':['r1','r2']}
        :param limit: int 100
        :return:  str
        """

        SELECT_CLAUSE = ','.join(cols)
        SAMPLE_CLAUSE = cls._assemble_sample(sample=sample)
        ARRAY_JOIN_CLAUSE = cls._assemble_array_join(array_join_list=array_join)
        JOIN_CLAUSE = cls._assemble_join(join)
        PREWHERE_CLAUSE = cls._assemble_where_like(prewhere, prefix='PREWHERE')
        WHERE_CLAUSE = cls._assemble_where_like(where, prefix='WHERE')
        HAVING_CLAUSE = cls._assemble_where_like(having, prefix='HAVING')
        GROUP_BY_CLAUSE = cls._assemble_group_by(group_by)
        ORDER_BY_CLAUSE = cls._assemble_order_by(order_by)
        LIMIT_N_CLAUSE = cls._assemble_limit_by(limit_by)
        LIMIT_CLAUSE = cls._assemble_limit(limit)

        return cls.raw_create_select_sql(SELECT_CLAUSE, DB_TABLE, SAMPLE_CLAUSE, ARRAY_JOIN_CLAUSE, JOIN_CLAUSE,
                                         PREWHERE_CLAUSE, WHERE_CLAUSE, GROUP_BY_CLAUSE, HAVING_CLAUSE, ORDER_BY_CLAUSE,
                                         LIMIT_N_CLAUSE, LIMIT_CLAUSE)

    @staticmethod
    def raw_create_select_sql(SELECT_CLAUSE: str, DB_TABLE: str, SAMPLE_CLAUSE: str, ARRAY_JOIN_CLAUSE: str,
                              JOIN_CLAUSE: str, PREWHERE_CLAUSE: str, WHERE_CLAUSE: str, GROUP_BY_CLAUSE: str,
                              HAVING_CLAUSE: str, ORDER_BY_CLAUSE: str, LIMIT_N_CLAUSE: str, LIMIT_CLAUSE: str):
        """

        :param SELECT_CLAUSE:
        :param DB_TABLE:
        :param SAMPLE_CLAUSE:
        :param ARRAY_JOIN_CLAUSE:
        :param JOIN_CLAUSE:
        :param PREWHERE_CLAUSE:
        :param WHERE_CLAUSE:
        :param GROUP_BY_CLAUSE:
        :param HAVING_CLAUSE:
        :param ORDER_BY_CLAUSE:
        :param LIMIT_N_CLAUSE:
        :param LIMIT_CLAUSE:
        :return:
        """
        """SELECT [DISTINCT] expr_list
                    [FROM [db.]table | (subquery) | table_function] [FINAL]
                    [SAMPLE sample_coeff]
                    [ARRAY JOIN ...]
                    [GLOBAL] ANY|ALL INNER|LEFT JOIN (subquery)|table USING columns_list
                    [PREWHERE expr]
                    [WHERE expr]
                    [GROUP BY expr_list] [WITH TOTALS]
                    [HAVING expr]
                    [ORDER BY expr_list]
                    [LIMIT n BY columns]
                    [LIMIT [n, ]m]
                    [UNION ALL ...]
                    [INTO OUTFILE filename]
                    [FORMAT format]"""
        if DB_TABLE.lower().startswith('select '):
            DB_TABLE = f"( {DB_TABLE} )"
        else:
            pass
        main_body = f"SELECT {SELECT_CLAUSE} FROM {DB_TABLE} {SAMPLE_CLAUSE}"
        join = f"{ARRAY_JOIN_CLAUSE} {JOIN_CLAUSE}"
        where_conds = f"{PREWHERE_CLAUSE} {WHERE_CLAUSE} {GROUP_BY_CLAUSE} {HAVING_CLAUSE} "
        order_limit = f"{ORDER_BY_CLAUSE} {LIMIT_N_CLAUSE} {LIMIT_CLAUSE}"
        sql = f"{main_body} {join} {where_conds} {order_limit}"
        return sql

    @classmethod
    def _group_by(cls, base_sql: str, by: list, agg_cols: list, where: list = None, having: list = None, order_by=None,
                  limit_by=None, limit=None):
        sql = cls.create_select_sql(DB_TABLE=base_sql, cols=by + agg_cols,
                                    sample=None,
                                    array_join=None, join=None,
                                    prewhere=None, where=where, having=having,
                                    group_by=by,
                                    order_by=order_by, limit_by=limit_by, limit=limit)

        return sql


if __name__ == '__main__':
    # test = "ClickHouse://{user}:{passwd}@{host}:0001/None"
    # ch = CH(test)
    # print(1)larity

    p = {'host': '***REMOVED***', 'port': 8123, 'user': 'default', 'password': '***REMOVED***', 'db': 'default'}

    sql = 'select * from default.user_test limit 10000 '
    r2 = ClickHouseDBPool(p)
    table_node = r2.default.user_test
    test2 = r2.default.test2
    columns = test2.columns
    # print(test2.head())
    # ['id', 'user_name', 'pass_word', 'today2', 'r1', 'r2', 'r3', 'r4']

    # sql = 'select *, today() as today2,rand64(1) as r1,randConstant(2) as r2,rand64(3) as r3,randConstant(4) as r4 from default.user_test'
    # t = f'create view if not exists default.test2 as {sql}'
    # df = table_node.query(t)
    # sql = 'select user_name, count(1) from default.test2 group by user_name limit 19'
    # sql3 = 'select * from default.test2 limit 19'
    # df = table_node.query(sql)
    # sql2 = SQLBuilder.create_select_sql('default.test2', ['user_name', 'count(1)'], group_by=['user_name'], limit=19)
    # df2 = table_node.query(sql)
    # table_node['default.user_test'] = table_node.query(sql3)
    # print(df)
    # print(df2)
    sql = Groupby._group_by('default.test2', by=['user_name'], agg_cols=['sum(r1)'], limit=10)
    print(sql)
    df = table_node.query(sql)

    print(1)

    # res = r2.query(sql)

    # c = r2.query(sql)
    # for i in range(120):
    #
    #     r2['default.user_test'] = c

    # print(c.read())
    # print(c)

    # {k: v * 1 if isinstance(v, bool) else v for k, v in updated_settings.items()}

    pass

# coding=utf-8


"""
Created on Sun Aug 18 12:17:40 2019

@author: lee1984
"""

import gzip
import json
import urllib
import warnings
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


class _ClickHouseBaseNode(_ClickHouseNodeBaseTool):
    def __init__(self, name, **db_settings):

        self._db_settings = db_settings

        self.db_name = self._db_settings['db']

        self._para = node(self._db_settings['host'], self._db_settings['port'], self._db_settings['user'],
                          self._db_settings['password'], self._db_settings['db'])

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
        self._test_connection()

        # components = urllib.parse.urlparse(url_str)

    def __getitem__(self, sql):
        return self.query(sql)

    def __setitem__(self, db_table, df):
        db, table = db_table.split('.')
        self._df_insert_(db, table, df)

    @property
    def _database_exist_status(self):
        return self.db_name in self.databases

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

    def close(self):
        self.session.close()
        self.is_closed = True

    def __exit__(self):
        self.close()

    def _describe(self, db, table):
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

    @property
    def session(self):
        # url_str = "http://{user}:{passwd}@{host}:{port}".format(host=self._para.host, port=int(self._para.port),
        #                                                         user=self._para.user, passwd=self._para.password)
        if not self._async:

            self._session = requests.sessions.Session()

            return self._session  # grequests.AsyncRequest
        else:
            return grequests

    @session.setter
    def session(self, value: bool):
        self._async = value

    def _test_connection(self):
        ret_value = self.session.get(self._base_url)

        print(ret_value.text)

    @property
    def _base_url(self):
        _base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port),
                                                    user=self._para.user, passwd=self._para.password)
        # params = urllib.parse.urlencode(self.http_settings)
        return _base_url  # , params

    def _async_request(self, sql_list, convert_to='dataframe', todf=True, transfer_sql_format=True, return_sql=True,
                       auto_switch=True):

        def exception_handler(request, exception):
            print("Request failed")

        if not self._async:
            raise ValueError("must manually switch to async mode")
        tasks = (self._request_unit_(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format) for sql in
                 sql_list)
        if return_sql:
            res_list = zip(sql_list,
                           self.session.map(tasks, exception_handler=exception_handler),
                           size=self.max_async_query_once)
            for sql, resp in res_list:
                if todf:
                    d = self._load_into_pd(resp.content, convert_to)
                    yield sql, d
                else:
                    yield resp.content
        else:
            for resp in self.session.map(tasks, exception_handler=exception_handler, size=self.max_async_query_once):
                if todf:
                    d = self._load_into_pd(resp.content, convert_to)

                    yield d
                else:
                    yield resp.content
        if auto_switch:
            self._async = False

    def _request_unit_(self, sql, convert_to='dataframe', transfer_sql_format=True):
        sql2 = self._transfer_sql_format(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)

        if self.http_settings['enable_http_compression'] == 1:
            url = self._base_url + urllib.parse.urlencode(self.http_settings)
            resp = self.session.post(url,
                                     data=gzip.compress(sql2.encode()),
                                     headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        else:
            settings = self.http_settings.copy()
            settings.update({'query': sql2})
            url = self._base_url + urllib.parse.urlencode(settings)
            resp = self.session.post(url)
        return resp

    # @timer
    @lru_cache(maxsize=10)
    def _request(self, sql, convert_to='dataframe', todf=True, transfer_sql_format=True):
        if self._async:
            raise ValueError('current async mode will disable sync mode!')
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

    # @timer
    def _async_query(self, sql_list):
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
        res = list(self._async_request(sql_list, convert_to='dataframe', todf=todf, return_sql=False))
        return res

    def _df_insert_(self, db, table, df):
        describe_table = self._describe(db, table)

        query_with_format = 'insert into {0} format JSONEachRow \n{1}'.format('{}.{}'.format(db, table), '\n'.join(
            [i for i in self._check_df_and_dump(df, describe_table)]))
        self._request(query_with_format, convert_to='dataframe', todf=False, transfer_sql_format=False)

    def execute(self, sql):
        if isinstance(sql, str):
            return self._execute(sql)
        elif isinstance(sql, (list, tuple)):
            max_queries = self.max_async_query_once * 2
            if len(sql) > max_queries:
                raise ValueError(f'too many queries,please reduce to less than {max_queries}!')
            self.session = True
            return self._async_query(sql)
        else:
            raise ValueError('sql must be str or list or tuple')

    def _execute(self, sql):

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

    def query(self, sql, optimize=False):
        if isinstance(sql, str) and sql.lower().startswith('insert into'):
            db_table = sql.lower().split('insert into')[-1].split(' ')[0]
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


class ClickHouseBaseNode(_ClickHouseBaseNode):
    def __init__(self, name, **settings):
        super(ClickHouseBaseNode, self).__init__(name, **settings)

    def _create_table(self, db, table, select_sql, keys_cols, table_engine='ReplacingMergeTree'):
        pass


class ClickHouseTableNode(ClickHouseBaseNode):
    def __init__(self, table_name, **settings):
        super(ClickHouseTableNode, self).__init__(table_name, **settings)
        self.table_name = table_name

    @property
    def _table_exist_status(self):
        return self.table_name in self.tables

    @property
    def table_structure(self):
        return self._describe(self.db_name, self.table_name)

    @property
    def columns(self):
        if self._table_exist_status:
            return self.table_structure['name'].values.tolist()
        else:
            raise ValueError(f'table: {self.table_name} may not be exists!')

    def head(self, top=10):
        sql = f'select * from {self.db_name}.{self.table_name} limit {top}'
        return self.execute(sql)


class CreateTableTools(object):
    @staticmethod
    def check_table_exists(db, table, obj):
        """

        :param db:
        :param table:
        :param obj:
        :return: exists =true not exists = False
        """
        sql = f"show tables from {db} like '{table}' "
        res = obj.qeury(sql)
        if res.empty:
            return False
        else:
            return True

    @staticmethod
    def translate_dtype(sdf):
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
    def create_table_dtype(cls, db, table, sdf, key_cols, engine_type='ReplacingMergeTree', extra_format_dict=None):
        if extra_format_dict is None:
            extra_format_dict = {}
        dtypes_df = cls.translate_dtype(sdf)
        cols_def = ','.join([f"{name} {dtype}" if name not in extra_format_dict.keys() else "{}.{}".format(name, str(
            extra_format_dict.get(name))) for name, dtype in dtypes_df.to_dict().items()])
        order_by_cols = ','.join(key_cols)

        base = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type} ORDER BY ({order_by_cols}) SETTINGS index_granularity = 8192 "
        return base

    @classmethod
    def create_table_sql(cls, obj, db, table, sql, key_cols, engine_type='ReplacingMergeTree', extra_format_dict=None):

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

        exist_status = cls.check_table_exists(db, table, obj)
        if exist_status:
            print('table:{table} already exists!')
        else:
            print('will create {table} at {db}')
            dtypes_df = query_func(describe_sql)
            cls.create_table_dtype(db, table, dtypes_df, key_cols, engine_type=engine_type,
                                   extra_format_dict=extra_format_dict)


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
    sql = 'select user_name, count(1) from default.test2 group by user_name limit 19'
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

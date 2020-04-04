# coding=utf-8


"""
Created on Sun Aug 18 12:17:40 2019

@author: lee1984
"""

import gzip
import json
import urllib
from collections import namedtuple

import grequests
import numpy as np
import pandas as pd
import requests

from Nodes.utils_node.lazy_load import LazyInit
from Nodes.utils_node.timer import timer


class ClickHouseNodeBaseTool(object):

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


class _ClickHouseBaseNode(ClickHouseNodeBaseTool):
    def __init__(self, name, **db_settings):
        self.name = name
        self._db_settings = db_settings
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

        self._test_connection()

        # components = urllib.parse.urlparse(url_str)

    def __exit__(self):
        self.session.close()

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
    def _para(self):
        node = namedtuple('clickhouse', ['host', 'port', 'user', 'password', 'db'])
        return node(self._db_settings['host'], self._db_settings['port'], self._db_settings['user'],
                    self._db_settings['password'], self._db_settings['db'])

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

    @timer
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

    @timer
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

    def query(self, sql):
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
            # res = self._request(sql, convert_to='dataframe', todf=True)
        elif sql.lower().startswith('insert') or sql.lower().startswith('optimize') or sql.lower().startswith('create'):
            todf = False
            # res = self._request(sql, convert_to='dataframe', todf=False)
        else:
            raise ValueError('Unknown sql! current only accept select, insert, show, optimize')
        res = self._request(sql, convert_to='dataframe', todf=todf)
        return res


class ClickHouseTableNode(_ClickHouseBaseNode):
    def __init__(self, table_name, settings):
        super(ClickHouseTableNode, self).__init__(table_name, **settings)
        self.table_name = table_name
        self.db_name = self._para.db

    @property
    def _exist_status(self):
        return self.table_name in self.tables

    @property
    def _table_structure(self):
        return self._describe(self.db_name, self.table_name)

    @property
    def tables(self):
        sql = 'SHOW TABLES FROM {db}'.format(db=self._para.db)
        res = self.query(sql).values.ravel().tolist()
        return res

    @property
    def databases(self):
        sql = 'SHOW DATABASES'
        res = self.query(sql).values.ravel().tolist()
        return res

    @property
    def columns(self):
        if self._exist_status:
            return self._table_structure['name'].values.tolist()
        else:
            raise ValueError(f'table: {self.table_name} may not be exists!')

    def __getitem__(self, item):
        return self._request(item, convert_to='dataframe', todf=True)

    def __setitem__(self, key, value):
        db, table = key.split('.')
        self._df_insert_(db, table, value)


class ClickHouseDBPool(LazyInit):  # lazy load to improve loading speed
    def __init__(self, db: str = 'default', settings: (str, dict, object, None) = None):
        super(ClickHouseDBPool, self).__init__()

        if settings is None:
            settings = ClickHouseSettings().get()
        if db is not None:
            settings['db'] = db
        else:
            db = settings['db']
        self.db = db
        settings, conn = ConnectionParser.checker_multi_and_create(db, settings, target_db_type='ClickHouse')
        self._settings = settings
        self._conn = conn
        for table in self.tables:
            if table != 'tables':
                try:
                    setattr(self, table, ClickHouseTableNode(table, self._conn))
                except Exception as e:
                    print(str(e))
                    pass

    @property
    def tables(self):
        tables = [table[0] for table in self._conn.SHOWTABLES()]
        return tables

    def __getitem__(self, table: str):
        return getattr(self, table)

    def __setitem__(self, table: str, table_obj):
        setattr(self, table, table_obj)


if __name__ == '__main__':
    # test = "ClickHouse://{user}:{passwd}@{host}:0001/None"
    # ch = CH(test)
    # print(1)

    p = {'host': '***REMOVED***', 'port': 8123, 'user': 'default', 'password': '***REMOVED***', 'db': 'default'}

    sql = 'select * from default.user_test limit 10000 '
    r2 = ClickHouseBaseNode('user_test', **p)

    ff = r2._describe('default', 'user_test')

    # res = r2.query(sql)

    # c = r2.query(sql)
    # for i in range(120):
    #
    #     r2['default.user_test'] = c

    # print(c.read())
    # print(c)

    # {k: v * 1 if isinstance(v, bool) else v for k, v in updated_settings.items()}

    pass

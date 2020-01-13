# coding=utf-8


"""
Created on Sun Aug 18 12:17:40 2019

@author: lee1984
"""

import gzip
import http
import json
import time
import urllib
from collections import namedtuple

import numpy as np
import pandas as pd

ch_conn_tuple = namedtuple('clickhouse_params', ['host', 'port', 'user', 'passwd', 'db'])


class CHBase(object):
    def __init__(self, name: str, user='default', passwd='123456', host='0.0.0.0', port=8123, db='default'):
        self.name = name

        self._para = ch_conn_tuple(host, port, user, passwd, db)

        # sample  ='http://user:password@clickhouse_host:8123'

        self.accepted_formats = ['DataFrame', 'TabSeparated', 'TabSeparatedRaw', 'TabSeparatedWithNames',
                                 'TabSeparatedWithNamesAndTypes', 'CSV', 'CSVWithNames', 'Values', 'Vertical', 'JSON',
                                 'JSONCompact', 'JSONEachRow', 'TSKV', 'Pretty', 'PrettyCompact',
                                 'PrettyCompactMonoBlock', 'PrettyNoEscapes', 'PrettySpace', 'XML']
        self.settings = self._merge_settings(None)
        http_get_params = {'user': self._para.user, 'password': self._para.passwd}

        http_get_params.update(self.settings)
        self.http_get_params = http_get_params

    def SHOWTABLES(self):
        res = self.get('SHOW TABLES').values
        return res

    def _create_conn(self):
        print()
        url_str = "http://{user}:{passwd}@{host}:{port}".format(host=self._para.host, port=int(self._para.port),
                                                                user=self._para.user, passwd=self._para.passwd)
        components = urllib.parse.urlparse(url_str)
        return http.client.HTTPConnection(components.hostname, port=components.port)

    @staticmethod
    def _check_sql_select_only(sql):
        if sql.strip(' \n\t').lower()[:6] not in ['select', 'descri', 'show t']:
            raise ValueError('"query" should start with "select" or "describe" or show, ' + \
                             'while the provided "query" starts with "{0}"'.format(sql.strip(' \n\t').split(' ')[0]))

    @staticmethod
    def _transfer_sql_format(sql, convert_to):
        clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
        query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
        return query_with_format

    @classmethod
    def _compression_switched_request(cls, query_with_format, conn, updated_settings, http_get_params):

        if updated_settings['enable_http_compression'] == 1:
            conn.request('POST', '/?' + urllib.parse.urlencode(http_get_params),
                         body=gzip.compress(query_with_format.encode()),
                         headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        else:
            conn.request('POST', '/?' + urllib.parse.urlencode(http_get_params), body=query_with_format.encode())
        return conn

    @staticmethod
    def _test_connection(conn):
        conn.request('GET', '/')
        ret_value = conn.getresponse().read().decode().replace('\n', '')
        print(ret_value)

    def _get_data(self, conn, updated_settings, auto_close=True):
        resp = conn.getresponse()

        if resp.status == 404:
            error_message = gzip.decompress(resp.read()).decode() if updated_settings['enable_http_compression'] == 1 \
                else resp.read().decode()
            if auto_close:
                conn.close()
            raise ValueError(error_message)
        elif resp.status == 401:
            if auto_close:
                conn.close()
            raise ConnectionRefusedError(resp.reason + '. The username or password is incorrect.')
        else:
            if resp.status != 200:
                error_message = gzip.decompress(resp.read()).decode() if updated_settings[
                                                                             'enable_http_compression'] == 1 \
                    else resp.read().decode()
                if auto_close:
                    conn.close()
                raise NotImplementedError('Unknown Error: status: {0}, reason: {1}, message: {2}'.format(
                    resp.status, resp.reason, error_message))

        total = bytes()
        bytes_downloaded = 0
        last_time = time.time()

        while not resp.isclosed():
            bytes_downloaded += 300 * 1024
            total += resp.read(300 * 1024)
            if time.time() - last_time > 1:
                last_time = time.time()
                print('\rDownloaded: %.1f MB.' % (bytes_downloaded / 1024 / 1024), end='\r')
        if auto_close:
            print('will close conn')
            conn.close()
            print('conn closed')
        ret_value = gzip.decompress(total).decode() if updated_settings[
                                                           'enable_http_compression'] == 1 else total.decode()
        return ret_value

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

    def get(self, sql, convert_to='DataFrame', auto_close=True):
        conn = self._create_conn()
        self._test_connection(conn)
        self._check_sql_select_only(sql)

        updated_settings = self.settings

        # if sql.strip(' \n\t').lower()[:6] not in ['select', 'descri']:
        #     raise ValueError('"query" should start with "select" or "describe", ' + \
        #                      'while the provided "query" starts with "{0}"'.format(sql.strip(' \n\t').split(' ')[0]))

        # clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
        # query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')

        # http_get_params = {'user': self._para.user, 'password': self._para.passwd}

        # http_get_params.update(self.settings)

        # if updated_settings['enable_http_compression'] == 1:
        #     conn.request('POST', '/?' + urllib.parse.urlencode(self.http_get_params),
        #                  body=gzip.compress(query_with_format.encode()),
        #                  headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        # else:
        #     conn.request('POST', '/?' + urllib.parse.urlencode(self.http_get_params), body=query_with_format.encode())
        query_with_format = self._transfer_sql_format(sql, convert_to)
        conn = self._compression_switched_request(query_with_format, conn, updated_settings, self.http_get_params)

        # resp = conn.getresponse()
        #
        # if resp.status == 404:
        #     error_message = gzip.decompress(resp.read()).decode() if updated_settings['enable_http_compression'] == 1 \
        #         else resp.read().decode()
        #     conn.close()
        #     raise ValueError(error_message)
        # elif resp.status == 401:
        #     conn.close()
        #     raise ConnectionRefusedError(resp.reason + '. The username or password is incorrect.')
        # else:
        #     if resp.status != 200:
        #         error_message = gzip.decompress(resp.read()).decode() if updated_settings[
        #                                                                      'enable_http_compression'] == 1 \
        #             else resp.read().decode()
        #         conn.close()
        #         raise NotImplementedError('Unknown Error: status: {0}, reason: {1}, message: {2}'.format(
        #             resp.status, resp.reason, error_message))
        #
        # total = bytes()
        # bytes_downloaded = 0
        # last_time = time.time()
        #
        # while not resp.isclosed():
        #     bytes_downloaded += 300 * 1024
        #     total += resp.read(300 * 1024)
        #     if time.time() - last_time > 1:
        #         last_time = time.time()
        #         print('\rDownloaded: %.1f MB.' % (bytes_downloaded / 1024 / 1024), end='\r')
        # print()
        # conn.close()
        #
        # ret_value = gzip.decompress(total).decode() if updated_settings[
        #                                                    'enable_http_compression'] == 1 else total.decode()
        ret_value = self._get_data(conn, updated_settings, auto_close=auto_close)

        # if convert_to.lower() == 'dataframe':
        #     result_dict = json.loads(ret_value, strict=False)
        #     dataframe = pd.DataFrame.from_records(result_dict['data'], columns=[i['name'] for i in result_dict['meta']])
        #
        #     for i in result_dict['meta']:
        #         if i['type'] in ['DateTime', 'Nullable(DateTime)']:
        #             dataframe[i['name']] = pd.to_datetime(dataframe[i['name']])
        #
        #     ret_value = dataframe
        ret_value = self._load_into_pd(ret_value, convert_to)

        return ret_value

    @staticmethod
    def _merge_settings(settings):
        updated_settings = {
            'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
            'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
            'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
            'wait_end_of_query': 0}

        if settings is not None:
            invalid_setting_keys = list(set(settings.keys()) - set(updated_settings.keys()))
            if len(invalid_setting_keys) > 0:
                raise ValueError('setting "{0}" is invalid, valid settings are: {1}'.format(
                    invalid_setting_keys[0], ', '.join(updated_settings.keys())))

            updated_settings.update(settings)

        for i in updated_settings:
            updated_settings[i] = 1 if updated_settings[i] == True else 0 if updated_settings[i] == False else \
                updated_settings[i]

        return updated_settings

    def get_describe_table(self, db, table):
        describe_sql = 'describe table {}.{}'.format(db, table)
        describe_table = self.get(describe_sql, auto_close=True)
        # non_nullable_columns = list(describe_table[~describe_table['type'].str.startswith('Nullable')]['name'])
        # integer_columns = list(describe_table[describe_table['type'].str.contains('Int', regex=False)]['name'])
        # missing_in_df = {i: np.where(df[i].isnull(), 1, 0).sum() for i in non_nullable_columns}
        #
        # df_columns = list(df.columns)
        # each_row = df.to_dict(orient='records')
        # del df
        return describe_table  # , integer_columns, non_nullable_columns

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
                        except:
                            raise ValueError('Column "{0}" is {1}, while value "{2}"'.format(col,
                                                                                             describe_table[
                                                                                                 describe_table[
                                                                                                     'name'] == col].iloc[
                                                                                                 0]['type'], row[col]) + \
                                             ' in the dataframe column cannot be converted to Integer.')
            yield json.dumps(row, ensure_ascii=False)
        # return df_columns, each_row

    # @classmethod
    # def check_and_dump(cls, df, describe_table, db, table):
    #     # db_table = '{}.{}'.format(db, table)
    #     # df_columns = list(df.columns)
    #
    #     # df_columns, each_row = cls._check_df(df, describe_table)
    #     # json_each_row = '\n'.join([json.dumps(i, ensure_ascii=False) for i in cls._check_df(df, describe_table)])
    #     # json_each_row = '\n'.join([i for i in cls._check_df(df, describe_table)])
    #     # del each_row
    #
    #     query_with_format = 'insert into {0} format JSONEachRow \n{1}'.format('{}.{}'.format(db, table), '\n'.join(
    #         [i for i in cls._check_df_and_dump(df, describe_table)]))
    #     # del json_each_row
    #     return query_with_format

    def insert(self, df: pd.DataFrame, db: str, table: str):

        describe_table = self.get_describe_table(db, table)
        # df_columns, each_row = self._check_df(df, describe_table)
        query_with_format = 'insert into {0} format JSONEachRow \n{1}'.format('{}.{}'.format(db, table), '\n'.join(
            [i for i in self._check_df_and_dump(df, describe_table)]))
        # json_each_row = '\n'.join([json.dumps(i, ensure_ascii=False) for i in each_row])
        # del each_row
        #
        # query_with_format = 'insert into {0} format JSONEachRow \n{1}'.format(db_table, json_each_row)
        # del json_each_row

        conn = self._create_conn()
        self._test_connection(conn)
        # self._check_sql_select_only(sql)

        updated_settings = self.settings

        # http_get_params = {'user': components.username, 'password': components.password}
        # http_get_params.update(updated_settings)
        # conn = http.client.HTTPConnection(components.hostname, port=components.port)

        conn = self._compression_switched_request(query_with_format, conn, updated_settings, self.http_get_params)

        # if updated_settings['enable_http_compression'] == 1:
        #     conn.request('POST', '/?' + urllib.parse.urlencode(http_get_params),
        #                  body=gzip.compress(query_with_format.encode()),
        #                  headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        # else:
        #     conn.request('POST', '/?' + urllib.parse.urlencode(http_get_params), body=query_with_format.encode())
        resp = conn.getresponse()

        if resp.status != 200:
            error_message = gzip.decompress(resp.read()).decode() if updated_settings['enable_http_compression'] == 1 \
                else resp.read().decode()
            conn.close()
            raise NotImplementedError('Unknown Error: status: {0}, reason: {1}, message: {2}'.format(
                resp.status, resp.reason, error_message))

        conn.close()
        print('Done.')


class ClickHouseNode(CHBase):
    pass


# class ClickHouseTableBaseNode(BasicNode):
#     def __init__(self, table: str, conn):
#         super(ClickHouseTableBaseNode, self).__init__(table)
#         pass
#
#     def query(self, sql):
#         pass
ClickHouseNodeName = 'ClickHouseNode'

if __name__ == '__main__':
    # test = "ClickHouse://{user}:{passwd}@{host}:0001/None"
    # ch = CH(test)
    # print(1)
    pass

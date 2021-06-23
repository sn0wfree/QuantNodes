# coding=utf-8
# import records
import sqlite3


class SQLiteBackend(object):



    @staticmethod
    def _df_read_(database, sql, **kwargs):
        with sqlite3.connect(database) as conn:
            return pd.read_sql(sql, conn, **kwargs)

    @staticmethod
    def _df_write_(database,table, df, if_exists='append',**kwargs):
        with sqlite3.connect(database) as conn:
            df.to_sql(table,conn,if_exists='append',**kwargs)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # with sqlite3.connect('tst.sqlite') as conn:
    df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
    with sqlite3.connect('tst.sqlite') as conn:
        test = pd.read_sql('select * from test', conn)

        print(test)
    print(1)

    pass

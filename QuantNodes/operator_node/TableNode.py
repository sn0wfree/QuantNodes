# coding=utf-8
import copy

# from nodes.clickhouse_node import ClickHouseDBNode
from QuantNodes.operator_node.SQLUtils import SQLBuilder


class OperatorBaseNode(SQLBuilder):
    def __init__(self, db_table: str, cols: tuple, mode='limit'):

        """

       :param having: str ["r1 >1 and r2 <2"]
       :param DB_TABLE: str default.test
       :param cols: list [ r1,r2,r3 ]
       :param sample: str 0.1 or 1000
       :param array_join: list ['arrayA as a','arrayB as b']
       :param join: dict {'type':'all left join','USING' : "r1,r2"}
       :param prewhere: str ["r1 >1" , "r2 <2"]
       :param where: str ["r1 >1.5" , "r2 <1.3"]
       :param group_by: list ['r1','r2']
       :param order_by: list ['r1 desc','r2 desc']
       :param limit_by: dict {'N':10,'limit_by_cols':['r1','r2']}
       :param limit: int 100
       :return:  str
        """

        self._is_sql = False
        self._temp_sql = None
        self._db_table = db_table
        db, table = db_table.split('.')

        self.mode = mode
        self._table = table
        # self._db_node = ClickHouseDBNode(settings)
        self.db = db
        # if table_name in self._db_node.tables:
        #     self._table_node = getattr(self._db_node, table_name)
        # else:
        #     raise ValueError(f'{table_name} not in database:{self.db}')

        self._columns = cols
        self._sample = None
        self._join = None
        self._array_join = None
        self._where = None
        self._pre_where = None
        self._group_by = None
        self._order_by = None
        self._limit_by = None
        self._limit = None
        self._having = None




    def groupby(self, by: list, agg_cols: list, where: list = None, having: list = None, order_by=None,
                limit_by=None, limit=None):
        a = copy.deepcopy(self)
        a._columns_ = by + agg_cols
        a._where_ = where
        a._having_ = having
        a._group_by_ = by
        a._order_by_ = order_by
        a._limit_by_ = limit_by
        a._limit_ = limit

        # sql = a._sql_

        return a

    # def __deepcopy__(self, obj):
    #     other = BaseNode(self._table_name, self._db_node.settings, mode=self.mode)
    #
    #     other._is_sql = True
    #     other._temp_sql = self._sql_
    #
    #     return other

    @property
    def _base_(self):
        if self._is_sql and self._temp_sql is not None:
            return "( " + self._temp_sql + " )"
        elif self._is_sql and self._temp_sql is None:
            raise ValueError('sql format has been translated, but got None sql')
        else:
            return self.db_table

    @property
    def _sql_(self):
        db_table = self._db_table
        cols = self._columns_
        sample = self._sample_
        array_join = self._array_join_
        join = self._join_
        pre_where = self._pre_where_
        where = self._where_
        having = self._having_
        group_by = self._group_by_
        order_by = self._order_by_
        limit_by = self._limit_by_
        limit = self._limit_
        sql = self.create_select_sql(db_table, cols,
                                     sample=sample,
                                     array_join=array_join, join=join,
                                     prewhere=pre_where, where=where, having=having,
                                     group_by=group_by,
                                     order_by=order_by, limit_by=limit_by,
                                     limit=limit)
        return sql.strip()

    def __query__(self):
        sql = self._sql_

        return self.query(sql + ' limit 10')

    @property
    def db_table(self):
        return "{}.{}".format(self.db, self.table)

    @property
    def _columns_(self):
        return self._columns

    @_columns_.setter
    def _columns_(self, cols):
        self._columns = cols

    @property
    def _sample_(self):
        return self._sample

    @_sample_.setter
    def _sample_(self, samples):
        self._sample = samples

    @property
    def _join_(self):
        return self._join

    @_join_.setter
    def _join_(self, joins):
        self._join = joins

    @property
    def _array_join_(self):
        return self._array_join

    @_array_join_.setter
    def _array_join_(self, joins):
        self._array_join = joins

    @property
    def _where_(self):
        return self._where

    @_where_.setter
    def _where_(self, wheres):
        self._where = wheres

    @property
    def _pre_where_(self):
        return self._pre_where

    @_pre_where_.setter
    def _pre_where_(self, pre_wheres):
        self._pre_where = pre_wheres

    @property
    def _group_by_(self):
        return self._group_by

    @_group_by_.setter
    def _group_by_(self, _group_bys):
        self._group_by = _group_bys

    @property
    def _order_by_(self):
        return self._order_by

    @_order_by_.setter
    def _order_by_(self, _order_bys):
        self._order_by = _order_bys

    @property
    def _limit_by_(self):
        return self._limit_by

    @_limit_by_.setter
    def _limit_by_(self, _limit_bys):
        self._limit_by = _limit_bys

    @property
    def _limit_(self):
        return self._limit

    @_limit_.setter
    def _limit_(self, limits):
        self._limit = limits

    @property
    def _having_(self):
        return self._limit

    @_having_.setter
    def _having_(self, limits):
        self._having = limits

    @property
    def table(self):
        return self._table_name

    @table.setter
    def table(self, table_name):
        self._table = table_name

    @property
    def query(self):
        return self._table_node.query


class Node2(OperatorBaseNode):
    def __init__(self, table_name, settings, mode='limit'):
        super(Node2, self).__init__(table_name, settings, mode=mode)
        self._sql_str = f'select * from {self.db_table}'
        pass

    @property
    def _sql(self):
        return self._sql_str  # f'select * from {self.db_table}'

    @_sql.setter
    def _sql(self, sql):
        self._sql_str = sql

    def __getitem__(self, item: str):  ## get cols
        if isinstance(item, (str, list)):
            if isinstance(item, str):
                cols_str = item
            else:
                cols_str = ','.join(item)
            db_table = self.db_table
            if self.mode == 'limit':
                limit_clause = 'limit 1000'
            else:
                limit_clause = ''
            sql = f'select {cols_str} from {db_table} {limit_clause}'
        else:
            raise ValueError('items must be str')
        self._sql = sql
        return self  # self.query(sql)

    def compute(self):
        return self.query(self._sql)


if __name__ == '__main__':
    conn_settings = "ClickHouse://default:***REMOVED***@***REMOVED***:8123/default"
    n = Node2('test2', conn_settings)
    d = n.groupby(by=['r2', 'r3'], agg_cols=['count(id)'])
    print(d.query('show tables'))
    pass

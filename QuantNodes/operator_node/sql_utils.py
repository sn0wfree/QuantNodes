# coding=utf-8
"""
SQL utilities

SQL building utilities for generating SQL queries.
"""


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
        ORDER_BY_CLAUSE = f"ORDER BY ( {order_by_cols_str} )"

        SAMPLE_CLAUSE = cls._assemble_cols_2_clause('SAMPLE BY', sample_by_cols, default='')

        PRIMARY_BY_CLAUSE = cls._assemble_cols_2_clause('PRIMARY BY', primary_by_cols, default='')

        PARTITION_by_CLAUSE = cls._assemble_cols_2_clause('PARTITION BY', partition_by_cols, default='')

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
        maid_body = f"CREATE TABLE IF NOT EXISTS {DB_TABLE} {ON_CLUSTER} ( {cols_def} ) ENGINE = {ENGINE_TYPE}"

        settings = "SETTINGS index_granularity = 8192"
        conds = f"{PARTITION_by_CLAUSE} {ORDER_BY_CLAUSE} {PRIMARY_BY_CLAUSE} {SAMPLE_CLAUSE} {TTL}"

        base = f"{maid_body} {conds}  {settings}"
        return base


class SQLBuilder:
    """SQL building utility class"""

    @staticmethod
    def _assemble_sample(sample):
        if sample is None:
            return ""
        elif isinstance(sample, (int, float)):
            if sample < 1:
                return f"SAMPLE {sample}"
            else:
                return f"SAMPLE {sample}"
        return ""

    @staticmethod
    def _assemble_array_join(array_join_list):
        if array_join_list is None:
            return ""
        clauses = []
        for arr in array_join_list:
            clauses.append(f"ARRAY JOIN {arr}")
        return " ".join(clauses)

    @staticmethod
    def _assemble_join(join_info):
        if join_info is None:
            return ""
        join_type = join_info.get('type', '')
        using = join_info.get('USING', '')
        return f"{join_type} USING ( {using} )"

    @staticmethod
    def _assemble_where_like(where_list, prefix='WHERE'):
        if where_list is None:
            return ""
        clauses = []
        for where in where_list:
            clauses.append(where)
        where_clause = ' AND '.join(clauses)
        return f"{prefix} {where_clause}"

    @staticmethod
    def _assemble_group_by(group_by_list):
        if group_by_list is None:
            return ""
        cols_str = ','.join(group_by_list)
        return f"GROUP BY ( {cols_str} )"

    @staticmethod
    def _assemble_order_by(order_by_list):
        if order_by_list is None:
            return ""
        cols_str = ','.join(order_by_list)
        return f"ORDER BY ( {cols_str} )"

    @staticmethod
    def _assemble_limit_by(limit_by):
        if limit_by is None:
            return ""
        n = limit_by.get('N', 10)
        cols = limit_by.get('limit_by_cols', [])
        cols_str = ','.join(cols)
        return f"LIMIT {n} BY {cols_str}"

    @staticmethod
    def _assemble_limit(limit):
        if limit is None:
            return ""
        return f"LIMIT {limit}"

    @classmethod
    def raw_create_select_sql(cls, SELECT_CLAUSE, DB_TABLE, SAMPLE_CLAUSE='', ARRAY_JOIN_CLAUSE='',
                           JOIN_CLAUSE='', PREWHERE_CLAUSE='', WHERE_CLAUSE='',
                           GROUP_BY_CLAUSE='', HAVING_CLAUSE='', ORDER_BY_CLAUSE='',
                           LIMIT_N_CLAUSE='', LIMIT_CLAUSE=''):
        if DB_TABLE.lower().startswith('select '):
            DB_TABLE = f"( {DB_TABLE} )"
        main_body = f"SELECT {SELECT_CLAUSE} FROM {DB_TABLE} {SAMPLE_CLAUSE}"
        join = f"{ARRAY_JOIN_CLAUSE} {JOIN_CLAUSE}"
        where_conditions = f"{PREWHERE_CLAUSE} {WHERE_CLAUSE} {GROUP_BY_CLAUSE} {HAVING_CLAUSE} "
        order_limit = f"{ORDER_BY_CLAUSE} {LIMIT_N_CLAUSE} {LIMIT_CLAUSE}"
        sql = f"{main_body} {join} {where_conditions} {order_limit}"
        return sql

    @classmethod
    def create_select_sql(cls, DB_TABLE: str, cols: list,
                          sample: (int, float, None) = None,
                          array_join: (list, None) = None, join: (dict, None) = None,
                          prewhere: (list, None) = None, where: (list, None) = None, having: (list, None) = None,
                          group_by: (list, None) = None,
                          order_by: (list, None) = None, limit_by: (dict, None) = None,
                          limit: (int, None) = None) -> str:
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


__all__ = ['SQLBuilder', 'TableEngineCreator']
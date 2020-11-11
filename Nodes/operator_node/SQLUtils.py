# coding=utf-8

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
            a_list_str = ' and '.join(a_list)
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

    # @classmethod
    # def group_by(cls, base_sql: str, by: list, agg_cols: list, where: list = None, having: list = None, order_by=None,
    #              limit_by=None, limit=None):
    #     sql = cls.create_select_sql(DB_TABLE=base_sql, cols=by + agg_cols,
    #                                 sample=None, array_join=None, join=None,
    #                                 prewhere=None, where=where, having=having,
    #                                 group_by=by, order_by=order_by, limit_by=limit_by, limit=limit)
    #
    #     return sql


if __name__ == '__main__':
    pass

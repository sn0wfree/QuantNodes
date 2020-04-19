# coding=utf-8


def query(obj):
    if hasattr(obj, '__query__'):
        return obj.__query__()
    else:
        raise ValueError(f'{obj} do not have query usage')


def groupby(base_sql, by: list, agg_cols: list, where: list = None, having: list = None, order_by=None,
             limit_by=None, limit=None):
    sql = base_sql.create_select_sql(DB_TABLE=base_sql, cols=by + agg_cols,
                                     sample=None, array_join=None, join=None,
                                     prewhere=None, where=where, having=having,
                                     group_by=by, order_by=order_by, limit_by=limit_by, limit=limit)

    return sql


if __name__ == '__main__':
    pass

# coding=utf-8
"""
符号计算引擎 - 数据库方言抽象

支持 ClickHouse、DuckDB、MySQL 等数据库的 SQL 方言差异。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Tuple


class DialectType(Enum):
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class SQLDialect(ABC):
    dialect_name: DialectType

    @abstractmethod
    def quote_identifier(self, name: str) -> str:
        pass

    @abstractmethod
    def quote_literal(self, value: Any) -> str:
        pass

    @abstractmethod
    def func_now(self) -> str:
        pass

    @abstractmethod
    def func_date_diff(self, unit: str, start: str, end: str) -> str:
        pass

    @abstractmethod
    def func_coalesce(self, *args: str) -> str:
        pass

    @abstractmethod
    def func_ifnull(self, expr: str, default: str) -> str:
        pass

    @abstractmethod
    def func_if(self, condition: str, then: str, else_: str) -> str:
        pass

    @abstractmethod
    def func_case(self, when_clauses: List[Tuple[str, str]], else_: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def func_cast(self, expr: str, target_type: str) -> str:
        pass

    @abstractmethod
    def func_interval(self, value: str, unit: str) -> str:
        pass

    @abstractmethod
    def is_distinct_from(self, a: str, b: str) -> str:
        pass

    @abstractmethod
    def is_not_distinct_from(self, a: str, b: str) -> str:
        pass

    @abstractmethod
    def func_rank(self) -> str:
        pass

    @abstractmethod
    def func_dense_rank(self) -> str:
        pass

    @abstractmethod
    def func_row_number(self) -> str:
        pass

    @abstractmethod
    def func_lag(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def func_lead(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def func_avg(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_sum(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_max(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_min(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_count(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_stddev(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_variance(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_quantile(self, expr: str, quantile: float) -> str:
        pass

    @abstractmethod
    def func_abs(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_ceil(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_floor(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_round(self, expr: str, decimals: int = 0) -> str:
        pass

    @abstractmethod
    def func_pow(self, expr: str, power: float) -> str:
        pass

    @abstractmethod
    def func_sqrt(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_ln(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_log10(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_cumsum(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_running_diff(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_length(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_concat(self, *args: str) -> str:
        pass

    @abstractmethod
    def func_upper(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_lower(self, expr: str) -> str:
        pass

    @abstractmethod
    def func_substring(self, expr: str, start: int, length: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def parse_datetime(self, expr: str, format: str) -> str:
        pass

    @abstractmethod
    def extract_date_part(self, expr: str, part: str) -> str:
        pass


class ClickHouseDialect(SQLDialect):
    dialect_name = DialectType.CLICKHOUSE

    def quote_identifier(self, name):
        return '`' + name + '`'

    def quote_literal(self, value):
        if value is None:
            return 'NULL'
        if isinstance(value, str):
            return "'" + value.replace("'", "\'") + "'"
        if isinstance(value, bool):
            return '1' if value else '0'
        return str(value)

    def func_now(self):
        return 'now()'

    def func_date_diff(self, unit, start, end):
        return "dateDiff('" + unit + "', " + start + ", " + end + ")"

    def func_coalesce(self, *args):
        return 'coalesce(' + ', '.join(args) + ')'

    def func_ifnull(self, expr, default):
        return 'ifNull(' + expr + ', ' + default + ')'

    def func_if(self, condition, then, else_):
        return 'if(' + condition + ', ' + then + ', ' + else_ + ')'

    def func_case(self, when_clauses, else_=None):
        parts = ['CASE WHEN ' + cond + ' THEN ' + val for cond, val in when_clauses]
        if else_:
            parts.append('ELSE ' + else_)
        parts.append('END')
        return ' '.join(parts)

    def func_cast(self, expr, target_type):
        return 'CAST(' + expr + ' AS ' + target_type + ')'

    def func_interval(self, value, unit):
        return 'INTERVAL ' + value + ' ' + unit

    def is_distinct_from(self, a, b):
        return a + ' IS DISTINCT FROM ' + b

    def is_not_distinct_from(self, a, b):
        return a + ' IS NOT DISTINCT FROM ' + b

    def func_rank(self):
        return 'rank()'

    def func_dense_rank(self):
        return 'dense_rank()'

    def func_row_number(self):
        return 'row_number()'

    def func_lag(self, expr, offset, default=None):
        if default is not None:
            return 'lagInFrame(' + expr + ', ' + str(offset) + ', ' + default + ')'
        return 'lagInFrame(' + expr + ', ' + str(offset) + ')'

    def func_lead(self, expr, offset, default=None):
        if default is not None:
            return 'leadInFrame(' + expr + ', ' + str(offset) + ', ' + default + ')'
        return 'leadInFrame(' + expr + ', ' + str(offset) + ')'

    def func_avg(self, expr):
        return 'avg(' + expr + ')'

    def func_sum(self, expr):
        return 'sum(' + expr + ')'

    def func_max(self, expr):
        return 'max(' + expr + ')'

    def func_min(self, expr):
        return 'min(' + expr + ')'

    def func_count(self, expr):
        return 'count(' + expr + ')'

    def func_stddev(self, expr):
        return 'stddevPop(' + expr + ')'

    def func_variance(self, expr):
        return 'varPop(' + expr + ')'

    def func_quantile(self, expr, quantile):
        return 'quantile(' + str(quantile) + ')(' + expr + ')'

    def func_abs(self, expr):
        return 'abs(' + expr + ')'

    def func_ceil(self, expr):
        return 'ceil(' + expr + ')'

    def func_floor(self, expr):
        return 'floor(' + expr + ')'

    def func_round(self, expr, decimals=0):
        return 'round(' + expr + ', ' + str(decimals) + ')'

    def func_pow(self, expr, power):
        return 'pow(' + expr + ', ' + str(power) + ')'

    def func_sqrt(self, expr):
        return 'sqrt(' + expr + ')'

    def func_ln(self, expr):
        return 'log(' + expr + ')'

    def func_log10(self, expr):
        return 'log10(' + expr + ')'

    def func_cumsum(self, expr):
        return 'sum(' + expr + ')'

    def func_running_diff(self, expr):
        return 'runningDifference(' + expr + ')'

    def func_length(self, expr):
        return 'length(' + expr + ')'

    def func_concat(self, *args):
        return 'concat(' + ', '.join(args) + ')'

    def func_upper(self, expr):
        return 'upper(' + expr + ')'

    def func_lower(self, expr):
        return 'lower(' + expr + ')'

    def func_substring(self, expr, start, length=None):
        if length is not None:
            return 'substring(' + expr + ', ' + str(start) + ', ' + str(length) + ')'
        return 'substring(' + expr + ', ' + str(start) + ')'

    def parse_datetime(self, expr, format):
        return 'parseDateTime(' + expr + ', '' + format + '')'

    def extract_date_part(self, expr, part):
        return 'extract(' + part + ' FROM ' + expr + ')'


class DuckDBDialect(SQLDialect):
    dialect_name = DialectType.DUCKDB

    def quote_identifier(self, name):
        return '"' + name + '"'

    def quote_literal(self, value):
        if value is None:
            return 'NULL'
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bool):
            return 'TRUE' if value else 'FALSE'
        return str(value)

    def func_now(self):
        return 'now()'

    def func_date_diff(self, unit, start, end):
        return "datediff('" + unit + "', " + start + ", " + end + ")"

    def func_coalesce(self, *args):
        return 'coalesce(' + ', '.join(args) + ')'

    def func_ifnull(self, expr, default):
        return 'ifnull(' + expr + ', ' + default + ')'

    def func_if(self, condition, then, else_):
        return 'if(' + condition + ', ' + then + ', ' + else_ + ')'

    def func_case(self, when_clauses, else_=None):
        parts = ['CASE WHEN ' + cond + ' THEN ' + val for cond, val in when_clauses]
        if else_:
            parts.append('ELSE ' + else_)
        parts.append('END')
        return ' '.join(parts)

    def func_cast(self, expr, target_type):
        return 'CAST(' + expr + ' AS ' + target_type + ')'

    def func_interval(self, value, unit):
        return "INTERVAL '" + value + "' " + unit

    def is_distinct_from(self, a, b):
        return a + ' IS DISTINCT FROM ' + b

    def is_not_distinct_from(self, a, b):
        return a + ' IS NOT DISTINCT FROM ' + b

    def func_rank(self):
        return 'rank()'

    def func_dense_rank(self):
        return 'dense_rank()'

    def func_row_number(self):
        return 'row_number()'

    def func_lag(self, expr, offset, default=None):
        if default is not None:
            return 'lag(' + expr + ', ' + str(offset) + ', ' + default + ')'
        return 'lag(' + expr + ', ' + str(offset) + ')'

    def func_lead(self, expr, offset, default=None):
        if default is not None:
            return 'lead(' + expr + ', ' + str(offset) + ', ' + default + ')'
        return 'lead(' + expr + ', ' + str(offset) + ')'

    def func_avg(self, expr):
        return 'avg(' + expr + ')'

    def func_sum(self, expr):
        return 'sum(' + expr + ')'

    def func_max(self, expr):
        return 'max(' + expr + ')'

    def func_min(self, expr):
        return 'min(' + expr + ')'

    def func_count(self, expr):
        return 'count(' + expr + ')'

    def func_stddev(self, expr):
        return 'stddev_pop(' + expr + ')'

    def func_variance(self, expr):
        return 'var_pop(' + expr + ')'

    def func_quantile(self, expr, quantile):
        return 'quantile_cont(' + expr + ', ' + str(quantile) + ')'

    def func_abs(self, expr):
        return 'abs(' + expr + ')'

    def func_ceil(self, expr):
        return 'ceil(' + expr + ')'

    def func_floor(self, expr):
        return 'floor(' + expr + ')'

    def func_round(self, expr, decimals=0):
        return 'round(' + expr + ', ' + str(decimals) + ')'

    def func_pow(self, expr, power):
        return 'power(' + expr + ', ' + str(power) + ')'

    def func_sqrt(self, expr):
        return 'sqrt(' + expr + ')'

    def func_ln(self, expr):
        return 'ln(' + expr + ')'

    def func_log10(self, expr):
        return 'log10(' + expr + ')'

    def func_cumsum(self, expr):
        return 'sum(' + expr + ')'

    def func_running_diff(self, expr):
        return expr + ' - lag(' + expr + ', 1)'

    def func_length(self, expr):
        return 'length(' + expr + ')'

    def func_concat(self, *args):
        return 'concat(' + ', '.join(args) + ')'

    def func_upper(self, expr):
        return 'upper(' + expr + ')'

    def func_lower(self, expr):
        return 'lower(' + expr + ')'

    def func_substring(self, expr, start, length=None):
        if length is not None:
            return 'substring(' + expr + ', ' + str(start) + ', ' + str(length) + ')'
        return 'substring(' + expr + ', ' + str(start) + ')'

    def parse_datetime(self, expr, format):
        return 'strptime(' + expr + ', '' + format + '')'

    def extract_date_part(self, expr, part):
        return "date_part('" + part + "', " + expr + ')'


class MySQLDialect(SQLDialect):
    dialect_name = DialectType.MYSQL

    def quote_identifier(self, name):
        return '`' + name + '`'

    def quote_literal(self, value):
        if value is None:
            return 'NULL'
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bool):
            return '1' if value else '0'
        return str(value)

    def func_now(self):
        return 'NOW()'

    def func_date_diff(self, unit, start, end):
        return 'DATEDIFF(' + end + ', ' + start + ')'

    def func_coalesce(self, *args):
        return 'COALESCE(' + ', '.join(args) + ')'

    def func_ifnull(self, expr, default):
        return 'IFNULL(' + expr + ', ' + default + ')'

    def func_if(self, condition, then, else_):
        return 'IF(' + condition + ', ' + then + ', ' + else_ + ')'

    def func_case(self, when_clauses, else_=None):
        parts = ['CASE WHEN ' + cond + ' THEN ' + val for cond, val in when_clauses]
        if else_:
            parts.append('ELSE ' + else_)
        parts.append('END')
        return ' '.join(parts)

    def func_cast(self, expr, target_type):
        return 'CAST(' + expr + ' AS ' + target_type + ')'

    def func_interval(self, value, unit):
        return 'INTERVAL ' + value + ' ' + unit

    def is_distinct_from(self, a, b):
        return '(' + a + ' IS NULL AND ' + b + ' IS NOT NULL OR ' + a + ' IS NOT NULL AND ' + b + ' IS NULL OR ' + a + ' <> ' + b + ')'

    def is_not_distinct_from(self, a, b):
        return '(' + a + ' IS NULL AND ' + b + ' IS NULL OR ' + a + ' = ' + b + ')'

    def func_rank(self):
        return 'RANK()'

    def func_dense_rank(self):
        return 'DENSE_RANK()'

    def func_row_number(self):
        return 'ROW_NUMBER()'

    def func_lag(self, expr, offset, default=None):
        if default is not None:
            return 'LAG(' + expr + ', ' + str(offset) + ', ' + default + ')'
        return 'LAG(' + expr + ', ' + str(offset) + ')'

    def func_lead(self, expr, offset, default=None):
        if default is not None:
            return 'LEAD(' + expr + ', ' + str(offset) + ', ' + default + ')'
        return 'LEAD(' + expr + ', ' + str(offset) + ')'

    def func_avg(self, expr):
        return 'AVG(' + expr + ')'

    def func_sum(self, expr):
        return 'SUM(' + expr + ')'

    def func_max(self, expr):
        return 'MAX(' + expr + ')'

    def func_min(self, expr):
        return 'MIN(' + expr + ')'

    def func_count(self, expr):
        return 'COUNT(' + expr + ')'

    def func_stddev(self, expr):
        return 'STDDEV_POP(' + expr + ')'

    def func_variance(self, expr):
        return 'VAR_POP(' + expr + ')'

    def func_quantile(self, expr, quantile):
        return 'QUANTILE_CONT(' + expr + ', ' + str(quantile) + ')'

    def func_abs(self, expr):
        return 'ABS(' + expr + ')'

    def func_ceil(self, expr):
        return 'CEILING(' + expr + ')'

    def func_floor(self, expr):
        return 'FLOOR(' + expr + ')'

    def func_round(self, expr, decimals=0):
        return 'ROUND(' + expr + ', ' + str(decimals) + ')'

    def func_pow(self, expr, power):
        return 'POWER(' + expr + ', ' + str(power) + ')'

    def func_sqrt(self, expr):
        return 'SQRT(' + expr + ')'

    def func_ln(self, expr):
        return 'LN(' + expr + ')'

    def func_log10(self, expr):
        return 'LOG10(' + expr + ')'

    def func_cumsum(self, expr):
        return 'SUM(' + expr + ')'

    def func_running_diff(self, expr):
        return expr + ' - LAG(' + expr + ', 1)'

    def func_length(self, expr):
        return 'LENGTH(' + expr + ')'

    def func_concat(self, *args):
        return 'CONCAT(' + ', '.join(args) + ')'

    def func_upper(self, expr):
        return 'UPPER(' + expr + ')'

    def func_lower(self, expr):
        return 'LOWER(' + expr + ')'

    def func_substring(self, expr, start, length=None):
        if length is not None:
            return 'SUBSTRING(' + expr + ', ' + str(start) + ', ' + str(length) + ')'
        return 'SUBSTRING(' + expr + ', ' + str(start) + ')'

    def parse_datetime(self, expr, format):
        return 'STR_TO_DATE(' + expr + ', '' + format + '')'

    def extract_date_part(self, expr, part):
        return 'EXTRACT(' + part + ' FROM ' + expr + ')'

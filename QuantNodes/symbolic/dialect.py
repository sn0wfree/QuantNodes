# coding=utf-8
"""符号计算引擎 - 数据库方言抽象"""
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
    def quote_identifier(self, name: str) -> str: pass
    @abstractmethod
    def quote_literal(self, value: Any) -> str: pass
    @abstractmethod
    def func_now(self) -> str: pass
    @abstractmethod
    def func_date_diff(self, unit: str, start: str, end: str) -> str: pass
    @abstractmethod
    def func_coalesce(self, *args: str) -> str: pass
    @abstractmethod
    def func_ifnull(self, expr: str, default: str) -> str: pass
    @abstractmethod
    def func_if(self, condition: str, then: str, else_: str) -> str: pass
    @abstractmethod
    def func_case(self, when_clauses: List[Tuple[str, str]], else_: Optional[str] = None) -> str: pass
    @abstractmethod
    def func_cast(self, expr: str, target_type: str) -> str: pass
    @abstractmethod
    def func_interval(self, value: str, unit: str) -> str: pass
    @abstractmethod
    def is_distinct_from(self, a: str, b: str) -> str: pass
    @abstractmethod
    def is_not_distinct_from(self, a: str, b: str) -> str: pass
    @abstractmethod
    def func_rank(self) -> str: pass
    @abstractmethod
    def func_dense_rank(self) -> str: pass
    @abstractmethod
    def func_row_number(self) -> str: pass
    @abstractmethod
    def func_lag(self, expr: str, offset: int, default: Optional[str] = None) -> str: pass
    @abstractmethod
    def func_lead(self, expr: str, offset: int, default: Optional[str] = None) -> str: pass
    @abstractmethod
    def func_avg(self, expr: str) -> str: pass
    @abstractmethod
    def func_sum(self, expr: str) -> str: pass
    @abstractmethod
    def func_max(self, expr: str) -> str: pass
    @abstractmethod
    def func_min(self, expr: str) -> str: pass
    @abstractmethod
    def func_count(self, expr: str) -> str: pass
    @abstractmethod
    def func_stddev(self, expr: str) -> str: pass
    @abstractmethod
    def func_variance(self, expr: str) -> str: pass
    @abstractmethod
    def func_quantile(self, expr: str, quantile: float) -> str: pass
    @abstractmethod
    def func_abs(self, expr: str) -> str: pass
    @abstractmethod
    def func_ceil(self, expr: str) -> str: pass
    @abstractmethod
    def func_floor(self, expr: str) -> str: pass
    @abstractmethod
    def func_round(self, expr: str, decimals: int = 0) -> str: pass
    @abstractmethod
    def func_pow(self, expr: str, power: float) -> str: pass
    @abstractmethod
    def func_sqrt(self, expr: str) -> str: pass
    @abstractmethod
    def func_ln(self, expr: str) -> str: pass
    @abstractmethod
    def func_log10(self, expr: str) -> str: pass
    @abstractmethod
    def func_cumsum(self, expr: str) -> str: pass
    @abstractmethod
    def func_running_diff(self, expr: str) -> str: pass
    @abstractmethod
    def func_length(self, expr: str) -> str: pass
    @abstractmethod
    def func_concat(self, *args: str) -> str: pass
    @abstractmethod
    def func_upper(self, expr: str) -> str: pass
    @abstractmethod
    def func_lower(self, expr: str) -> str: pass
    @abstractmethod
    def func_substring(self, expr: str, start: int, length: Optional[int] = None) -> str: pass
    @abstractmethod
    def parse_datetime(self, expr: str, format: str) -> str: pass
    @abstractmethod
    def extract_date_part(self, expr: str, part: str) -> str: pass
class BaseSQLDialect(SQLDialect):
    def func_case(self, when_clauses: List[Tuple[str, str]], else_: Optional[str] = None) -> str:
        parts = ["CASE WHEN " + cond + " THEN " + val for cond, val in when_clauses]
        if else_:
            parts.append("ELSE " + else_)
        parts.append("END")
        return " ".join(parts)

    def func_cast(self, expr: str, target_type: str) -> str:
        return "CAST(" + expr + " AS " + target_type + ")"

    def func_rank(self) -> str: return "rank()"
    def func_dense_rank(self) -> str: return "dense_rank()"
    def func_row_number(self) -> str: return "row_number()"
    def func_avg(self, expr: str) -> str: return "avg(" + expr + ")"
    def func_sum(self, expr: str) -> str: return "sum(" + expr + ")"
    def func_max(self, expr: str) -> str: return "max(" + expr + ")"
    def func_min(self, expr: str) -> str: return "min(" + expr + ")"
    def func_count(self, expr: str) -> str: return "count(" + expr + ")"
    def func_abs(self, expr: str) -> str: return "abs(" + expr + ")"
    def func_floor(self, expr: str) -> str: return "floor(" + expr + ")"
    def func_round(self, expr: str, decimals: int = 0) -> str: return "round(" + expr + ", " + str(decimals) + ")"
    def func_pow(self, expr: str, power: float) -> str: return "pow(" + expr + ", " + str(power) + ")"
    def func_sqrt(self, expr: str) -> str: return "sqrt(" + expr + ")"
    def func_concat(self, *args: str) -> str: return "concat(" + ", ".join(args) + ")"
    def func_upper(self, expr: str) -> str: return "upper(" + expr + ")"
    def func_lower(self, expr: str) -> str: return "lower(" + expr + ")"

    def func_substring(self, expr: str, start: int, length: Optional[int] = None) -> str:
        if length is not None:
            return "substring(" + expr + ", " + str(start) + ", " + str(length) + ")"
        return "substring(" + expr + ", " + str(start) + ")"

class ClickHouseDialect(BaseSQLDialect):
    dialect_name = DialectType.CLICKHOUSE

    def quote_identifier(self, name: str) -> str: return "`" + name + "`"

    def quote_literal(self, value: Any) -> str:
        if value is None: return "NULL"
        if isinstance(value, str): return "'" + value.replace("'", "\'\'") + "'"
        if isinstance(value, bool): return "1" if value else "0"
        return str(value)

    def func_now(self) -> str: return "now()"
    def func_date_diff(self, unit: str, start: str, end: str) -> str: return "dateDiff('" + unit + "', " + start + ", " + end + ")"
    def func_coalesce(self, *args: str) -> str: return "coalesce(" + ", ".join(args) + ")"
    def func_ifnull(self, expr: str, default: str) -> str: return "ifNull(" + expr + ", " + default + ")"
    def func_if(self, condition: str, then: str, else_: str) -> str: return "if(" + condition + ", " + then + ", " + else_ + ")"
    def func_interval(self, value: str, unit: str) -> str: return "INTERVAL " + value + " " + unit
    def is_distinct_from(self, a: str, b: str) -> str: return a + " IS DISTINCT FROM " + b
    def is_not_distinct_from(self, a: str, b: str) -> str: return a + " IS NOT DISTINCT FROM " + b

    def func_lag(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        if default is not None: return "lagInFrame(" + expr + ", " + str(offset) + ", " + default + ")"
        return "lagInFrame(" + expr + ", " + str(offset) + ")"

    def func_lead(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        if default is not None: return "leadInFrame(" + expr + ", " + str(offset) + ", " + default + ")"
        return "leadInFrame(" + expr + ", " + str(offset) + ")"

    def func_stddev(self, expr: str) -> str: return "stddevPop(" + expr + ")"
    def func_variance(self, expr: str) -> str: return "varPop(" + expr + ")"
    def func_quantile(self, expr: str, quantile: float) -> str: return "quantile(" + str(quantile) + ")(" + expr + ")"
    def func_ceil(self, expr: str) -> str: return "ceil(" + expr + ")"
    def func_ln(self, expr: str) -> str: return "log(" + expr + ")"
    def func_log10(self, expr: str) -> str: return "log10(" + expr + ")"
    def func_cumsum(self, expr: str) -> str: return "sum(" + expr + ")"
    def func_running_diff(self, expr: str) -> str: return "runningDifference(" + expr + ")"
    def func_length(self, expr: str) -> str: return "length(" + expr + ")"
    def parse_datetime(self, expr: str, format: str) -> str: return "parseDateTime(" + expr + ", '" + format + "')"
    def extract_date_part(self, expr: str, part: str) -> str: return "extract(" + part + " FROM " + expr + ")"

class DuckDBDialect(BaseSQLDialect):
    dialect_name = DialectType.DUCKDB

    def quote_identifier(self, name: str) -> str: return '"' + name + '"'

    def quote_literal(self, value: Any) -> str:
        if value is None: return "NULL"
        if isinstance(value, str): return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bool): return "TRUE" if value else "FALSE"
        return str(value)

    def func_now(self) -> str: return "now()"
    def func_date_diff(self, unit: str, start: str, end: str) -> str: return "datediff('" + unit + "', " + start + ", " + end + ")"
    def func_coalesce(self, *args: str) -> str: return "coalesce(" + ", ".join(args) + ")"
    def func_ifnull(self, expr: str, default: str) -> str: return "ifnull(" + expr + ", " + default + ")"
    def func_if(self, condition: str, then: str, else_: str) -> str: return "if(" + condition + ", " + then + ", " + else_ + ")"
    def func_interval(self, value: str, unit: str) -> str: return "INTERVAL '" + value + "' " + unit
    def is_distinct_from(self, a: str, b: str) -> str: return a + " IS DISTINCT FROM " + b
    def is_not_distinct_from(self, a: str, b: str) -> str: return a + " IS NOT DISTINCT FROM " + b

    def func_lag(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        if default is not None: return "lag(" + expr + ", " + str(offset) + ", " + default + ")"
        return "lag(" + expr + ", " + str(offset) + ")"

    def func_lead(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        if default is not None: return "lead(" + expr + ", " + str(offset) + ", " + default + ")"
        return "lead(" + expr + ", " + str(offset) + ")"

    def func_stddev(self, expr: str) -> str: return "stddev_pop(" + expr + ")"
    def func_variance(self, expr: str) -> str: return "var_pop(" + expr + ")"
    def func_quantile(self, expr: str, quantile: float) -> str: return "quantile_cont(" + expr + ", " + str(quantile) + ")"
    def func_ceil(self, expr: str) -> str: return "ceil(" + expr + ")"
    def func_ln(self, expr: str) -> str: return "ln(" + expr + ")"
    def func_log10(self, expr: str) -> str: return "log10(" + expr + ")"
    def func_cumsum(self, expr: str) -> str: return "sum(" + expr + ")"
    def func_running_diff(self, expr: str) -> str: return expr + " - lag(" + expr + ", 1)"
    def func_length(self, expr: str) -> str: return "length(" + expr + ")"
    def parse_datetime(self, expr: str, format: str) -> str: return "strptime(" + expr + ", '" + format + "')"
    def extract_date_part(self, expr: str, part: str) -> str: return "date_part('" + part + "', " + expr + ")"

class MySQLDialect(BaseSQLDialect):
    dialect_name = DialectType.MYSQL

    def quote_identifier(self, name: str) -> str: return "`" + name + "`"

    def quote_literal(self, value: Any) -> str:
        if value is None: return "NULL"
        if isinstance(value, str): return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bool): return "1" if value else "0"
        return str(value)

    def func_now(self) -> str: return "NOW()"
    def func_date_diff(self, unit: str, start: str, end: str) -> str: return "DATEDIFF(" + end + ", " + start + ")"
    def func_coalesce(self, *args: str) -> str: return "COALESCE(" + ", ".join(args) + ")"
    def func_ifnull(self, expr: str, default: str) -> str: return "IFNULL(" + expr + ", " + default + ")"
    def func_if(self, condition: str, then: str, else_: str) -> str: return "IF(" + condition + ", " + then + ", " + else_ + ")"
    def func_interval(self, value: str, unit: str) -> str: return "INTERVAL " + value + " " + unit

    def is_distinct_from(self, a: str, b: str) -> str:
        return "(" + a + " IS NULL AND " + b + " IS NOT NULL OR " + a + " IS NOT NULL AND " + b + " IS NULL OR " + a + " <> " + b + ")"

    def is_not_distinct_from(self, a: str, b: str) -> str:
        return "(" + a + " IS NULL AND " + b + " IS NULL OR " + a + " = " + b + ")"

    def func_lag(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        if default is not None: return "LAG(" + expr + ", " + str(offset) + ", " + default + ")"
        return "LAG(" + expr + ", " + str(offset) + ")"

    def func_lead(self, expr: str, offset: int, default: Optional[str] = None) -> str:
        if default is not None: return "LEAD(" + expr + ", " + str(offset) + ", " + default + ")"
        return "LEAD(" + expr + ", " + str(offset) + ")"

    def func_stddev(self, expr: str) -> str: return "STDDEV_POP(" + expr + ")"
    def func_variance(self, expr: str) -> str: return "VAR_POP(" + expr + ")"
    def func_quantile(self, expr: str, quantile: float) -> str: return "QUANTILE_CONT(" + expr + ", " + str(quantile) + ")"
    def func_ceil(self, expr: str) -> str: return "CEILING(" + expr + ")"
    def func_ln(self, expr: str) -> str: return "LN(" + expr + ")"
    def func_log10(self, expr: str) -> str: return "LOG10(" + expr + ")"
    def func_cumsum(self, expr: str) -> str: return "SUM(" + expr + ")"
    def func_running_diff(self, expr: str) -> str: return expr + " - LAG(" + expr + ", 1)"
    def func_length(self, expr: str) -> str: return "LENGTH(" + expr + ")"
    def func_upper(self, expr: str) -> str: return "UPPER(" + expr + ")"
    def func_lower(self, expr: str) -> str: return "LOWER(" + expr + ")"
    def func_substring(self, expr: str, start: int, length: Optional[int] = None) -> str:
        if length is not None: return "SUBSTRING(" + expr + ", " + str(start) + ", " + str(length) + ")"
        return "SUBSTRING(" + expr + ", " + str(start) + ")"
    def parse_datetime(self, expr: str, format: str) -> str: return "STR_TO_DATE(" + expr + ", '" + format + "')"
    def extract_date_part(self, expr: str, part: str) -> str: return "EXTRACT(" + part + " FROM " + expr + ")"

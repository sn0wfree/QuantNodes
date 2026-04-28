# coding=utf-8
"""
符号计算引擎

提供表达式到 SQL 的编译能力，支持多种数据库方言。
"""

from QuantNodes.symbolic.dialect import (
    DialectType,
    SQLDialect,
    ClickHouseDialect,
    DuckDBDialect,
    MySQLDialect,
)
from QuantNodes.symbolic.compiler import SQLCompiler, compile_expression
from QuantNodes.symbolic.expression import (
    SQLExpression,
    ColumnRef,
    LiteralValue,
    SQLBinaryOp,
    SQLUnaryOp,
    SQLComparison,
    SQLLogicalOp,
    SQLFunction,
    SQLCase,
)
from QuantNodes.symbolic.functions import (
    TechnicalFunctions,
    TA_FUNCTIONS,
)

__all__ = [
    "DialectType",
    "SQLDialect",
    "ClickHouseDialect",
    "DuckDBDialect",
    "MySQLDialect",
    "SQLCompiler",
    "compile_expression",
    "SQLExpression",
    "ColumnRef",
    "LiteralValue",
    "SQLBinaryOp",
    "SQLUnaryOp",
    "SQLComparison",
    "SQLLogicalOp",
    "SQLFunction",
    "SQLCase",
    "TechnicalFunctions",
    "TA_FUNCTIONS",
]

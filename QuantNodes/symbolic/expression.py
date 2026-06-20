# coding=utf-8
"""
符号计算引擎 - SQL 表达式节点

定义用于 SQL 编译的表达式 AST 节点。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class SQLExpression(ABC):
    """SQL 表达式基类"""

    @abstractmethod
    def to_sql(self, dialect: "SQLDialect") -> str:  # noqa: F821
        """编译为 SQL 字符串"""
        pass

    def __add__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(self, "+", wrap_sql_expr(other))

    def __radd__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(wrap_sql_expr(other), "+", self)

    def __sub__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(self, "-", wrap_sql_expr(other))

    def __rsub__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(wrap_sql_expr(other), "-", self)

    def __mul__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(self, "*", wrap_sql_expr(other))

    def __rmul__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(wrap_sql_expr(other), "*", self)

    def __truediv__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(self, "/", wrap_sql_expr(other))

    def __rtruediv__(self, other: Any) -> "SQLBinaryOp":
        return SQLBinaryOp(wrap_sql_expr(other), "/", self)

    def __neg__(self) -> "SQLUnaryOp":
        return SQLUnaryOp("-", self)

    def __gt__(self, other: Any) -> "SQLComparison":
        return SQLComparison(self, ">", wrap_sql_expr(other))

    def __ge__(self, other: Any) -> "SQLComparison":
        return SQLComparison(self, ">=", wrap_sql_expr(other))

    def __lt__(self, other: Any) -> "SQLComparison":
        return SQLComparison(self, "<", wrap_sql_expr(other))

    def __le__(self, other: Any) -> "SQLComparison":
        return SQLComparison(self, "<=", wrap_sql_expr(other))

    def __eq__(self, other: Any) -> "SQLComparison":
        return SQLComparison(self, "==", wrap_sql_expr(other))

    def __ne__(self, other: Any) -> "SQLComparison":
        return SQLComparison(self, "!=", wrap_sql_expr(other))

    def __and__(self, other: Any) -> "SQLLogicalOp":
        return SQLLogicalOp("AND", self, wrap_sql_expr(other))

    def __or__(self, other: Any) -> "SQLLogicalOp":
        return SQLLogicalOp("OR", self, wrap_sql_expr(other))

    def __invert__(self) -> "SQLLogicalOp":
        return SQLLogicalOp("NOT", self)


class ColumnRef(SQLExpression):
    """列引用表达式"""

    def __init__(self, name: str, table: Optional[str] = None):
        self.name = name
        self.table = table

    def to_sql(self, dialect) -> str:
        if self.table:
            return f"{dialect.quote_identifier(self.table)}.{dialect.quote_identifier(self.name)}"
        return dialect.quote_identifier(self.name)

    def __repr__(self) -> str:
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name


class LiteralValue(SQLExpression):
    """字面量值表达式"""

    def __init__(self, value: Any):
        self.value = value

    def to_sql(self, dialect) -> str:
        return dialect.quote_literal(self.value)

    def __repr__(self) -> str:
        return repr(self.value)


class SQLBinaryOp(SQLExpression):
    """二元运算表达式"""

    def __init__(self, left: SQLExpression, op: str, right: SQLExpression):
        self.left = left
        self.op = op
        self.right = right

    def to_sql(self, dialect) -> str:
        left_sql = self.left.to_sql(dialect)
        right_sql = self.right.to_sql(dialect)
        return f"({left_sql} {self.op} {right_sql})"

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class SQLUnaryOp(SQLExpression):
    """一元运算表达式"""

    def __init__(self, op: str, operand: SQLExpression):
        self.op = op
        self.operand = operand

    def to_sql(self, dialect) -> str:
        operand_sql = self.operand.to_sql(dialect)
        return f"({self.op} {operand_sql})"

    def __repr__(self) -> str:
        return f"({self.op} {self.operand})"


class SQLComparison(SQLExpression):
    """比较运算表达式"""

    def __init__(self, left: SQLExpression, op: str, right: SQLExpression):
        self.left = left
        self.op = op
        self.right = right

    def to_sql(self, dialect) -> str:
        left_sql = self.left.to_sql(dialect)
        right_sql = self.right.to_sql(dialect)
        return f"({left_sql} {self.op} {right_sql})"

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class SQLLogicalOp(SQLExpression):
    """逻辑运算表达式"""

    def __init__(self, op: str, *operands: SQLExpression):
        self.op = op
        self.operands = operands

    def to_sql(self, dialect) -> str:
        if self.op == "NOT":
            operand_sql = self.operands[0].to_sql(dialect)
            return f"(NOT {operand_sql})"
        parts = [op.to_sql(dialect) for op in self.operands]
        return "(" + self.op + " " + " ".join(parts) + ")"

    def __repr__(self) -> str:
        if self.op == "NOT":
            return f"(NOT {self.operands[0]})"
        return f"({' ' + self.op + ' '.join(str(op) for op in self.operands)})"


class SQLFunction(SQLExpression):
    """SQL 函数调用表达式"""

    def __init__(self, name: str, *args: SQLExpression, **kwargs: SQLExpression):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def to_sql(self, dialect) -> str:
        args_sql = [arg.to_sql(dialect) for arg in self.args]
        kwargs_sql = [f"{k}={v.to_sql(dialect)}" for k, v in self.kwargs.items()]
        all_args = ", ".join(args_sql + kwargs_sql)
        return f"{self.name}({all_args})"

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.name}({args_str})"


class SQLCase(SQLExpression):
    """CASE WHEN 表达式"""

    def __init__(
        self,
        when_clauses: List[Tuple[SQLExpression, SQLExpression]],
        else_: Optional[SQLExpression] = None,
    ):
        self.when_clauses = when_clauses
        self.else_ = else_

    def to_sql(self, dialect) -> str:
        when_sql = [(cond.to_sql(dialect), val.to_sql(dialect)) for cond, val in self.when_clauses]
        else_sql = self.else_.to_sql(dialect) if self.else_ else None
        return dialect.func_case(when_sql, else_sql)

    def __repr__(self) -> str:
        else_str = f" ELSE {self.else_}" if self.else_ else ""
        when_str = " ".join(f"WHEN {cond} THEN {val}" for cond, val in self.when_clauses)
        return f"CASE {when_str}{else_str} END"


def wrap_sql_expr(value: Any) -> SQLExpression:
    """将值包装为 SQL 表达式"""
    if isinstance(value, SQLExpression):
        return value
    if isinstance(value, str):
        return ColumnRef(value)
    return LiteralValue(value)


def col(name: str, table: Optional[str] = None) -> ColumnRef:
    """创建列引用"""
    return ColumnRef(name, table)


def lit(value: Any) -> LiteralValue:
    """创建字面量"""
    return LiteralValue(value)

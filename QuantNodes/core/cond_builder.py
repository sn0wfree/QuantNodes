# coding=utf-8
"""
Cond DSL 入口构建器

提供简洁的表达式构建 API：
    Cond('close') > 50           # 按列名访问
    Cond.attr('metrics').sharpe  # 按属性名访问
    Cond['close']                # 下标访问
    Cond.close                   # 属性访问
"""

from typing import Any

from QuantNodes.core.expression import (
    ExpressionBuilder,
    InputExpr,
    VariableExpr,
    AttributeExpr,
    ConstantExpr,
)


class _CondBuilder:
    """Cond 入口构建器 - 只在入口处提供特殊语法"""

    def __call__(self, name: str) -> ExpressionBuilder:
        """Cond('column_name') - 按列/变量名访问"""
        return ExpressionBuilder(VariableExpr(name))

    def attr(self, name: str) -> ExpressionBuilder:
        """Cond.attr('metrics') - 按属性名访问"""
        return ExpressionBuilder(AttributeExpr(InputExpr(), name))

    def constant(self, value: Any) -> ExpressionBuilder:
        """常量值"""
        return ExpressionBuilder(ConstantExpr(value))

    def __getattr__(self, name: str) -> ExpressionBuilder:
        """支持 Cond.metrics 形式的属性访问"""
        return ExpressionBuilder(AttributeExpr(InputExpr(), name))

    def __getitem__(self, key: Any) -> ExpressionBuilder:
        """支持下标访问 Cond['close']"""
        return ExpressionBuilder(VariableExpr(key))

    @property
    def input(self) -> ExpressionBuilder:
        """返回输入数据本身"""
        return ExpressionBuilder(InputExpr())


# 全局单例
Cond = _CondBuilder()

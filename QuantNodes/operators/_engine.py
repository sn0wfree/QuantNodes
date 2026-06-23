# coding=utf-8
"""Dual-Engine support for Composite DAG operators (PR-QN-4, 2026-06-22)

Provides engine detection for LLM-generated code and dual whitelists
for YAML template validation (polars vs pandas, strict separation).

Usage:
    from QuantNodes.operators._engine import Engine, detect_engine

    engine = detect_engine("import pandas as pd\\nresult = df.groupby('x').mean()")
    assert engine == Engine.PANDAS

    engine = detect_engine("import polars as pl\\nresult = pl.col('x').mean()")
    assert engine == Engine.POLARS
"""
from __future__ import annotations

import ast
from enum import Enum
from typing import Set


class Engine(str, Enum):
    POLARS = "polars"
    PANDAS = "pandas"
    AUTO = "auto"


def detect_engine(code: str) -> Engine:
    """Scan code for import statements to detect which engine is used.

    Heuristics:
      - `import polars as pl` or `from polars` → POLARS
      - `import pandas as pd` or `from pandas` → PANDAS
      - Both present → POLARS (default, faster path)
      - Neither present → POLARS (safe default)

    Returns:
        Engine.POLARS or Engine.PANDAS (never AUTO)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Engine.POLARS

    has_pl = False
    has_pd = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "polars" or alias.name.startswith("polars."):
                    has_pl = True
                elif alias.name == "pandas" or alias.name.startswith("pandas."):
                    has_pd = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "polars" or node.module.startswith("polars.")):
                has_pl = True
            elif node.module and (node.module == "pandas" or node.module.startswith("pandas.")):
                has_pd = True

    if has_pd and not has_pl:
        return Engine.PANDAS
    return Engine.POLARS


# ===== YAML Template Whitelists (Strict Separation) =====

ALLOWED_FUNC_NAMES_POLARS: Set[str] = {
    # polars Expr methods
    "col", "lit", "when", "then", "otherwise",
    "abs", "log", "sqrt", "pow", "exp",
    "rolling_mean", "rolling_std", "rolling_corr",
    "rolling_sum", "rolling_min", "rolling_max", "rolling_median",
    "ewm_mean", "ewm_std",
    "shift", "diff", "pct_change", "rank",
    "mean", "std", "sum", "min", "max", "median", "quantile",
    "count", "first", "last",
    "group_by", "over", "alias",
    "clip", "fill_null", "fill_nan", "drop_nulls", "drop_nans",
    "is_null", "is_nan", "is_not_null",
    "round", "floor", "ceil",
    "and_", "or_", "not_",
}

ALLOWED_FUNC_NAMES_PANDAS: Set[str] = {
    # pandas Series/DataFrame methods
    "groupby", "transform", "agg", "apply", "pipe",
    "rolling", "expanding", "ewm",
    "shift", "diff", "pct_change", "rank",
    "fillna", "dropna", "isna", "notna", "isnull", "notnull",
    "clip", "round", "abs", "astype",
    "mean", "std", "sum", "min", "max", "median", "quantile",
    "count", "first", "last",
    "where", "mask", "assign",
    "resample", "asfreq",
    "merge", "join", "concat",
    "reset_index", "set_index",
    "head", "tail", "sort_values",
    "to_numpy", "values",
}

__all__ = [
    "Engine",
    "detect_engine",
    "ALLOWED_FUNC_NAMES_POLARS",
    "ALLOWED_FUNC_NAMES_PANDAS",
]

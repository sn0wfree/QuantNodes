# coding=utf-8
"""
alpha_evaluate.py - Alpha 公式评估工具（包 M4 PolarsAlphaCalculator）

Alpha-GPT 工作流的【第 3 阶段：Evaluator】核心工具。
把 LLM 生成的 polars 公式批量评估为 IC / IR / decay 等量化指标。

复用：
- M4 PolarsAlphaCalculator（M1-M4 适配器）
- M1 OperatorVocab（162 算子白名单校验）

Usage (作为 nanobot tool)::

    from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool
    tool = AlphaEvaluateTool()
    result = await tool.execute(
        formulas=["rank(-ts_mean(returns, 20))"],
        data=df,
        forward_returns=[1, 5, 20],
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .base import Tool

logger = logging.getLogger(__name__)


class AlphaEvaluateTool(Tool):
    """Alpha 公式评估工具

    批量评估一组 polars 公式，返回 IC / IR / decay 指标。
    输入支持：data_path（parquet/csv）或已加载的 polars.DataFrame。
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "alpha_evaluate"

    @property
    def description(self) -> str:
        return (
            "批量评估 alpha 公式。"
            "输入 formulas 列表（polars 表达式字符串）和数据，"
            "返回每个公式的 IC / IR / ic_decay 指标。"
            "可选：forward_returns 前瞻期列表（默认 [1]）。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "formulas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "待评估的 polars 公式列表",
                },
                "data_path": {
                    "type": "string",
                    "description": "数据路径（parquet/csv）；若与 data 同时给出，优先用 data",
                },
                "forward_returns": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [1],
                    "description": "前瞻期列表（默认 [1]）",
                },
                "date_column": {
                    "type": "string",
                    "default": "date",
                    "description": "日期列名",
                },
                "code_column": {
                    "type": "string",
                    "default": "code",
                    "description": "股票代码列名",
                },
                "max_workers": {
                    "type": "integer",
                    "default": 4,
                    "description": "并行评估的最大 worker 数（默认 4）",
                },
            },
            "required": ["formulas"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        formulas: Sequence[str],
        data_path: Optional[str] = None,
        forward_returns: Sequence[int] = (1,),
        date_column: str = "date",
        code_column: str = "code",
        max_workers: int = 4,
        data: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """批量评估公式

        Args:
            formulas: 待评估的 polars 表达式字符串列表
            data_path: 数据路径（与 data 互斥）
            forward_returns: 前瞻期列表
            date_column: 日期列名
            code_column: 代码列名
            max_workers: 并行 worker 数
            data: 已加载的 polars.DataFrame（用于测试场景）
            **kwargs: 兼容 nanobot 框架的额外参数

        Returns:
            {
                "evaluations": [
                    {
                        "formula": "rank(-ts_mean(returns, 20))",
                        "status": "success" | "failed",
                        "metrics": {ic_mean, ic_std, ir, ic_decay: {1: ..., 5: ...}},
                        "error_msg": str | None
                    }
                ],
                "summary": {total, success, failed, avg_ir, best_ir}
            }
        """
        try:
            import polars as pl
            from QuantNodes.research.quant_alpha.adapters import (
                PolarsAlphaCalculator,
            )

            df = self._resolve_data(data, data_path, date_column, code_column)
            if df is None:
                return self._err("No data provided (need data_path or data)")

            forward_returns_dict = self._build_forward_returns(
                df, list(forward_returns), date_column, code_column,
            )

            calc = PolarsAlphaCalculator(
                data=df,
                forward_returns=forward_returns_dict,
                date_column=date_column,
                code_column=code_column,
            )

            from QuantNodes.research.quant_alpha.adapters.expression import (
                expression_to_formula,
            )
            from QuantNodes.research.quant_alpha.adapters.expression import Feature

            evaluations = []
            for formula in formulas:
                evaluations.append(self._eval_one(calc, formula))

            return self._summarize(evaluations)
        except Exception as exc:
            logger.exception("alpha_evaluate failed")
            return self._err(str(exc))

    def _eval_one(
        self,
        calc: Any,
        formula: str,
    ) -> Dict[str, Any]:
        """评估单个公式"""
        try:
            expr = self._parse_simple_formula(formula)
            offsets = sorted(calc.forward_returns.keys())
            ic_means = {}
            for offset in offsets:
                arr = calc.calc_single_IC_ret(expr, ret_offset=offset)
                arr = arr[~np.isnan(arr)]
                ic_means[offset] = float(np.mean(arr)) if arr.size else 0.0

            primary_offset = offsets[0]
            primary_arr = calc.calc_single_IC_ret(expr, ret_offset=primary_offset)
            primary_arr = primary_arr[~np.isnan(primary_arr)]
            ic_mean = float(np.mean(primary_arr)) if primary_arr.size else 0.0
            ic_std = float(np.std(primary_arr)) if primary_arr.size else 0.0
            ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

            return {
                "formula": formula,
                "status": "success",
                "metrics": {
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "ir": ir,
                    "ic_decay": {str(k): v for k, v in ic_means.items()},
                },
                "error_msg": None,
            }
        except Exception as exc:
            return {
                "formula": formula,
                "status": "failed",
                "metrics": {},
                "error_msg": str(exc),
            }

    @staticmethod
    def _parse_simple_formula(formula: str):
        """把简单字符串公式解析为 Expression（轻量 parser，仅支持算子形式）

        支持的形态：
        - "Feature('close')" → Feature('close')
        - "Ref(Feature('close'), 2)" → Ref(Feature('close'), 2)
        - "-x" → Neg(x) (unary negation)
        - "rank(x)" / "ts_mean(x, 5)" / "ts_zscore(x, 20)" 等
        - 二元: "Add(a, b)" / "Sub(a, b)" / "Mul(a, b)" / "Div(a, b)"
        - Literal: "1e-12"
        - 简写字段名 "close" → Feature('close')

        重算子写在字符串外层，内层是子公式字符串（递归）。
        """
        from QuantNodes.research.quant_alpha.adapters.expression import (
            Literal,
            Ref,
            Feature,
            BinaryOp,
            UnaryOp,
            RollingOp,
        )
        import re

        formula = formula.strip()

        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", formula):
            return Literal(float(formula))

        if formula.startswith("-"):
            inner = formula[1:].strip()
            return UnaryOp(AlphaEvaluateTool._parse_simple_formula(inner), "neg")

        m = re.fullmatch(r"Feature\(['\"]([\w_]+)['\"]\)", formula)
        if m:
            return Feature(m.group(1))

        m = re.fullmatch(r"Ref\(Feature\(['\"]([\w_]+)['\"]\),\s*(\d+)\)", formula)
        if m:
            return Ref(Feature(m.group(1)), int(m.group(2)))

        # Ref(close, 1) 格式
        m = re.fullmatch(r"Ref\((\w+),\s*(\d+)\)", formula)
        if m:
            return Ref(Feature(m.group(1)), int(m.group(2)))

        # close.shift(1) 格式
        m = re.fullmatch(r"(\w+)\.shift\((\d+)\)", formula)
        if m:
            return Ref(Feature(m.group(1)), int(m.group(2)))

        # 简写字段名: "close", "vol" 等
        # 特殊处理 "returns" → (close - delay(close, 1)) / delay(close, 1)
        if formula == "returns":
            close_feat = Feature("close")
            delay1 = Ref(close_feat, 1)
            return BinaryOp(BinaryOp(close_feat, delay1, "sub"), delay1, "div")

        if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", formula):
            return Feature(formula)

        m = re.fullmatch(r"([\w_]+)\((.+)\)", formula, re.DOTALL)
        if m:
            op = m.group(1)
            inner = m.group(2)
            args = split_args(inner)
            parsed_args = [AlphaEvaluateTool._parse_simple_formula(a) for a in args]

            if op in {"Add", "Sub", "Mul", "Div", "Greater", "Less",
                      "add", "sub", "mul", "div", "greater", "less"}:
                op_key = op.lower()
                cls = {
                    "add": lambda a, b: BinaryOp(a, b, "add"),
                    "sub": lambda a, b: BinaryOp(a, b, "sub"),
                    "mul": lambda a, b: BinaryOp(a, b, "mul"),
                    "div": lambda a, b: BinaryOp(a, b, "div"),
                    "greater": lambda a, b: BinaryOp(a, b, "gt"),
                    "less": lambda a, b: BinaryOp(a, b, "lt"),
                }[op_key]
                if len(parsed_args) != 2:
                    raise ValueError(f"{op} needs 2 args, got {len(parsed_args)}")
                return cls(parsed_args[0], parsed_args[1])

            window_ops = {
                "ts_mean": "mean", "mean": "mean", "Mean": "mean",
                "ts_std": "std", "std": "std", "Std": "std",
                "ts_sum": "sum", "sum": "sum", "Sum": "sum",
                "ts_max": "max", "max": "max", "Max": "max",
                "ts_min": "min", "min": "min", "Min": "min",
                "ts_zscore": "zscore",
                "ts_rank": "rank",
                "ts_median": "median", "median": "median",
                "ts_skew": "skew", "skew": "skew",
                "ts_kurt": "kurt", "kurt": "kurt",
                "ts_var": "var", "var": "var",
                "delta": "delta", "Delta": "delta",
                "ts_delta": "delta",
                "ts_decay_linear": "decay_linear",
                "ts_corr": "corr",
                "ts_cov": "cov",
            }
            if op in window_ops:
                if len(parsed_args) != 2:
                    raise ValueError(f"{op} needs 2 args")
                return RollingOp(parsed_args[0], int(float(parsed_args[1].value)), window_ops[op])

            cross_sectional_ops = {"abs": "abs", "log": "log", "sqrt": "sqrt", "sign": "sign"}
            if op in cross_sectional_ops:
                if len(parsed_args) != 1:
                    raise ValueError(f"{op} needs 1 arg")
                return UnaryOp(cross_sectional_ops[op], parsed_args[0])

            if op in {"Abs", "Log", "Sqrt", "Sign"}:
                return UnaryOp(op.lower(), parsed_args[0])

            # rank (cross-sectional) - treat as unary "rank" op
            if op in {"rank", "Rank"}:
                if len(parsed_args) != 1:
                    raise ValueError(f"{op} needs 1 arg")
                return UnaryOp(parsed_args[0], "rank")

            # zscore (cross-sectional)
            if op in {"zscore", "Zscore"}:
                if len(parsed_args) != 1:
                    raise ValueError(f"{op} needs 1 arg")
                return UnaryOp(parsed_args[0], "zscore")

            # winsorize
            if op in {"winsorize", "Winsorize"}:
                if len(parsed_args) != 1:
                    raise ValueError(f"{op} needs 1 arg")
                return UnaryOp(parsed_args[0], "winsorize")

            # signedpower
            if op in {"signedpower", "SignedPower"}:
                if len(parsed_args) != 2:
                    raise ValueError(f"{op} needs 2 args")
                return BinaryOp(parsed_args[0], parsed_args[1], "signedpower")

            # IndNeutralize - treat as unary passthrough (no industry data)
            if op in {"IndNeutralize", "indneutralize"}:
                if len(parsed_args) != 1:
                    raise ValueError(f"{op} needs 1 arg")
                return parsed_args[0]

            # returns shorthand: returns = (close - delay(close, 1)) / delay(close, 1)
            if op in {"returns", "Returns"}:
                if len(parsed_args) != 1:
                    raise ValueError(f"{op} needs 1 arg")
                feat = parsed_args[0]
                delay1 = Ref(feat, 1)
                return BinaryOp(BinaryOp(feat, delay1, "sub"), delay1, "div")

            # delay(x, n) → Ref(x, n)
            if op in {"delay", "Delay"}:
                if len(parsed_args) != 2:
                    raise ValueError(f"{op} needs 2 args")
                return Ref(parsed_args[0], int(float(parsed_args[1].value)))

        raise ValueError(f"Cannot parse formula: {formula!r}")

    @staticmethod
    def _summarize(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        success = [e for e in evaluations if e["status"] == "success"]
        irs = [e["metrics"].get("ir", 0.0) for e in success]
        return {
            "evaluations": evaluations,
            "summary": {
                "total": len(evaluations),
                "success": len(success),
                "failed": len(evaluations) - len(success),
                "avg_ir": float(np.mean(irs)) if irs else 0.0,
                "best_ir": float(np.max(irs)) if irs else 0.0,
            },
        }

    @staticmethod
    def _resolve_data(
        data: Any,
        data_path: Optional[str],
        date_column: str,
        code_column: str,
    ) -> Any:
        import polars as pl

        if data is not None:
            return data
        if data_path is None:
            return None
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"data not found: {data_path}")
        if p.suffix == ".parquet":
            return pl.read_parquet(p)
        if p.suffix == ".csv":
            df = pl.read_csv(p)
            if date_column in df.columns and df[date_column].dtype == pl.Utf8:
                df = df.with_columns(pl.col(date_column).str.to_date())
            return df
        raise ValueError(f"Unsupported data format: {p.suffix}")

    @staticmethod
    def _build_forward_returns(
        df: Any,
        forward_returns: List[int],
        date_column: str,
        code_column: str,
    ) -> Dict[int, Any]:
        """从 close 计算前瞻 N 日收益"""
        import polars as pl

        if "close" not in df.columns:
            raise ValueError("data must have 'close' column for forward returns")

        out: Dict[int, Any] = {}
        sorted_df = df.sort([code_column, date_column])
        for offset in forward_returns:
            col_name = f"_fwd_ret_{offset}d"
            ret = sorted_df.with_columns(
                pl.col("close").shift(-offset).over(code_column).alias("_next_close")
            ).with_columns(
                ((pl.col("_next_close") - pl.col("close")) / pl.col("close")).alias(col_name)
            )[col_name]
            out[offset] = ret
        return out

    @staticmethod
    def _err(msg: str) -> Dict[str, Any]:
        return {"evaluations": [], "summary": {"error": msg}}


def split_args(s: str) -> List[str]:
    """Split a function-argument string by top-level commas, respecting parens/quotes."""
    parts: List[str] = []
    depth = 0
    in_str: Optional[str] = None
    buf: List[str] = []
    for ch in s:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


__all__ = ["AlphaEvaluateTool", "split_args"]

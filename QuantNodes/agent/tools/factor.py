# coding=utf-8
"""
因子分析工具

提供 IC 分析、相关性分析等功能。
"""

from typing import Any, Dict

from QuantNodes.agent.tools.base import Tool


class FactorTool(Tool):
    """因子分析工具

    对因子进行 IC 分析、相关性分析等。

    通过 CodeSandbox 安全执行因子代码，获取因子值，
    然后计算 IC、ICIR 等统计指标。

    factor_code 示例:
        import polars as pl
        result = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["A", "B", "A", "B"],
            "factor_value": [0.1, 0.2, 0.3, 0.4],
            "forward_return": [0.05, 0.03, 0.02, 0.01],
        })
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "factor"

    @property
    def description(self) -> str:
        return (
            "对因子进行IC分析、相关性分析等。"
            "factor_code 中需将结果赋给 result 变量，"
            "result 应包含 date, code, factor_value, forward_return 列。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "factor_code": {
                    "type": "string",
                    "description": "因子的Python代码，结果应赋给 'result' 变量，包含 date/code/factor_value/forward_return 列"
                },
                "analysis_type": {
                    "type": "string",
                    "description": "分析类型：ic（IC分析）、correlation（相关性分析）、both",
                    "enum": ["ic", "correlation", "both"],
                    "default": "both"
                },
                "start_date": {
                    "type": "string",
                    "description": "分析开始日期"
                },
                "end_date": {
                    "type": "string",
                    "description": "分析结束日期"
                }
            },
            "required": ["factor_code", "analysis_type"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        factor_code: str,
        analysis_type: str = "both",
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        result = {
            "status": "success",
            "analysis": {
                "ic": {},
                "correlation": {}
            }
        }

        try:
            from QuantNodes.ai.sandbox import CodeSandbox
            import polars as pl

            sandbox = CodeSandbox()
            validation = sandbox.validate(factor_code)

            if not validation.is_safe:
                result["status"] = "error"
                result["errors"] = validation.errors
                result["security_status"] = "unsafe"
                return result

            result["security_status"] = "safe"

            namespace = sandbox.validate_and_execute(factor_code)
            factor_result = namespace.get("result")

            if factor_result is None:
                result["status"] = "error"
                result["errors"] = [
                    "No 'result' variable found. Factor code must assign to 'result', e.g.:\n"
                    "result = pl.DataFrame({...})"
                ]
                return result

            if not isinstance(factor_result, pl.DataFrame):
                result["status"] = "error"
                result["errors"] = [
                    f"Expected Polars DataFrame, got {type(factor_result).__name__}. "
                    "Factor code should create a Polars DataFrame."
                ]
                return result

            required_cols = {"factor_value", "forward_return"}
            missing = required_cols - set(factor_result.columns)
            if missing:
                result["status"] = "error"
                result["errors"] = [
                    f"Missing required columns: {missing}. "
                    f"Available columns: {list(factor_result.columns)}. "
                    "Result must have 'factor_value' and 'forward_return' columns."
                ]
                return result

            if "date" in factor_result.columns and start_date and end_date:
                factor_result = factor_result.filter(
                    (pl.col("date") >= start_date) &
                    (pl.col("date") <= end_date)
                )

            if analysis_type in ("ic", "both"):
                result["analysis"]["ic"] = self._compute_ic(factor_result)

            if analysis_type in ("correlation", "both"):
                result["analysis"]["correlation"] = self._compute_correlation(factor_result)

            result["data_rows"] = len(factor_result)
            result["columns"] = list(factor_result.columns)

        except Exception as e:
            result["status"] = "error"
            result["errors"] = [str(e)]

        return result

    def _compute_ic(self, df) -> Dict[str, Any]:
        """计算 IC 分析结果"""
        import polars as pl

        if "date" not in df.columns:
            factor = df["factor_value"]
            fwd = df["forward_return"]
            ic_val = factor.to_frame().select(pl.corr(factor, fwd)).to_series()[0]
            rank_ic_val = factor.rank().to_frame().select(pl.corr(factor.rank(), fwd.rank())).to_series()[0]
            return {
                "ic_mean": ic_val,
                "ic_std": 0.0,
                "icir": ic_val,
                "rank_ic_mean": rank_ic_val,
                "note": "Single cross-section IC (no date column for time-series analysis)",
            }

        ic_series = df.group_by("date").agg([
            pl.corr("factor_value", "forward_return").alias("ic"),
        ]).sort("date")

        ic_values = ic_series["ic"].drop_nulls()
        ic_mean = ic_values.mean()
        ic_std = ic_values.std()
        icir = ic_mean / (ic_std + 1e-8) if ic_std else 0.0

        rank_ic_series = df.group_by("date").agg([
            pl.corr(
                pl.col("factor_value").rank(),
                pl.col("forward_return").rank()
            ).alias("rank_ic"),
        ]).sort("date")

        rank_ic_values = rank_ic_series["rank_ic"].drop_nulls()
        rank_ic_mean = rank_ic_values.mean()

        return {
            "ic_mean": round(ic_mean, 6) if ic_mean is not None else None,
            "ic_std": round(ic_std, 6) if ic_std is not None else None,
            "icir": round(icir, 6) if icir is not None else None,
            "rank_ic_mean": round(rank_ic_mean, 6) if rank_ic_mean is not None else None,
            "ic_series": ic_series.to_dicts(),
            "n_dates": len(ic_series),
        }

    def _compute_correlation(self, df) -> Dict[str, Any]:
        """计算相关性分析"""
        import polars as pl

        result = {}

        if "factor_value" in df.columns and "forward_return" in df.columns:
            factor = df["factor_value"]
            fwd = df["forward_return"]
            corr = factor.to_frame().select(pl.corr(factor, fwd)).to_series()[0]
            result["factor_return_corr"] = round(corr, 6) if corr is not None else None

        numeric_cols = [c for c in df.columns if df[c].dtype in (
            pl.Float32, pl.Float64, pl.Int32, pl.Int64
        )]
        if len(numeric_cols) >= 2:
            corr_matrix = df.select(numeric_cols).corr()
            result["correlation_matrix"] = {
                "columns": numeric_cols,
                "values": corr_matrix.to_dicts(),
            }

        return result

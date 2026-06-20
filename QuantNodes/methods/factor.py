# coding=utf-8
"""
Factor Method

analyze_factor(factor_code, analysis_type) -> FactorAnalysisResult

Performs IC analysis, correlation analysis on factors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FactorAnalysisResult:
    status: str
    analysis: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    security_status: str = "unknown"
    data_rows: int = 0
    columns: List[str] = field(default_factory=list)


def analyze_factor(
    factor_code: str,
    analysis_type: str = "both",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> FactorAnalysisResult:
    """Analyze a factor with IC and correlation analysis.

    Args:
        factor_code: Factor code that assigns result variable
        analysis_type: "ic", "correlation", or "both"
        start_date: Analysis start date
        end_date: Analysis end date

    Returns:
        FactorAnalysisResult with IC and correlation metrics
    """
    result = FactorAnalysisResult(
        status="success",
        analysis={"ic": {}, "correlation": {}}
    )

    try:
        from QuantNodes.ai.sandbox import CodeSandbox
        import polars as pl

        sandbox = CodeSandbox()
        validation = sandbox.validate(factor_code)

        if not validation.is_safe:
            result.status = "error"
            result.errors = validation.errors
            result.security_status = "unsafe"
            return result

        result.security_status = "safe"

        namespace = sandbox.validate_and_execute(factor_code)
        factor_result = namespace.get("result")

        if factor_result is None:
            result.status = "error"
            result.errors = [
                "No 'result' variable found. Factor code must assign to 'result', e.g.:\n"
                "result = pl.DataFrame({...})"
            ]
            return result

        if not isinstance(factor_result, pl.DataFrame):
            result.status = "error"
            result.errors = [
                f"Expected Polars DataFrame, got {type(factor_result).__name__}. "
                "Factor code should create a Polars DataFrame."
            ]
            return result

        required_cols = {"factor_value", "forward_return"}
        missing = required_cols - set(factor_result.columns)
        if missing:
            result.status = "error"
            result.errors = [
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
            result.analysis["ic"] = _compute_ic(factor_result)

        if analysis_type in ("correlation", "both"):
            result.analysis["correlation"] = _compute_correlation(factor_result)

        result.data_rows = len(factor_result)
        result.columns = list(factor_result.columns)

    except Exception as e:
        result.status = "error"
        result.errors = [str(e)]

    return result


def _compute_ic(df) -> Dict[str, Any]:
    """Compute IC analysis results."""
    import polars as pl

    if "date" not in df.columns:
        factor = df["factor_value"]
        fwd = df["forward_return"]
        ic_val = factor.to_frame().select(pl.corr(factor, fwd)).to_series()[0]
        rank_ic_val = factor.rank().to_frame().select(
            pl.corr(factor.rank(), fwd.rank())
        ).to_series()[0]
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


def _compute_correlation(df) -> Dict[str, Any]:
    """Compute correlation analysis."""
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

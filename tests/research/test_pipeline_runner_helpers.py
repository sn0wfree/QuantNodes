"""pipeline_runner.py helper 边界测试 (10 tests)。"""
from __future__ import annotations

import pytest

from QuantNodes.research.factor_test.pipeline_runner import _extract_metrics_from_ctx


class TestExtractMetricsFromCtx:
    def test_empty_ctx(self):
        assert _extract_metrics_from_ctx({}) == {}

    def test_ic_metrics(self):
        ctx = {"ICAnalyzer": {"ic_result": {
            "IC均值": 0.01,
            "Rank IC均值": "0.02",
            "ICIR": 1.5,
        }}}
        m = _extract_metrics_from_ctx(ctx)
        assert m == {"ic_mean": 0.01, "rank_ic_mean": 0.02, "ic_ir": 1.5}

    def test_longshort_metrics(self):
        ctx = {"LongShort": {
            "sharpe": "1.2",
            "annualized_return": 0.3,
            "max_drawdown": 0.1,
            "calmar": 3.0,
        }}
        m = _extract_metrics_from_ctx(ctx)
        assert m == {"sharpe": 1.2, "arr": 0.3, "mdd": 0.1, "calmar": 3.0}

    def test_combined_metrics(self):
        ctx = {
            "ICAnalyzer": {"ic_result": {"IC均值": 0.01}},
            "LongShort": {"sharpe": 1.0},
        }
        m = _extract_metrics_from_ctx(ctx)
        assert m["ic_mean"] == 0.01
        assert m["sharpe"] == 1.0

    def test_invalid_values_skipped(self):
        ctx = {
            "ICAnalyzer": {"ic_result": {"IC均值": "bad", "ICIR": None}},
            "LongShort": {"sharpe": object(), "annualized_return": None},
        }
        assert _extract_metrics_from_ctx(ctx) == {}

    def test_non_dict_sections(self):
        ctx = {"ICAnalyzer": "bad", "LongShort": "bad"}
        assert _extract_metrics_from_ctx(ctx) == {}

    def test_missing_ic_result(self):
        ctx = {"ICAnalyzer": {"other": 1}}
        assert _extract_metrics_from_ctx(ctx) == {}

    def test_nan_float_kept(self):
        ctx = {"LongShort": {"sharpe": float("nan")}}
        m = _extract_metrics_from_ctx(ctx)
        assert "sharpe" in m
        assert m["sharpe"] != m["sharpe"]

    def test_zero_values_kept(self):
        ctx = {"LongShort": {"sharpe": 0, "annualized_return": 0}}
        assert _extract_metrics_from_ctx(ctx) == {"sharpe": 0.0, "arr": 0.0}

    def test_extra_keys_ignored(self):
        ctx = {"LongShort": {"sharpe": 1.0, "foo": 2}}
        assert _extract_metrics_from_ctx(ctx) == {"sharpe": 1.0}

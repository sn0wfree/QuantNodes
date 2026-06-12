"""core/constants.py 统一常量 parametrize (~20 tests)。"""
from __future__ import annotations

import pytest

from QuantNodes.core.constants import (
    BASE_FEATURE_NAMES,
    EXTENDED_METRIC_KEYS,
    METRIC_KEYS,
    PARQUET_COLUMNS,
)


class TestBaseFeatureNames:
    @pytest.mark.parametrize("name", [
        "open", "high", "low", "close", "volume", "amount",
        "vwap", "turnover", "mv_float", "total_mv", "circ_mv",
        "returns", "vwap_adj", "industry", "cap",
    ])
    def test_required_features(self, name):
        assert name in BASE_FEATURE_NAMES

    def test_count(self):
        assert len(BASE_FEATURE_NAMES) == 15

    def test_is_frozenset(self):
        assert isinstance(BASE_FEATURE_NAMES, frozenset)

    def test_dedup(self):
        """重复添加应去重。"""
        assert len(BASE_FEATURE_NAMES) == len(set(BASE_FEATURE_NAMES))


class TestMetricKeys:
    @pytest.mark.parametrize("key", [
        "ic_mean", "rank_ic_mean", "sharpe", "arr", "mdd", "calmar",
    ])
    def test_required(self, key):
        assert key in METRIC_KEYS

    def test_count(self):
        assert len(METRIC_KEYS) == 6

    def test_is_tuple(self):
        assert isinstance(METRIC_KEYS, tuple)

    def test_no_duplicates(self):
        assert len(METRIC_KEYS) == len(set(METRIC_KEYS))


class TestExtendedMetricKeys:
    @pytest.mark.parametrize("key", [
        "ic", "rank_ic", "sharpe", "arr", "mdd", "calmar",
        "turnover", "win_rate", "ic_ir",
    ])
    def test_required(self, key):
        assert key in EXTENDED_METRIC_KEYS

    def test_count(self):
        assert len(EXTENDED_METRIC_KEYS) == 9


class TestParquetColumns:
    @pytest.mark.parametrize("col", [
        "entry_id", "round_idx", "operation", "parent_ids",
        "decision", "duration_ms", "timestamp", "factor_name", "summary",
        "ic_mean", "rank_ic_mean", "sharpe", "arr", "mdd", "calmar",
    ])
    def test_required_columns(self, col):
        assert col in PARQUET_COLUMNS

    def test_count(self):
        """9 业务字段 + 6 metric = 15。"""
        assert len(PARQUET_COLUMNS) == 15

    def test_metric_subset(self):
        """PARQUET_COLUMNS 包含全部 METRIC_KEYS。"""
        for m in METRIC_KEYS:
            assert m in PARQUET_COLUMNS


class TestUnifiedAcrossModules:
    """验证 4 处定义现在指向同一来源。"""

    def test_feedback_channels_uses_unified(self):
        from QuantNodes.core.feedback.channels import _BASE_FEATURE_NAMES
        assert _BASE_FEATURE_NAMES is BASE_FEATURE_NAMES

    def test_quality_gate_complexity_uses_unified(self):
        from QuantNodes.core.quality_gate.complexity import _BASE_FEATURE_NAMES
        assert _BASE_FEATURE_NAMES is BASE_FEATURE_NAMES

    def test_trajectory_entry_uses_unified(self):
        from QuantNodes.core.trajectory.entry import _METRIC_KEYS
        assert _METRIC_KEYS == METRIC_KEYS

    def test_trajectory_pool_uses_unified(self):
        from QuantNodes.core.trajectory.pool import _PARQUET_COLUMNS
        assert _PARQUET_COLUMNS == PARQUET_COLUMNS

    def test_feedback_dataclass_uses_unified(self):
        from QuantNodes.core.feedback.dataclass import KNOWN_METRICS
        assert KNOWN_METRICS == EXTENDED_METRIC_KEYS

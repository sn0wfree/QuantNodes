"""_jsonify + TrajectoryEntry 序列化边界测试 (15 tests)。

聚焦:
    - _jsonify: 各种类型 (str/int/float/bool/None/list/dict)
    - _jsonify: datetime / pd.Timestamp / ndarray / 带 isoformat 的对象
    - 嵌套结构
    - 不可序列化对象 → 转 str
    - TrajectoryEntry 序列化含 ndarray/pd.Timestamp/None
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.trajectory import TrajectoryEntry
from QuantNodes.core.trajectory.entry import _jsonify


# ============================================================================
# 1. _jsonify 基本类型 (6 tests)
# ============================================================================

class TestJsonifyBasic:
    def test_none(self):
        assert _jsonify(None) is None

    def test_str(self):
        assert _jsonify("hello") == "hello"

    def test_int(self):
        assert _jsonify(42) == 42

    def test_float(self):
        assert _jsonify(3.14) == 3.14

    def test_bool(self):
        assert _jsonify(True) is True
        assert _jsonify(False) is False

    def test_empty_list(self):
        assert _jsonify([]) == []

    def test_empty_dict(self):
        assert _jsonify({}) == {}


# ============================================================================
# 2. _jsonify datetime (3 tests)
# ============================================================================

class TestJsonifyDatetime:
    def test_datetime_isoformat(self):
        dt = datetime(2025, 6, 11, 10, 30, 0)
        assert _jsonify(dt) == "2025-06-11T10:30:00"

    def test_pd_timestamp_isoformat(self):
        ts = pd.Timestamp("2025-06-11")
        assert _jsonify(ts) == "2025-06-11T00:00:00"

    def test_nested_datetime(self):
        """嵌套 dict 含 datetime → 全部 isoformat。"""
        dt = datetime(2025, 6, 11)
        result = _jsonify({"date": dt, "name": "x"})
        assert result["name"] == "x"
        assert result["date"] == "2025-06-11T00:00:00"


# ============================================================================
# 3. _jsonify 容器 (3 tests)
# ============================================================================

class TestJsonifyContainers:
    def test_list_of_mixed(self):
        result = _jsonify([1, "x", None, True, 3.14])
        assert result == [1, "x", None, True, 3.14]

    def test_dict_of_mixed(self):
        result = _jsonify({"a": 1, "b": [1, 2], "c": None})
        assert result == {"a": 1, "b": [1, 2], "c": None}

    def test_nested_list_of_dict(self):
        data = [{"x": 1}, {"y": 2}]
        result = _jsonify(data)
        assert result == [{"x": 1}, {"y": 2}]

    def test_tuple_converted_to_list(self):
        result = _jsonify((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)


# ============================================================================
# 4. _jsonify 复杂对象 (3 tests)
# ============================================================================

class TestJsonifyComplex:
    def test_ndarray_converted_to_str(self):
        """ndarray 无 isoformat → 走 str() 兜底。"""
        arr = np.array([1, 2, 3])
        result = _jsonify(arr)
        assert isinstance(result, str)
        assert "1" in result

    def test_object_with_isoformat(self):
        """自定义对象有 isoformat → 调用它。"""
        class WithIso:
            def isoformat(self):
                return "custom-iso"
        result = _jsonify(WithIso())
        assert result == "custom-iso"

    def test_object_without_isoformat_falls_to_str(self):
        class NoIso:
            def __str__(self):
                return "fallback_str"
        result = _jsonify(NoIso())
        assert result == "fallback_str"

    def test_isoformat_raises_falls_to_str(self):
        class BadIso:
            def isoformat(self):
                raise ValueError("nope")
        result = _jsonify(BadIso())
        # except 走 str()
        assert isinstance(result, str)


# ============================================================================
# 5. TrajectoryEntry 含复杂字段 (3 tests)
# ============================================================================

class TestTrajectoryEntryComplex:
    def test_config_with_ndarray(self):
        """config_snapshot 含 ndarray。"""
        arr = np.array([1.0, 2.0])
        e = TrajectoryEntry(
            entry_id="e1",
            config_snapshot={"data": arr},
        )
        d = e.to_json_dict()
        # ndarray → str
        assert isinstance(d["config_snapshot"]["data"], str)
        # 仍可 JSON 序列化
        json.dumps(d)

    def test_context_with_timestamp(self):
        ts = pd.Timestamp("2025-06-11")
        e = TrajectoryEntry(
            entry_id="e1",
            context_subset={"date": ts},
        )
        d = e.to_json_dict()
        assert d["context_subset"]["date"] == "2025-06-11T00:00:00"

    def test_metrics_with_ndarray(self):
        arr = np.array([0.1, 0.2])
        e = TrajectoryEntry(
            entry_id="e1",
            metrics={"vals": arr, "sharpe": 0.5},
        )
        d = e.to_json_dict()
        # ndarray → str
        assert isinstance(d["metrics"]["vals"], str)
        # 其他字段保留
        assert d["metrics"]["sharpe"] == 0.5
        # 可 JSON
        json.dumps(d)

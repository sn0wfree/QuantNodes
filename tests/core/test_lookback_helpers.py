# coding=utf-8
"""QuantNodes.core._lookback_helpers 单元测试"""
import numpy as np
import pytest

from QuantNodes.core._lookback_helpers import compute_lookback_params, extend_dt_ruler


class TestComputeLookbackParams:
    def test_rolling_window(self):
        lookback = [5, 10]
        modes = ["滚动窗口", "滚动窗口"]
        result, max_lb, max_len = compute_lookback_params(lookback, modes)
        assert result == [(5, 6), (10, 11)]
        assert max_lb == 10
        assert max_len == 11

    def test_expanding_window(self):
        lookback = [3]
        modes = ["扩展窗口"]
        result, max_lb, max_len = compute_lookback_params(lookback, modes)
        assert result == [(3, np.inf)]
        assert max_lb == 3
        assert max_len == np.inf

    def test_mixed_modes(self):
        lookback = [5, 3]
        modes = ["滚动窗口", "扩展窗口"]
        result, max_lb, max_len = compute_lookback_params(lookback, modes)
        assert result[0] == (5, 6)
        assert result[1] == (3, np.inf)
        assert max_lb == 5
        assert max_len == np.inf

    def test_single_descriptor(self):
        lookback = [20]
        modes = ["滚动窗口"]
        result, max_lb, max_len = compute_lookback_params(lookback, modes)
        assert len(result) == 1
        assert max_lb == 20
        assert max_len == 21


class TestExtendDtRuler:
    def test_sufficient_history(self):
        dt_ruler = list(range(10))
        dts = [5]
        result = extend_dt_ruler(dt_ruler, dts, max_lookback=3)
        assert result == list(range(2, 10))

    def test_insufficient_history(self):
        dt_ruler = list(range(5))
        dts = [2]
        result = extend_dt_ruler(dt_ruler, dts, max_lookback=5)
        assert result[0] is None
        assert dt_ruler[0] in result

    def test_no_lookback(self):
        dt_ruler = list(range(10))
        dts = [0]
        result = extend_dt_ruler(dt_ruler, dts, max_lookback=0)
        assert result == list(range(10))

    def test_exact_boundary(self):
        dt_ruler = list(range(10))
        dts = [3]
        result = extend_dt_ruler(dt_ruler, dts, max_lookback=3)
        assert len(result) == 10

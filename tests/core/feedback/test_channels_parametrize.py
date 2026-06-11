"""channels.py 100% 参数覆盖测试 (~30 tests, 每个函数每个参数)。

使用 @pytest.mark.parametrize 遍历每个函数的所有参数组合 + 异常。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback import (
    collect_code,
    collect_execution,
    collect_shape,
    collect_value,
)


# ============================================================================
# 1. collect_execution 6 参数 (2 tests)
# ============================================================================

class TestCollectExecutionParams:
    @pytest.mark.parametrize("exit_code,expected_passed", [
        (0, True), (1, False), (-1, False), (127, False), (2, False),
    ])
    def test_exit_code_variants(self, exit_code, expected_passed):
        r = collect_execution("out", "err", exit_code)
        assert r.passed is expected_passed
        assert r.metadata["exit_code"] == exit_code

    @pytest.mark.parametrize("stdout,stderr", [
        ("", ""),
        ("x" * 100, "y" * 100),
        ("中文 output\n", "stderr 错误\n"),
        (None, None),
    ])
    def test_stdout_stderr_variants(self, stdout, stderr):
        r = collect_execution(stdout, stderr, 0)
        assert r.passed is True
        # None 走 str() → "None"
        assert r.detail is not None

    @pytest.mark.parametrize("exit_code,should_truncate", [
        (0, False),  # passed, detail 含 "exit=0\nstdout: ...\nstderr: ..."
        (1, False),  # 也截断
    ])
    def test_long_stderr_truncated(self, exit_code, should_truncate):
        long = "x" * 2000
        r = collect_execution(long, long, exit_code)
        # stdout/stderr 截断至 500 字符, total < 1100
        assert len(r.detail) < 1500


# ============================================================================
# 2. collect_shape 4 参数组合 (10 tests)
# ============================================================================

class TestCollectShapeParams:
    @pytest.mark.parametrize("actual,expected,expected_passed", [
        ((3, 5), (3, 5), True),
        ((3, 4), (3, 5), False),
        ((4, 5), (3, 5), False),
        ((3, 5, 2), (3, 5), False),  # 多维
        ((3,), (3, 5), False),  # 少维
        ((), (3, 5), False),  # 空
        ((3, 5), (3, 5, 2), False),  # expected 多维
    ])
    def test_shape_variants(self, actual, expected, expected_passed):
        r = collect_shape(actual, expected)
        assert r.passed is expected_passed

    @pytest.mark.parametrize("container_type", [list, tuple])
    def test_list_vs_tuple_mix(self, container_type):
        """list 与 tuple 互相比较也能 pass。"""
        r1 = collect_shape([3, 5], container_type([3, 5]))
        assert r1.passed is True

    def test_zero_dim(self):
        r = collect_shape((0, 0), (0, 0))
        assert r.passed is True


# ============================================================================
# 3. collect_code 7 参数 (8 tests)
# ============================================================================

class TestCollectCodeParams:
    @pytest.mark.parametrize("expr,should_pass,desc_match", [
        ("close", True, "OK"),
        ("close - open", True, "OK"),
        ("close + open", True, "OK"),
    ])
    def test_valid_passes(self, expr, should_pass, desc_match):
        r = collect_code(expr)
        assert r.passed is should_pass
        assert desc_match in r.detail

    @pytest.mark.parametrize("expr,err_keyword", [
        ("close +", "语法错误"),
        ("(((", "语法错误"),
        ("@invalid", "语法错误"),  # @ 不是合法字符
    ])
    def test_invalid_syntax(self, expr, err_keyword):
        r = collect_code(expr)
        if "语法错误" in err_keyword:
            assert r.passed is False
            assert err_keyword in r.detail

    @pytest.mark.parametrize("threshold,should_pass", [
        (10, True),  # "close" 长度 5 < 10
        (3, False),  # 长度 5 > 3
        (5, True),   # 长度 5 不 > 5
        (6, True),   # 长度 5 < 6
        (4, False),  # 长度 5 > 4
    ])
    def test_symbol_length_threshold(self, threshold, should_pass):
        r = collect_code("close", symbol_length_threshold=threshold)
        assert r.passed is should_pass

    @pytest.mark.parametrize("features_threshold,should_pass", [
        (5, True),   # "close" = 1 feature < 5
        (0, False),  # 1 > 0
        (1, True),   # 1 == 1 OK
        (2, True),   # 1 < 2
    ])
    def test_base_features_threshold(self, features_threshold, should_pass):
        r = collect_code("close", base_features_threshold=features_threshold)
        assert r.passed is should_pass

    @pytest.mark.parametrize("free_ratio,should_pass", [
        (0.5, True),   # 0 free
        (0.0, True),   # 0%
        (1.0, True),   # 全 free 但 0
        (0.4, True),
    ])
    def test_free_args_ratio_threshold(self, free_ratio, should_pass):
        r = collect_code("close", free_args_ratio_threshold=free_ratio)
        assert r.passed is should_pass

    @pytest.mark.parametrize("n_features", [1, 3, 5, 10])
    def test_feature_counting(self, n_features):
        """n_features 控制: 同一表达式 (1 个 base feature) 在不同阈值下通过/失败。"""
        # "close" 1 feature
        if n_features < 1:
            r = collect_code("close", base_features_threshold=n_features)
            assert r.passed is False
        else:
            r = collect_code("close", base_features_threshold=n_features)
            assert r.passed is True


# ============================================================================
# 4. collect_value 5 参数 (10 tests)
# ============================================================================

class TestCollectValueParams:
    @pytest.mark.parametrize("values,should_pass,nan_pct_expected", [
        ([1.0, 2.0, 3.0], True, 0.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan], True, 0.222),  # 2/9 ≈ 0.222
        ([np.nan] * 5, False, 1.0),  # 早返回
        ([np.inf] * 5, False, 0.0),
        ([1.0, 2.0, -np.inf], False, 0.0),
    ])
    def test_series_variants(self, values, should_pass, nan_pct_expected):
        r = collect_value(pd.Series(values))
        assert r.passed is should_pass
        if nan_pct_expected == 1.0 and len(set(values)) == 1 and values[0] != values[0]:
            assert r.metadata == {}
        else:
            assert abs(r.metadata["nan_pct"] - nan_pct_expected) < 0.05

    @pytest.mark.parametrize("threshold,should_pass_with_30pct_nan", [
        (0.3, True),   # 0.3 > 0.3 False
        (0.29, False), # 0.3 > 0.29 True
        (0.4, True),
        (0.5, True),
        (0.2, False),
    ])
    def test_nan_threshold_variants(self, threshold, should_pass_with_30pct_nan):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan, np.nan])
        r = collect_value(s, nan_threshold=threshold)
        assert r.passed is should_pass_with_30pct_nan

    @pytest.mark.parametrize("std_threshold,should_pass_with_2_5_std1", [
        (0.5, True),    # std≈1.58 > 0.5
        (1.0, True),    # 1.58 > 1.0
        (2.0, False),   # 1.58 < 2.0
    ])
    def test_std_threshold(self, std_threshold, should_pass_with_2_5_std1):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # std ≈ 1.58
        r = collect_value(s, std_threshold=std_threshold)
        assert r.passed is should_pass_with_2_5_std1

    @pytest.mark.parametrize("input_data", [
        pd.Series([1.0, 2.0, 3.0]),  # Series
        [1.0, 2.0, 3.0],  # list
        np.array([1.0, 2.0, 3.0]),  # array
        (1.0, 2.0, 3.0),  # tuple
    ])
    def test_input_types(self, input_data):
        """接受各种可迭代。"""
        r = collect_value(input_data)
        assert r.passed is True

    def test_zero_length_early_return(self):
        """空 series → 早返回, metadata={}。"""
        r = collect_value(pd.Series([], dtype=float))
        assert r.passed is False
        assert r.metadata == {}

    @pytest.mark.parametrize("data,desc_match", [
        (pd.Series([1.0, np.inf, 3.0]), "Inf"),
        (pd.Series([1.0, -np.inf, 3.0]), "Inf"),
    ])
    def test_inf_violation(self, data, desc_match):
        r = collect_value(data)
        assert r.passed is False
        assert desc_match in r.detail


# ============================================================================
# 5. 边界组合 (5 tests)
# ============================================================================

class TestCombinations:
    @pytest.mark.parametrize("expr,should_pass", [
        ("close", True),
        ("close + close + close + close", True),  # 长度 28
        ("a + b + c + d + e + f + g", False),  # 太多 free args
    ])
    def test_long_acceptable(self, expr, should_pass):
        r = collect_code(expr, symbol_length_threshold=200,
                         base_features_threshold=5,
                         free_args_ratio_threshold=0.5)
        assert r.passed is should_pass

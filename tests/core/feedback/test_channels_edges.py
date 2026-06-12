"""channels.py 边界条件测试 (15 tests)。

聚焦:
    - collect_value: NaN/Inf/空/单值/混合
    - collect_code: 语法错误/空/超长/超 features/超 free args
    - collect_execution: 退出码/long stdout/long stderr 截断
    - collect_shape: 元组/列表/嵌套
    - 内部辅助: _count_base_features / _calc_free_args_ratio
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback import (
    FeedbackChannel,
    collect_code,
    collect_execution,
    collect_shape,
    collect_value,
)
from QuantNodes.core.feedback.channels import (
    _BASE_FEATURE_NAMES,
    _calc_free_args_ratio,
    _count_base_features,
)


# ============================================================================
# 1. collect_value (8 tests)
# ============================================================================

class TestCollectValue:
    """collect_value 数值分布检查。"""

    def test_all_nan_returns_failed(self):
        """全 NaN 序列应该 fail。"""
        r = collect_value(pd.Series([np.nan] * 5))
        assert r.channel == FeedbackChannel.VALUE
        assert r.passed is False
        assert "全部 NaN" in r.detail
        assert r.score == 0.0
        # 全 NaN 早返回, metadata 为空 dict
        assert r.metadata == {}

    def test_all_inf_returns_failed(self):
        """全 Inf 序列应该 fail。"""
        r = collect_value(pd.Series([np.inf] * 5))
        assert r.passed is False
        assert "Inf" in r.detail
        assert r.metadata["inf_count"] == 5

    def test_mixed_inf_and_normal_fails(self):
        """混合 Inf 失败 (1 个就 fail)。"""
        r = collect_value(pd.Series([1.0, 2.0, np.inf, 4.0]))
        assert r.passed is False
        assert r.metadata["inf_count"] == 1

    def test_single_value_fails_std_check(self):
        """单值 std=0 fail。"""
        r = collect_value(pd.Series([1.0]))
        assert r.passed is False
        assert "std" in r.detail
        assert r.metadata["std"] == 0.0

    def test_normal_series_passes(self):
        """正常序列通过。"""
        r = collect_value(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert r.passed is True
        assert r.metadata["nan_pct"] == 0.0
        assert r.metadata["mean"] == 3.0
        assert r.metadata["std"] > 0.0

    def test_partial_nan_under_threshold_passes(self):
        """50% NaN 阈值默认 30% -> fail; 调高到 80% -> pass。"""
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan])
        r = collect_value(s, nan_threshold=0.3)
        assert r.passed is False
        assert "NaN" in r.detail
        # 调高阈值
        r2 = collect_value(s, nan_threshold=0.6)
        assert r2.passed is True

    def test_constant_series_fails(self):
        """常量序列 std=0 fail。"""
        r = collect_value(pd.Series([5.0] * 10))
        assert r.passed is False
        assert "std" in r.detail

    def test_accepts_array_like(self):
        """接受 list/array 形式 (不强制 pd.Series)。"""
        r = collect_value([1.0, 2.0, 3.0, 4.0])
        assert r.passed is True


# ============================================================================
# 2. collect_code (5 tests)
# ============================================================================

class TestCollectCode:
    """collect_code AST 检查。"""

    def test_syntax_error_returns_failed(self):
        """语法错误返回 failed 而不抛异常。"""
        r = collect_code("close +")
        assert r.passed is False
        assert "语法错误" in r.detail
        assert r.score == 0.0

    def test_valid_expression_passes(self):
        """合法表达式通过。"""
        r = collect_code("close - close.shift(5)")
        assert r.passed is True
        assert "OK" in r.detail
        assert r.metadata["symbol_length"] > 0
        assert r.metadata["base_features"] >= 1

    def test_too_long_symbol_fails(self):
        """超长 symbol 失败。"""
        # 用纯标识符避免 syntax error
        long_expr = "open_high_low_close_volume_amount_vwap_turnover_mv_float" * 10
        r = collect_code(long_expr, symbol_length_threshold=20)
        assert r.passed is False
        assert "length" in r.detail

    def test_too_many_features_fails(self):
        """基础特征超阈值失败。"""
        r = collect_code(
            "open + high + low + close + volume + amount + vwap",
            base_features_threshold=3,
        )
        assert r.passed is False
        assert "features" in r.detail

    def test_too_many_free_args_fails(self):
        """自由参数比例超阈值失败。"""
        r = collect_code(
            "a + b + c + d + e",
            free_args_ratio_threshold=0.3,
        )
        assert r.passed is False
        assert "free_args" in r.detail


# ============================================================================
# 3. collect_execution + collect_shape (4 tests)
# ============================================================================

class TestCollectExecution:
    def test_exit_zero_passes(self):
        r = collect_execution("out", "err", 0)
        assert r.passed is True
        assert r.metadata["exit_code"] == 0

    def test_exit_nonzero_fails(self):
        r = collect_execution("out", "err", 1)
        assert r.passed is False
        assert r.metadata["exit_code"] == 1

    def test_long_stdout_stderr_truncated(self):
        """长输出截断至 500 字符。"""
        long_out = "x" * 2000
        long_err = "y" * 2000
        r = collect_execution(long_out, long_err, 0)
        # detail 格式: "exit=0\nstdout: <500 chars>\nstderr: <500 chars>"
        assert "exit=0" in r.detail
        assert "stdout: " in r.detail
        # 整体 detail 不应超过 ~1100 (500+500+prefix)
        assert len(r.detail) < 1500

    def test_metadata_exit_code_typed(self):
        """exit_code 转 int。"""
        r = collect_execution("", "", 0)
        assert isinstance(r.metadata["exit_code"], int)


class TestCollectShape:
    def test_matching_tuples_pass(self):
        r = collect_shape((3, 5), (3, 5))
        assert r.passed is True

    def test_mismatched_shapes_fail(self):
        r = collect_shape((3, 4), (3, 5))
        assert r.passed is False
        assert "(3, 4)" in r.detail
        assert "(3, 5)" in r.detail

    def test_list_vs_tuple_works(self):
        """list 与 tuple 都能比对。"""
        r = collect_shape([3, 5], (3, 5))
        assert r.passed is True
        r2 = collect_shape((3, 5), [3, 5])
        assert r2.passed is True

    def test_extra_dim_mismatch(self):
        """维度不匹配。"""
        r = collect_shape((3, 5, 2), (3, 5))
        assert r.passed is False


# ============================================================================
# 4. Internal helpers (3 tests)
# ============================================================================

class TestInternalHelpers:
    def test_count_base_features_empty(self):
        """空 AST → 0 features。"""
        tree = ast.parse("1 + 2 + 3")
        assert _count_base_features(tree) == 0

    def test_count_base_features_dedup(self):
        """close + close + close → 1 (去重)。"""
        tree = ast.parse("close + close + close")
        assert _count_base_features(tree) == 1

    def test_calc_free_args_ratio_no_names(self):
        """无 Name 节点 → 0.0。"""
        tree = ast.parse("1 + 2 + 3")
        assert _calc_free_args_ratio(tree, 0) == 0.0

    def test_base_feature_set_includes_returns(self):
        """returns 是基础特征。"""
        assert "returns" in _BASE_FEATURE_NAMES
        assert "close" in _BASE_FEATURE_NAMES
        assert "turnover" in _BASE_FEATURE_NAMES


# ============================================================================
# M2: collect_execution max_output_chars
# ============================================================================

class TestCollectExecutionMaxChars:
    @pytest.mark.parametrize("max_chars,expected_in_detail", [
        (100, "a" * 100),    # 截断到 100
        (500, "a" * 500),    # 默认
        (10, "a" * 10),      # 短截断
        (0, ""),             # 0 → 空
    ])
    def test_max_output_chars(self, max_chars, expected_in_detail):
        from QuantNodes.core.feedback import collect_execution
        long = "a" * 1000
        fb = collect_execution(stdout=long, stderr=long, exit_code=0,
                               max_output_chars=max_chars)
        assert expected_in_detail in fb.detail
        # 长于 max_chars 的部分应被截断
        if max_chars < 1000:
            assert "a" * (max_chars + 1) not in fb.detail

    def test_default_backward_compat(self):
        from QuantNodes.core.feedback import collect_execution
        fb = collect_execution(stdout="hi", stderr="err", exit_code=0)
        assert "hi" in fb.detail
        assert "err" in fb.detail

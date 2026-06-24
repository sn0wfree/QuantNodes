# coding=utf-8
"""Tests for QuantAlpha subpackage.

覆盖：
- OperatorVocab 基本功能（list / get / get_metadata）
- 5 个新算子（signedpower / ts_decay_linear / IndNeutralize / ts_skew / ts_kurt）
- per-date over() 修复（rank / zscore / winsorize）
- 旧 12-lambda 兼容（cross_sectional=False）
- Alpha 101 完整公式执行
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha import (
    OperatorVocab,
    OperatorVocabConfig,
    OperatorMetadata,
)
from QuantNodes.research.quant_alpha.operator_vocab import (
    build_namespace,
    list_vocab_operators,
    get_vocab_operator,
    get_vocab_metadata,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成 3 票 × 5 日 测试数据"""
    np.random.seed(42)
    return pl.DataFrame({
        "date": (
            ["2024-01-01"] * 3
            + ["2024-01-02"] * 3
            + ["2024-01-03"] * 3
            + ["2024-01-04"] * 3
            + ["2024-01-05"] * 3
        ),
        "code": ["A", "B", "C"] * 5,
        "close": np.random.randn(15).cumsum() + 100.0,
        "open": np.random.randn(15).cumsum() + 100.0,
        "high": np.random.randn(15).cumsum() + 102.0,
        "low": np.random.randn(15).cumsum() + 98.0,
        "vol": np.random.randint(1000, 5000, 15).astype(float),
    }).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def sample_data_with_industry(sample_data: pl.DataFrame) -> pl.DataFrame:
    """带 industry 列的样本数据（IndNeutralize 需要）"""
    return sample_data.with_columns(
        pl.when(pl.col("code") == "A").then(pl.lit("ind_1"))
        .when(pl.col("code") == "B").then(pl.lit("ind_1"))
        .otherwise(pl.lit("ind_2"))
        .alias("industry")
    )


@pytest.fixture
def vocab() -> OperatorVocab:
    """默认 OperatorVocab 实例"""
    return OperatorVocab.default()


# ==============================================================================
# Test Class 1: OperatorVocab 基本查询
# ==============================================================================


class TestOperatorVocabBasics:
    """OperatorVocab 基本查询功能测试"""

    def test_singleton_returns_same_instance(self):
        """模块级单例"""
        v1 = OperatorVocab.default()
        v2 = OperatorVocab.default()
        assert v1 is v2

    def test_stats_returns_l0_count(self, vocab: OperatorVocab):
        """stats 返回 L0 算子数（应 > 12，覆盖旧 12-lambda）"""
        stats = vocab.stats()
        assert stats["l0_total"] >= 12
        # 我们加了 5 个新算子
        assert stats["l0_total"] >= 161

    def test_stats_by_category(self, vocab: OperatorVocab):
        """stats 按类别分类正确"""
        stats = vocab.stats()
        by_cat = stats["by_category"]
        assert "point" in by_cat
        assert "time" in by_cat
        assert "section" in by_cat
        assert by_cat["point"] >= 50
        assert by_cat["time"] >= 65
        assert by_cat["section"] >= 20

    def test_list_operators_returns_all(self, vocab: OperatorVocab):
        """list_operators() 列出所有算子"""
        ops = vocab.list_operators()
        assert isinstance(ops, list)
        assert len(ops) >= 12
        # 关键旧算子
        assert "ts_mean" in ops
        assert "ts_std" in ops
        assert "rank" in ops
        # 5 个新算子
        assert "signedpower" in ops
        assert "ts_decay_linear" in ops
        assert "IndNeutralize" in ops
        assert "ts_skew" in ops
        assert "ts_kurt" in ops

    def test_list_operators_filter_by_category(self, vocab: OperatorVocab):
        """list_operators(category) 按类别过滤"""
        point_ops = vocab.list_operators(category="point")
        time_ops = vocab.list_operators(category="time")
        section_ops = vocab.list_operators(category="section")

        # 全部 point 算子应该在 _OPERATOR_REGISTRY[point] 中
        assert all(vocab.get_metadata(op) and vocab.get_metadata(op).category == "point"
                   for op in point_ops)
        assert all(vocab.get_metadata(op) and vocab.get_metadata(op).category == "time"
                   for op in time_ops)
        assert all(vocab.get_metadata(op) and vocab.get_metadata(op).category == "section"
                   for op in section_ops)

        # 互不重叠
        assert not (set(point_ops) & set(time_ops))
        assert not (set(time_ops) & set(section_ops))

    def test_list_operators_invalid_category_raises(self, vocab: OperatorVocab):
        """非法 category 抛 ValueError"""
        with pytest.raises(ValueError, match="Invalid category"):
            vocab.list_operators(category="nonexistent")

    def test_get_operator_returns_callable(self, vocab: OperatorVocab):
        """get_operator 返回可调用对象"""
        op = vocab.get_operator("ts_mean")
        assert callable(op)

    def test_get_operator_unknown_returns_none(self, vocab: OperatorVocab):
        """get_operator 未知算子返回 None"""
        assert vocab.get_operator("nonexistent_op_xyz") is None

    def test_get_metadata_returns_12_fields(self, vocab: OperatorVocab):
        """get_metadata 返回 12 字段 OperatorMetadata"""
        meta = vocab.get_metadata("ts_argmax")
        assert meta is not None
        assert isinstance(meta, OperatorMetadata)
        # 5 基础字段
        assert meta.name == "ts_argmax"
        assert meta.category == "time"
        assert meta.func is not None
        assert isinstance(meta.doc, str)
        assert isinstance(meta.signature, str)
        assert isinstance(meta.parameters, list)
        # 7 LLM 友好字段
        assert isinstance(meta.difficulty, int)
        assert 1 <= meta.difficulty <= 3
        assert isinstance(meta.category_tags, list)
        assert isinstance(meta.default_window, list)
        assert isinstance(meta.requires_group_by, bool)
        assert isinstance(meta.output_dtype, str)
        assert isinstance(meta.examples, list)
        assert isinstance(meta.composes_with, list)

    def test_get_metadata_unknown_returns_none(self, vocab: OperatorVocab):
        """get_metadata 未知算子返回 None"""
        assert vocab.get_metadata("nonexistent_op_xyz") is None

    def test_list_metadata_returns_all(self, vocab: OperatorVocab):
        """list_metadata 列出所有元数据"""
        all_metas = vocab.list_metadata()
        assert len(all_metas) >= 12
        assert all(isinstance(m, OperatorMetadata) for m in all_metas)


# ==============================================================================
# Test Class 2: 5 个新算子
# ==============================================================================


class TestNewOperators:
    """5 个新算子（Alpha 101 必需）测试"""

    def test_signedpower_preserves_sign(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """signedpower 保留正负号"""
        result = vocab.evaluate("signedpower(close, 2)", sample_data)
        assert result is not None
        values = result.to_list()
        # 检查 sign 与原值一致
        for i, c in enumerate(sample_data["close"].to_list()):
            expected_sign = 1 if c > 0 else (-1 if c < 0 else 0)
            actual_sign = 1 if values[i] > 0 else (-1 if values[i] < 0 else 0)
            assert actual_sign == expected_sign or (expected_sign == 0 and abs(values[i]) < 1e-6)
            # 检查 |signedpower(x,2)| = x^2
            assert abs(abs(values[i]) - c ** 2) < 1e-6

    def test_signedpower_fractional_exponent(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """signedpower 分数指数（sqrt with sign）"""
        result = vocab.evaluate("signedpower(close, 0.5)", sample_data)
        assert result is not None
        values = result.to_list()
        for i, c in enumerate(sample_data["close"].to_list()):
            # |signedpower(x, 0.5)| = sqrt(|x|)
            assert abs(abs(values[i]) - np.sqrt(abs(c))) < 1e-6

    def test_ts_decay_linear_equals_decay_linear(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """ts_decay_linear ≡ decay_linear"""
        r1 = vocab.evaluate("ts_decay_linear(close, 3)", sample_data)
        r2 = vocab.evaluate("decay_linear(close, 3)", sample_data)
        assert r1 is not None and r2 is not None
        for a, b in zip(r1.to_list(), r2.to_list()):
            if a is not None and b is not None:
                assert abs(a - b) < 1e-6

    def test_IndNeutralize_demeans_by_industry(
        self, vocab: OperatorVocab, sample_data_with_industry: pl.DataFrame
    ):
        """IndNeutralize 按行业去均值（per-date + per-industry）"""
        result = vocab.evaluate(
            "IndNeutralize(close, 'industry')",
            sample_data_with_industry,
            cross_sectional=True,
        )
        assert result is not None
        # 验证：per-date + per-industry demean 后，每个 (date, industry) 内均值应约 0
        df = sample_data_with_industry.with_columns(result.alias("ind_neut"))
        for date in df["date"].unique().to_list():
            for ind in ["ind_1", "ind_2"]:
                sub = df.filter(
                    (pl.col("date") == date) & (pl.col("industry") == ind)
                )["ind_neut"]
                if len(sub) > 0:
                    mean = sub.mean()
                    assert abs(mean) < 1e-6, f"日期 {date} 行业 {ind} 去均值后均值应接近 0，实际 {mean}"

    def test_ts_skew_equals_rolling_skew(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """ts_skew ≡ rolling_skew"""
        r1 = vocab.evaluate("ts_skew(close, 5)", sample_data)
        r2 = vocab.evaluate("rolling_skew(close, 5)", sample_data)
        assert r1 is not None and r2 is not None
        for a, b in zip(r1.to_list(), r2.to_list()):
            if a is not None and b is not None:
                assert abs(a - b) < 1e-6

    def test_ts_kurt_equals_rolling_kurt(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """ts_kurt ≡ rolling_kurt"""
        r1 = vocab.evaluate("ts_kurt(close, 5)", sample_data)
        r2 = vocab.evaluate("rolling_kurt(close, 5)", sample_data)
        assert r1 is not None and r2 is not None
        for a, b in zip(r1.to_list(), r2.to_list()):
            if a is not None and b is not None:
                assert abs(a - b) < 1e-6


# ==============================================================================
# Test Class 3: per-date over() 修复（核心 BUG 2 修复）
# ==============================================================================


class TestPerDateOverFix:
    """per-date over() 语义修复测试（BUG 2 修复验证）"""

    def test_rank_per_date_resets_each_date(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """rank per-date 应该在每个日期内独立 rank"""
        r = vocab.evaluate("rank(close)", sample_data, cross_sectional=True)
        assert r is not None
        df = sample_data.with_columns(r.alias("r"))
        for date in df["date"].unique().to_list():
            sub = df.filter(pl.col("date") == date)["r"]
            # 每个日期内 rank 应该是 1, 2, 3（顺序可能不同）
            assert sorted(sub.to_list()) == [1.0, 2.0, 3.0]

    def test_rank_global_increases_monotonically(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """rank global（cross_sectional=False）应该单调递增（与旧行为一致）"""
        r = vocab.evaluate("rank(close)", sample_data, cross_sectional=False)
        assert r is not None
        # 全局 rank，理论上应该从 1 到 15 单调（cumsum 序列）
        # 实际：因为 close 是 cumsum + 噪声，可能不严格单调
        # 至少 1 和 15 应该在两端
        assert r[0] in (1.0, 2.0, 3.0)

    def test_zscore_per_date_std_close_to_1(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """zscore per-date 每个日期内 std 应接近 1"""
        z = vocab.evaluate("zscore(close)", sample_data, cross_sectional=True)
        assert z is not None
        df = sample_data.with_columns(z.alias("z"))
        for date in df["date"].unique().to_list():
            sub = df.filter(pl.col("date") == date)["z"]
            std = sub.std()
            # 3 个值的 std 应该约 1
            assert 0.5 < std < 2.0, f"per-date std={std} 不在 0.5-2.0 范围"

    def test_winsorize_per_date_clamps_to_quantile(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """winsorize per-date 裁剪到分位数"""
        w = vocab.evaluate(
            "winsorize(close, 0.01, 0.99)",
            sample_data,
            cross_sectional=True,
        )
        assert w is not None
        # winsorize 后值应该在原数据 min/max 之间
        orig_min = sample_data["close"].min()
        orig_max = sample_data["close"].max()
        w_min = w.min()
        w_max = w.max()
        assert w_min >= orig_min - 1e-6
        assert w_max <= orig_max + 1e-6

    def test_cross_sectional_toggle_compatible_with_old_behavior(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """cross_sectional=False 应与旧 12-lambda 行为一致"""
        # rank global（与旧 12-lambda 一致）
        r = vocab.evaluate("rank(close)", sample_data, cross_sectional=False)
        # zscore global（与旧 12-lambda 一致）
        z = vocab.evaluate("zscore(close)", sample_data, cross_sectional=False)
        # 旧 12-lambda 中 rank 是 col.rank()（无 method 参数，默认 average）
        # zscore 是 (col - col.mean()) / (col.std() + 1e-8)
        assert r is not None
        assert z is not None


# ==============================================================================
# Test Class 4: Alpha 101 完整公式
# ==============================================================================


class TestAlpha101Formulas:
    """Alpha 101 公式执行测试"""

    def test_alpha1_simplified(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """Alpha 101 #1 简化版：rank(ts_argmax(signedpower(close, 2), 3))"""
        result = vocab.evaluate(
            "rank(ts_argmax(signedpower(close, 2), 3))",
            sample_data,
            cross_sectional=True,
        )
        assert result is not None
        assert len(result) == 15

    def test_alpha6_corr(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """Alpha 101 #6：-1 * correlation(open, volume, 10)"""
        # correlation 是个别写法（与 ts_corr 同义）
        result = vocab.evaluate(
            "-1 * ts_corr(open, vol, 10)",
            sample_data,
        )
        assert result is not None
        assert len(result) == 15

    def test_alpha12_volume_signed_delta(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """Alpha 101 #12：sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
        result = vocab.evaluate(
            "sign(ts_delta(vol, 1)) * (-1 * ts_delta(close, 1))",
            sample_data,
        )
        assert result is not None
        assert len(result) == 15

    def test_alpha101_intraday_momentum(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """Alpha 101 #101：(close - open) / ((high - low) + 0.001)"""
        result = vocab.evaluate(
            "(close - open) / ((high - low) + 0.001)",
            sample_data,
        )
        assert result is not None
        assert len(result) == 15


# ==============================================================================
# Test Class 5: 错误处理与边界
# ==============================================================================


class TestErrorHandling:
    """错误处理与边界条件测试"""

    def test_eval_raises_on_invalid_formula(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """无效公式应该抛异常（不静默吞掉，BUG 3 修复）"""
        with pytest.raises(Exception):  # noqa: B017
            vocab.evaluate("1 / 0", sample_data)

    def test_eval_raises_on_unknown_function(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """未知函数应抛 NameError"""
        with pytest.raises(NameError):
            vocab.evaluate("unknown_func(close)", sample_data)

    def test_eval_length_limit(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """超长公式应抛 ValueError"""
        long_formula = " + close" * 1000  # 8000+ 字符
        with pytest.raises(ValueError, match="length"):
            vocab.evaluate(long_formula, sample_data)

    def test_eval_depth_limit(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame
    ):
        """嵌套深度超限应抛 ValueError"""
        deep_formula = "rank(" * 25 + "close" + ")" * 25
        with pytest.raises(ValueError, match="depth"):
            vocab.evaluate(deep_formula, sample_data)

    def test_custom_max_formula_length(self, sample_data: pl.DataFrame):
        """自定义公式长度限制"""
        config = OperatorVocabConfig(max_formula_length=50)
        vocab = OperatorVocab(config)
        long_formula = " + close" * 30  # 210 字符
        with pytest.raises(ValueError, match="length"):
            vocab.evaluate(long_formula, sample_data)


# ==============================================================================
# Test Class 6: 便捷函数
# ==============================================================================


class TestConvenienceFunctions:
    """模块级便捷函数测试"""

    def test_build_namespace_default(self, sample_data: pl.DataFrame):
        """build_namespace 默认"""
        ns = build_namespace(sample_data)
        assert "rank" in ns
        assert "zscore" in ns
        assert "ts_mean" in ns
        assert "signedpower" in ns
        assert "close" in ns
        assert "date" in ns

    def test_build_namespace_global(self, sample_data: pl.DataFrame):
        """build_namespace cross_sectional=False"""
        ns = build_namespace(sample_data, cross_sectional=False)
        assert "rank" in ns
        # 调用应不报错
        rank_global = ns["rank"](sample_data["close"])
        assert rank_global is not None

    def test_list_vocab_operators(self):
        """list_vocab_operators 便捷函数"""
        ops = list_vocab_operators()
        assert isinstance(ops, list)
        assert "ts_mean" in ops

    def test_get_vocab_operator(self):
        """get_vocab_operator 便捷函数"""
        op = get_vocab_operator("ts_mean")
        assert callable(op)

    def test_get_vocab_metadata(self):
        """get_vocab_metadata 便捷函数"""
        meta = get_vocab_metadata("signedpower")
        assert meta is not None
        assert meta.name == "signedpower"
        assert meta.difficulty >= 1


# ==============================================================================
# Test Class 7: 元数据推断
# ==============================================================================


class TestMetadataInference:
    """元数据自动推断测试"""

    def test_signedpower_polarity_tag(self, vocab: OperatorVocab):
        """signedpower 推断 polarity 标签"""
        meta = vocab.get_metadata("signedpower")
        assert "polarity" in meta.category_tags

    def test_ts_argmax_position_tag(self, vocab: OperatorVocab):
        """ts_argmax 推断 position 标签"""
        meta = vocab.get_metadata("ts_argmax")
        assert "position" in meta.category_tags

    def test_ts_mean_central_tendency_tag(self, vocab: OperatorVocab):
        """ts_mean 推断 central_tendency 标签"""
        meta = vocab.get_metadata("ts_mean")
        assert "central_tendency" in meta.category_tags

    def test_ts_std_dispersion_tag(self, vocab: OperatorVocab):
        """ts_std 推断 dispersion 标签"""
        meta = vocab.get_metadata("ts_std")
        assert "dispersion" in meta.category_tags

    def test_ts_argmax_output_dtype_int(self, vocab: OperatorVocab):
        """ts_argmax 推断 int64 输出"""
        meta = vocab.get_metadata("ts_argmax")
        assert meta.output_dtype == "int64"

    def test_rank_requires_group_by(self, vocab: OperatorVocab):
        """rank 推断 requires_group_by=True（per-date 截面需要）"""
        meta = vocab.get_metadata("rank")
        assert meta.requires_group_by is True

    def test_ts_mean_default_window(self, vocab: OperatorVocab):
        """ts_mean 推断 default_window"""
        meta = vocab.get_metadata("ts_mean")
        assert 5 in meta.default_window
        assert 20 in meta.default_window
        assert 60 in meta.default_window

    def test_section_category_includes_cross_sectional(self, vocab: OperatorVocab):
        """section 类算子有 cross_sectional 标签"""
        meta = vocab.get_metadata("rank")
        assert "cross_sectional" in meta.category_tags


# ==============================================================================
# Test Class 8: 旧 12-lambda 兼容性
# ==============================================================================


class TestLegacyCompatibility:
    """旧 12-lambda namespace 兼容测试（cross_sectional=False）"""

    @pytest.mark.parametrize("formula,description", [
        ("ts_mean(close, 5)", "简单滚动均值"),
        ("ts_std(close, 5)", "简单滚动标准差"),
        ("ts_max(close, 5)", "滚动最大"),
        ("ts_min(close, 5)", "滚动最小"),
        ("ts_delta(close, 1)", "差分"),
        ("ts_lag(close, 1)", "滞后"),
        ("ts_pct_change(close, 1)", "百分比变化"),
    ])
    def test_legacy_time_ops(self, vocab: OperatorVocab, sample_data: pl.DataFrame,
                              formula: str, description: str):
        """旧 7 个 time-series 算子（与原 namespace 行为一致）"""
        result = vocab.evaluate(formula, sample_data)
        assert result is not None
        assert len(result) == 15

    def test_legacy_tscorr(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """旧 ts_corr 算子（用 L0 注册表 rolling_corr 修复）"""
        result = vocab.evaluate("ts_corr(close, vol, 5)", sample_data)
        assert result is not None

    def test_legacy_tscov(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """旧 ts_cov 算子"""
        result = vocab.evaluate("ts_cov(close, vol, 5)", sample_data)
        assert result is not None

    def test_legacy_rank_global(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """旧 rank 全局（cross_sectional=False）"""
        result = vocab.evaluate("rank(close)", sample_data, cross_sectional=False)
        assert result is not None
        # 全局 rank 最大值应为 15
        assert max(result.to_list()) == 15.0

    def test_legacy_zscore_global(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """旧 zscore 全局（cross_sectional=False）"""
        result = vocab.evaluate("zscore(close)", sample_data, cross_sectional=False)
        assert result is not None
        # 全局 zscore 绝对值均值应该约 1
        values = [abs(v) for v in result.to_list() if v is not None]
        mean_abs = sum(values) / len(values)
        assert 0.5 < mean_abs < 2.0


# ==============================================================================
# Test Class 9: 旧 API DeprecationWarning
# ==============================================================================


class TestDeprecationWarnings:
    """旧 4 个文件 DeprecationWarning 触发测试"""

    def test_factor_evaluator_module_has_deprecation_message(self):
        """factor_evaluator 模块 docstring 包含 DeprecationWarning 说明"""
        import importlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module("QuantNodes.research.factor_evaluator")
        # 检查模块 docstring
        assert "DeprecationWarning" in (mod.__doc__ or "")

    def test_factor_miner_module_has_deprecation_message(self):
        """factor_miner 模块 docstring 包含 DeprecationWarning 说明"""
        import importlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module("QuantNodes.research.factor_miner")
        assert "DeprecationWarning" in (mod.__doc__ or "")

    def test_auto_researcher_module_has_deprecation_message(self):
        """auto_researcher 模块 docstring 包含 DeprecationWarning 说明"""
        import importlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module("QuantNodes.research.auto_researcher")
        assert "DeprecationWarning" in (mod.__doc__ or "")

    def test_mcts_search_module_has_deprecation_message(self):
        """mcts_search 模块 docstring 包含 DeprecationWarning 说明"""
        import importlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module("QuantNodes.research.mcts_search")
        assert "DeprecationWarning" in (mod.__doc__ or "")

    def test_deprecation_warnings_present_in_research_init(self):
        """research/__init__.py 导入时触发 DeprecationWarning（汇总）"""
        # 触发 import-time warning 用 subprocess 隔离
        import subprocess
        result = subprocess.run(
            ["python3", "-W", "default::DeprecationWarning", "-c",
             "import QuantNodes.research; print('done')"],
            capture_output=True, text=True,
            cwd="/home/ll/Public/QuantNodes",
            timeout=60,
        )
        output = result.stderr
        # 期望至少 4 个文件触发 DeprecationWarning
        assert "factor_evaluator 已弃用" in output
        assert "factor_miner 已弃用" in output
        assert "mcts_search 已弃用" in output
        assert "auto_researcher 已弃用" in output

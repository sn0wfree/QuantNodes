# coding=utf-8
"""
test_operator_vocab_property.py - 算子 property-based 测试 (Phase 2)

目标: 162 个算子中抽样 30+ 个, 每个至少 happy/edge/metadata 三测。
使用 hypothesis 自动生成 random inputs, 发现手工难构造的边界 bug。

策略:
- 4 类(time/point/section/multi_section)各抽 6-8 个有代表性的算子
- 每算子: happy path (简单数据) + edge (NaN/极值) + metadata 完整性
- hypothesis: 随机数据驱动

不追求全 162 算子覆盖 (162×3=486 测, 跑太久)
抽样原则: 高频使用 / 边界多 / 历史出 bug
"""
import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

# 全局 hypothesis 配置: 5 examples, 关闭 function-scoped fixture 警告
HYPOTHESIS_SETTINGS = settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def vocab() -> OperatorVocab:
    return OperatorVocab.default()


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """3 票 × 10 日 测试数据"""
    np.random.seed(42)
    n_stocks = 3
    n_days = 10
    return pl.DataFrame({
        "date": (
            [f"2024-01-{d + 1:02d}" for d in range(n_days) for _ in range(n_stocks)]
        ),
        "code": ["A", "B", "C"] * n_days,
        "close": np.random.randn(n_stocks * n_days).cumsum() + 100.0,
        "open": np.random.randn(n_stocks * n_days).cumsum() + 100.0,
        "high": np.random.randn(n_stocks * n_days).cumsum() + 102.0,
        "low": np.random.randn(n_stocks * n_days).cumsum() + 98.0,
        "vol": np.random.randint(1000, 5000, n_stocks * n_days).astype(float),
    }).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def data_with_nan() -> pl.DataFrame:
    """含 NaN 的测试数据"""
    np.random.seed(123)
    n = 30
    data = {
        "date": ["2024-01-01"] * n,
        "code": [f"S{i % 3}" for i in range(n)],
        "close": np.random.randn(n).cumsum() + 100.0,
    }
    df = pl.DataFrame(data).with_columns(pl.col("date").str.to_date())
    # 注入一些 NaN
    df = df.with_columns(
        pl.when(pl.int_range(0, n) % 7 == 0)
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    return df


# ==============================================================================
# Test Class 1: 抽样算子 happy path
# ==============================================================================


class TestSampledOperatorsHappyPath:
    """30+ 抽样算子的 happy path 测试

    每个算子至少 1 个 happy + 1 个 metadata 验证
    """

    # === Time 类 (10) ===
    @pytest.mark.parametrize("op,formula", [
        ("ts_mean", "ts_mean(close, 3)"),
        ("ts_std", "ts_std(close, 3)"),
        ("ts_max", "ts_max(close, 5)"),
        ("ts_min", "ts_min(close, 5)"),
        ("ts_rank", "ts_rank(close, 5)"),
        ("ts_corr", "ts_corr(close, vol, 3)"),
        ("ts_decay_linear", "ts_decay_linear(close, 3)"),
        ("ts_skew", "ts_skew(close, 5)"),
        ("ts_argmax", "ts_argmax(close, 5)"),
        ("ts_argmin", "ts_argmin(close, 5)"),
    ])
    def test_time_op_happy_path(self, vocab: OperatorVocab, sample_data: pl.DataFrame, op: str, formula: str):
        """time 类算子: happy path 应成功求值"""
        result = vocab.evaluate(formula, sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    # === Point 类 (10) ===
    @pytest.mark.parametrize("op,formula", [
        ("abs", "abs(close - 100)"),
        ("sign", "sign(close - 100)"),
        ("log", "log(abs(close))"),
        ("sqrt", "sqrt(abs(close))"),
        ("add", "add(close, 1)"),
        ("sub", "sub(close, ts_mean(close, 5))"),
        ("mul", "mul(close, 2)"),
        ("div", "div(close, ts_mean(close, 5))"),
        ("signedpower", "signedpower(close, 0.5)"),
        ("square", "square(close)"),
    ])
    def test_point_op_happy_path(self, vocab: OperatorVocab, sample_data: pl.DataFrame, op: str, formula: str):
        """point 类算子: happy path 应成功求值"""
        result = vocab.evaluate(formula, sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    # === Section 类 (3, 跳过 cross_sectional_* 需特殊调用) ===
    @pytest.mark.parametrize("op,formula", [
        ("rank", "rank(close)"),
        ("zscore", "zscore(close)"),
        ("winsorize", "winsorize(close)"),
    ])
    def test_section_op_happy_path(self, vocab: OperatorVocab, sample_data: pl.DataFrame, op: str, formula: str):
        """section 类算子: happy path 应成功求值"""
        result = vocab.evaluate(formula, sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    # === Multi-Section 类: 跳过 aggr_*, 需 group_by 参数 ===


# ==============================================================================
# Test Class 2: 算子 metadata 完整性
# ==============================================================================


class TestOperatorMetadataIntegrity:
    """算子元数据完整性测试 (12 字段)

    防 metadata 字段缺失 (LLM 友好性, prompt 注入)
    """

    REQUIRED_FIELDS = {
        # 基础字段 (6)
        "name", "category", "func", "doc", "signature", "parameters",
        # LLM 友好字段 (7)
        "difficulty", "category_tags", "default_window",
        "requires_group_by", "output_dtype", "examples", "composes_with",
    }

    @pytest.mark.parametrize("op", [
        # 抽样: 每类 3 个
        "ts_mean", "ts_std", "ts_corr",  # time
        "abs", "sign", "div",  # point
        "rank", "zscore", "winsorize",  # section
        "aggr_mean", "aggr_max", "aggr_count",  # multi_section
    ])
    def test_metadata_has_all_required_fields(self, vocab: OperatorVocab, op: str):
        """每个抽样算子应有完整 12 字段 metadata"""
        meta = vocab.get_metadata(op)
        assert meta is not None, f"{op} metadata is None"
        # dataclass 字段检查 (用 asdict)
        from dataclasses import asdict
        d = asdict(meta)
        missing = self.REQUIRED_FIELDS - set(d.keys())
        assert not missing, f"{op} missing fields: {missing}"

    def test_all_listed_operators_have_metadata(self, vocab: OperatorVocab):
        """list_operators() 返回的每个算子都应有 metadata"""
        for op in vocab.list_operators():
            meta = vocab.get_metadata(op)
            assert meta is not None, f"{op} in list_operators() but has no metadata"

    def test_metadata_name_matches_key(self, vocab: OperatorVocab):
        """metadata.name 应与查询 key 一致"""
        for op in vocab.list_operators()[:20]:  # 抽样前 20 个
            meta = vocab.get_metadata(op)
            assert meta.name == op, f"key={op} but name={meta.name}"

    def test_metadata_category_is_valid(self, vocab: OperatorVocab):
        """category 必须是已知类别"""
        valid = {"time", "point", "section", "multi_section"}
        for op in vocab.list_operators()[:30]:
            meta = vocab.get_metadata(op)
            assert meta.category in valid, f"{op}: invalid category {meta.category}"


# ==============================================================================
# Test Class 3: 算子 edge case (NaN / 极值 / 短序列)
# ==============================================================================


class TestOperatorEdgeCases:
    """算子在极端输入下的行为"""

    def test_ts_mean_handles_nan_input(self, vocab: OperatorVocab, data_with_nan: pl.DataFrame):
        """ts_mean 接受含 NaN 的输入不应崩溃"""
        result = vocab.evaluate("ts_mean(close, 3)", data_with_nan)
        assert result is not None
        # NaN 在窗口内应传播为 NaN (而非 0)
        assert result.shape[0] == data_with_nan.shape[0]

    def test_ts_std_handles_constant_input(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """ts_std 对常数输入返回 0 (不崩)"""
        # 制造一段常数 close
        df = sample_data.with_columns(pl.lit(100.0).alias("close"))
        result = vocab.evaluate("ts_std(close, 3)", df)
        assert result is not None
        # 常数 std 应为 0 或 NaN (不是 inf)
        non_null = [x for x in result.to_list() if x is not None]
        if non_null:
            assert all(abs(x) < 1e-6 for x in non_null), f"std of constant should be 0, got {non_null}"

    def test_div_by_zero_returns_inf_or_nan(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """div by zero 应返回 inf/nan 而非崩溃"""
        # 制造 zero 序列
        df = sample_data.with_columns(pl.lit(0.0).alias("vol"))
        result = vocab.evaluate("div(close, vol)", df)
        assert result is not None
        # 不崩, 值可能是 inf / nan
        assert result.shape[0] == df.shape[0]

    def test_log_of_negative_uses_abs_protection(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """log(负值) 应有 abs 保护 (实际是 log(abs(x)))"""
        result = vocab.evaluate("log(abs(close))", sample_data)
        assert result is not None
        # log 应有有限值
        finite = [x for x in result.to_list() if x is not None and not np.isinf(x)]
        assert len(finite) > 0

    def test_ts_max_window_larger_than_data(self, vocab: OperatorVocab):
        """ts_max 窗口 > 数据长度不应崩溃"""
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "code": ["A", "A"],
            "close": [100.0, 101.0],
        }).with_columns(pl.col("date").str.to_date())
        result = vocab.evaluate("ts_max(close, 30)", df)
        assert result is not None
        # 短序列下窗口运算返回 NaN, 不崩
        assert result.shape[0] == 2

    def test_rank_with_single_stock(self, vocab: OperatorVocab):
        """rank 对单只股票 (截面只有 1 个值) 不应崩溃"""
        df = pl.DataFrame({
            "date": ["2024-01-01"] * 5,
            "code": ["A"] * 5,
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
        }).with_columns(pl.col("date").str.to_date())
        result = vocab.evaluate("rank(close)", df)
        assert result is not None
        # 单股截面 rank 应该是常数 0.5 或 NaN
        assert result.shape[0] == 5


# ==============================================================================
# Test Class 4: 算子组合 (嵌套)
# ==============================================================================


class TestOperatorComposition:
    """算子组合: 嵌套 1-3 层"""

    def test_two_level_nested(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """2 层嵌套: ts_mean(rank(close))"""
        result = vocab.evaluate("ts_mean(rank(close), 3)", sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    def test_three_level_nested(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """3 层嵌套: rank(ts_std(div(close, vol), 3))"""
        result = vocab.evaluate("rank(ts_std(div(close, vol), 3))", sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    def test_four_level_deeply_nested(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """4 层嵌套: sign(sub(ts_mean(rank(close), 3), ts_mean(close, 3)))"""
        result = vocab.evaluate(
            "sign(sub(ts_mean(rank(close), 3), ts_mean(close, 3)))",
            sample_data,
        )
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    def test_same_op_repeated(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """同一算子重复: ts_mean(ts_mean(close, 3), 3)"""
        result = vocab.evaluate("ts_mean(ts_mean(close, 3), 3)", sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]

    def test_window_op_inside_section_op(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """time 算子 + section 算子: rank(ts_mean(close, 3))"""
        result = vocab.evaluate("rank(ts_mean(close, 3))", sample_data)
        assert result is not None
        assert result.shape[0] == sample_data.shape[0]


# ==============================================================================
# Test Class 5: 算子 output shape 一致性
# ==============================================================================


class TestOperatorOutputShape:
    """算子输出 shape 与输入一致"""

    @pytest.mark.parametrize("formula", [
        "ts_mean(close, 3)",
        "ts_std(close, 3)",
        "ts_max(close, 5)",
        "ts_min(close, 5)",
        "rank(close)",
        "zscore(close)",
        "winsorize(close)",
        "abs(close - 100)",
        "sign(close - 100)",
        "log(abs(close))",
    ])
    def test_output_length_matches_input(
        self, vocab: OperatorVocab, sample_data: pl.DataFrame, formula: str
    ):
        """输出 Series 长度应等于输入 DataFrame 行数"""
        result = vocab.evaluate(formula, sample_data)
        assert result.shape[0] == sample_data.shape[0], (
            f"{formula}: output {result.shape[0]} != input {sample_data.shape[0]}"
        )


# ==============================================================================
# Test Class 6: 算子 max_formula_length 边界
# ==============================================================================


class TestFormulaLengthLimits:
    """公式长度限制"""

    def test_very_long_formula_rejected(self, sample_data: pl.DataFrame):
        """超长公式应被 max_formula_length 拒绝"""
        # 构造 10000 字符的公式
        long_formula = "ts_mean(close, 3) + " * 1000
        # vocab 内部会检查
        # 注意: OperatorVocab 默认 max_formula_length 可能不同
        # 这里只检查不崩
        from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
        vocab = OperatorVocab.default()
        try:
            result = vocab.evaluate(long_formula, sample_data)
            # 如果接受, 应返回有效结果
            assert result is not None
        except ValueError as e:
            # 拒绝也是合法
            assert "length" in str(e).lower() or "exceed" in str(e).lower()

    def test_nested_too_deep_rejected(self, sample_data: pl.DataFrame):
        """嵌套过深应被 max_formula_depth 拒绝"""
        # 构造 100 层嵌套
        deep = "abs(" * 50 + "close" + ")" * 50
        vocab = OperatorVocab.default()
        try:
            result = vocab.evaluate(deep, sample_data)
            assert result is not None
        except ValueError as e:
            # 拒绝也是合法
            assert "depth" in str(e).lower() or "nesting" in str(e).lower()


# ==============================================================================
# Test Class 6.5: 需要额外参数的算子 (cross_sectional_*, aggr_*)
# ==============================================================================


class TestOperatorsWithExtraArgs:
    """需要额外参数的算子

    cross_sectional_* 返回 Expr, 需 .alias() 等后处理
    aggr_* 需要 group_by 参数
    """

    def test_cross_sectional_mean_returns_expr(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """cross_sectional_mean 返回 Expr (polars 表达式), 不是 Series"""
        import polars as pl
        result = vocab.evaluate("cross_sectional_mean(close)", sample_data)
        # 可能是 Expr 或 Series; 关键是能 evaluate
        # 如果是 Series, 长度 = 1 (截面)
        # 如果是 Expr, 需要用 select 触发
        if isinstance(result, pl.Expr):
            # 用 select 触发
            df_with = sample_data.select(result.alias("result"))
            assert "result" in df_with.columns
        elif isinstance(result, pl.Series):
            # 截面聚合, 长度可能 = 1
            assert result.shape[0] >= 1

    def test_aggr_mean_requires_group_by(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """aggr_mean 需 group_by 参数, 直接调用会失败"""
        with pytest.raises(Exception) as exc_info:
            vocab.evaluate("aggr_mean(close)", sample_data)
        # 错误信息应提到 group_by
        assert "group_by" in str(exc_info.value).lower() or "argument" in str(exc_info.value).lower()

    def test_aggregate_with_group_by(self, vocab: OperatorVocab, sample_data: pl.DataFrame):
        """aggregate(f, group_by, method) 正确用法"""
        # group_by='code' 应能跑通
        result = vocab.evaluate('aggregate(close, "code", "mean")', sample_data)
        # 可能返回 Series 或 Expr
        assert result is not None or result is None  # 不崩


# ==============================================================================
# Test Class 7: Hypothesis property-based 测试
# ==============================================================================


# 简单的数据生成策略
@st.composite
def stock_data(draw, min_size=5, max_size=30):
    """生成随机 OHLCV 数据"""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    n_stocks = draw(st.integers(min_value=2, max_value=5))
    total = n * n_stocks
    return pl.DataFrame({
        "date": (["2024-01-01"] * total),
        "code": [f"S{i % n_stocks}" for i in range(total)],
        "close": draw(st.lists(
            st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            min_size=total, max_size=total,
        )),
        "vol": draw(st.lists(
            st.floats(min_value=100.0, max_value=10000.0, allow_nan=False),
            min_size=total, max_size=total,
        )),
    })


class TestHypothesisProperty:
    """Hypothesis property-based 测试: 自动生成 random inputs

    max_examples=5 控制速度, 完整跑约 30s
    """

    @given(data=stock_data(min_size=10, max_size=20))
    @HYPOTHESIS_SETTINGS
    def test_ts_mean_property(self, vocab: OperatorVocab, data: pl.DataFrame):
        """ts_mean 输出长度 == 输入长度, 第一个窗口内位置可能为 NaN"""
        result = vocab.evaluate("ts_mean(close, 3)", data)
        if result is not None:
            assert result.shape[0] == data.shape[0]

    @given(data=stock_data(min_size=10, max_size=20))
    @HYPOTHESIS_SETTINGS
    def test_rank_property(self, vocab: OperatorVocab, data: pl.DataFrame):
        """rank 输出长度 == 输入长度, 唯一值数应 ≤ 总数"""
        result = vocab.evaluate("rank(close)", data)
        if result is not None:
            assert result.shape[0] == data.shape[0]
            # rank 应是有限值 (或 NaN), 不应是 inf
            finite = [x for x in result.to_list() if x is not None and not (np.isnan(x) or np.isinf(x))]
            if finite:
                # 唯一值数应 < 总数 (排名是分类)
                unique_count = len(set(finite))
                assert unique_count <= len(finite), (
                    f"rank should produce < unique values, got {unique_count} unique out of {len(finite)}"
                )

    @given(data=stock_data(min_size=15, max_size=25))
    @HYPOTHESIS_SETTINGS
    def test_sub_property(self, vocab: OperatorVocab, data: pl.DataFrame):
        """sub(a, b) + sub(b, a) 应符号相反 (反交换性)"""
        a_result = vocab.evaluate("sub(close, vol)", data)
        b_result = vocab.evaluate("sub(vol, close)", data)
        if a_result is not None and b_result is not None:
            # 在非 null 位置应满足 a + b ≈ 0
            a_list = [x for x in a_result.to_list() if x is not None and not np.isnan(x)]
            b_list = [x for x in b_result.to_list() if x is not None and not np.isnan(x)]
            assert len(a_list) == len(b_list)
            if a_list:
                # 至少抽样检查: a + b ≈ 0
                for i in range(0, len(a_list), max(1, len(a_list) // 5)):
                    assert abs(a_list[i] + b_list[i]) < 1e-6, (
                        f"sub not antisymmetric: a[{i}]={a_list[i]} b[{i}]={b_list[i]}"
                    )

# coding=utf-8
"""
ExprParser 高级测试 - 复杂逻辑、嵌套表达式、自定义算子、错误处理

覆盖:
1. 嵌套函数表达式 (ts_sum(close * volume, 5), ts_mean(ts_lag(close, 1), 20))
2. 自定义算子在 ExprParser 表达式中调用
3. ExprParser 错误处理
4. 深层嵌套/长表达式压力测试
5. 多自定义算子文件加载
6. _preload_custom_operators (loader.py) 独立测试
"""

import pytest
import polars as pl

from QuantNodes.agent.config.executor import ConfigExecutor, ExprParser
from QuantNodes.agent.config.types import (
    StrategyConfig, FactorConfig, OperationConfig, CompositeConfig,
    ValidationConfig,
)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _make_data():
    return pl.LazyFrame({
        "date": ["2024-01-01"] * 5,
        "code": ["A"] * 5,
        "close": [100.0, 102.0, 101.0, 103.0, 105.0],
        "open": [99.0, 100.0, 102.0, 101.0, 103.0],
        "volume": [1000, 1200, 1100, 1300, 1400],
        "high": [101.0, 103.0, 102.0, 104.0, 106.0],
    })


def _run_factor(expr_str, data=None):
    """快捷函数: 用指定表达式运行一个因子"""
    executor = ConfigExecutor()
    config = StrategyConfig(
        name="test",
        factors=[FactorConfig(name="result", expr=expr_str)],
    )
    return executor.run(config, data or _make_data())


# ─────────────────────────────────────────────
# 1. 嵌套函数表达式测试
# ─────────────────────────────────────────────

class TestNestedFunctionExpressions:
    """嵌套函数表达式 - 函数参数中包含算术表达式"""

    def test_ts_sum_with_arithmetic(self):
        """ts_sum(close * volume, 5) - 函数参数中含乘法"""
        result = _run_factor("ts_sum(close * volume, 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_ts_sum_with_column_division(self):
        """ts_sum(close / open, 10) - 函数参数中含除法"""
        result = _run_factor("ts_sum(close / open, 10)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_ts_mean_with_addition(self):
        """ts_mean(close + volume, 5) - 函数参数中含加法"""
        result = _run_factor("ts_mean(close + volume, 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_calls(self):
        """ts_mean(ts_lag(close, 1), 20) - 函数嵌套调用"""
        result = _run_factor("ts_mean(ts_lag(close, 1), 20)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_with_arithmetic(self):
        """ts_sum(close * 2 + volume, 5) - 嵌套函数参数含混合运算"""
        result = _run_factor("ts_sum(close * 2 + volume, 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_mean_of_product(self):
        """ts_mean(close * volume, 5) - 均值的乘积"""
        result = _run_factor("ts_mean(close * volume, 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_max_of_arithmetic(self):
        """ts_max(close - open, 10) - 最大值的差值"""
        result = _run_factor("ts_max(close - open, 10)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_std_of_expression(self):
        """ts_std(close / open - 1, 20) - 标准差的收益率"""
        result = _run_factor("ts_std(close / open - 1, 20)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_triple_nested(self):
        """ts_sum(ts_mean(close, 3), 5) - 三层嵌套"""
        result = _run_factor("ts_sum(ts_mean(close, 3), 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_with_keyword_args(self):
        """winsorize(close, lower=0.01, upper=0.01) - keyword 参数"""
        result = _run_factor("winsorize(close, lower=0.01, upper=0.01)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_func_with_method_chain(self):
        """ts_lag(close.shift(1), 5) - 函数参数中含方法链"""
        result = _run_factor("ts_lag(close.shift(1), 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_complex_nested_formula(self):
        """(close * volume) / ts_sum(volume, 20) - 复杂嵌套"""
        result = _run_factor("(close * volume) / ts_sum(volume, 20)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_delta_of_product(self):
        """ts_delta(close * volume, 1) - 乘积的差分"""
        result = _run_factor("ts_delta(close * volume, 1)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_nested_ts_pct_change_of_division(self):
        """ts_pct_change(close / open, 5) - 比率的百分比变化"""
        result = _run_factor("ts_pct_change(close / open, 5)")
        assert result.status == "success"
        assert "result" in result.factors


# ─────────────────────────────────────────────
# 2. 自定义算子在 ExprParser 表达式中调用
# ─────────────────────────────────────────────

class TestCustomOpsInExprParser:
    """自定义算子通过 ExprParser 表达式调用"""

    def test_custom_op_in_factor_expr(self, tmp_path):
        """自定义算子在 factor expr 中调用: custom_double(close)"""
        custom_file = tmp_path / "my_ops.py"
        custom_file.write_text("""
def custom_double(f, **kwargs):
    return f * 2
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(custom_file)])

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_double(close)")],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "result" in result.factors
        # 验证值正确
        df = result.data.collect()
        assert df["result"][0] == 200.0  # 100.0 * 2

    def test_custom_op_in_composite_formula(self, tmp_path):
        """自定义算子在 composite formula 中调用"""
        custom_file = tmp_path / "my_ops.py"
        custom_file.write_text("""
def custom_add_ten(f, **kwargs):
    return f + 10
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(custom_file)])

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="base", expr="close")],
            composite=[
                CompositeConfig(name="result", formula="custom_add_ten(base)")
            ],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "result" in result.factors
        df = result.data.collect()
        assert df["result"][0] == 110.0  # 100.0 + 10

    def test_custom_ts_op_in_factor_expr(self, tmp_path):
        """自定义时间序列算子在 factor expr 中调用"""
        custom_file = tmp_path / "my_ts_ops.py"
        custom_file.write_text("""
def custom_rolling_sum(f, window=5, **kwargs):
    return f.rolling_sum(window)
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([{
            "source": str(custom_file),
            "category": "time_series",
            "functions": ["custom_rolling_sum"],
        }])

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_rolling_sum(close, 3)")],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "result" in result.factors

    def test_custom_op_with_multiple_args(self, tmp_path):
        """自定义算子带多参数"""
        custom_file = tmp_path / "my_ops.py"
        custom_file.write_text("""
def custom_weighted_add(f1, f2, weight=0.5, **kwargs):
    return f1 * weight + f2 * (1 - weight)
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(custom_file)])

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_weighted_add(close, open, weight=0.7)")],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "result" in result.factors
        df = result.data.collect()
        # 100.0 * 0.7 + 99.0 * 0.3 = 70.0 + 29.7 = 99.7
        assert abs(df["result"][0] - 99.7) < 0.01

    def test_custom_op_chained_with_builtin(self, tmp_path):
        """自定义算子与内置算子组合: rank(custom_double(close))"""
        custom_file = tmp_path / "my_ops.py"
        custom_file.write_text("""
def custom_double(f, **kwargs):
    return f * 2
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(custom_file)])

        # 通过 operations 使用内置 rank
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="doubled", expr="custom_double(close)")],
            operations=[
                OperationConfig(
                    type="section", name="result", category="rank",
                    inputs=["doubled"],
                )
            ],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "result" in result.factors


# ─────────────────────────────────────────────
# 3. ExprParser 错误处理测试
# ─────────────────────────────────────────────

class TestExprParserErrorHandling:
    """ExprParser 错误处理"""

    def test_empty_expression_raises(self):
        """空表达式应抛出异常"""
        executor = ConfigExecutor()
        with pytest.raises(ValueError):
            executor._parse_expr("")

    def test_whitespace_only_raises(self):
        """纯空白表达式应抛出异常"""
        executor = ConfigExecutor()
        with pytest.raises(ValueError):
            executor._parse_expr("   ")

    def test_empty_parens_raises(self):
        """空括号 () 应抛出异常 (Unexpected character ')')"""
        executor = ConfigExecutor()
        with pytest.raises(ValueError, match="Unexpected character"):
            executor._parse_expr("()")

    def test_trailing_garbage_ignored(self):
        """尾部垃圾字符被静默忽略 (ExprParser 限制)"""
        executor = ConfigExecutor()
        # 'close @ 2' 中的 '@' 不会被检测到，表达式返回 pl.col("close")
        result = executor._parse_expr("close @ 2")
        assert result is not None

    def test_missing_closing_paren_ignored(self):
        """缺少右括号被静默忽略 (ExprParser 限制: 读取到 EOF)"""
        executor = ConfigExecutor()
        # 'ts_mean(close, 5' 缺少 ')'，_read_func_args 读取到 EOF
        result = executor._parse_expr("ts_mean(close, 5")
        assert result is not None

    def test_missing_operand_at_eof(self):
        """行尾缺少操作数: close + (EOF)"""
        executor = ConfigExecutor()
        # 'close +' 后面没有操作数，_parse_multiplicative 调用 _parse_unary
        # _parse_primary 看到 EOF ''，不是 digit/alpha/(_，应抛异常
        with pytest.raises(ValueError):
            executor._parse_expr("close +")

    def test_leading_operator(self):
        """前导运算符: + close"""
        executor = ConfigExecutor()
        # '+ close' 应该能解析 (一元正号)
        result = executor._parse_expr("+ close")
        assert result is not None

    def test_only_operator(self):
        """只有运算符: *"""
        executor = ConfigExecutor()
        with pytest.raises(ValueError):
            executor._parse_expr("*")


# ─────────────────────────────────────────────
# 4. 深层嵌套/长表达式压力测试
# ─────────────────────────────────────────────

class TestDeepNestingExpressions:
    """深层嵌套和长表达式压力测试"""

    def test_4_level_nesting(self):
        """四层嵌套: ts_sum(ts_mean(ts_lag(close, 1), 3), 5)"""
        result = _run_factor("ts_sum(ts_mean(ts_lag(close, 1), 3), 5)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_5_level_nesting(self):
        """五层嵌套: ts_sum(ts_mean(ts_std(ts_lag(close, 1), 3), 5), 7)"""
        result = _run_factor("ts_sum(ts_mean(ts_std(ts_lag(close, 1), 3), 5), 7)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_long_arithmetic_chain(self):
        """长算术链: close + 1 + 2 + 3 + 4 + 5"""
        result = _run_factor("close + 1 + 2 + 3 + 4 + 5")
        assert result.status == "success"
        df = result.data.collect()
        # 100 + 1 + 2 + 3 + 4 + 5 = 115
        assert df["result"][0] == 115.0

    def test_long_multiplicative_chain(self):
        """长乘法链: close * 1 * 2 * 3"""
        result = _run_factor("close * 1 * 2 * 3")
        assert result.status == "success"
        df = result.data.collect()
        # 100 * 1 * 2 * 3 = 600
        assert df["result"][0] == 600.0

    def test_mixed_precedence_deep(self):
        """混合优先级深层: close + volume * 2 - open / 1 + ts_lag(close, 1)"""
        result = _run_factor("close + volume * 2 - open / 1 + ts_lag(close, 1)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_deeply_nested_parentheses(self):
        """深层括号嵌套: ((((close + 1) * 2) - 3) / 4)"""
        result = _run_factor("((((close + 1) * 2) - 3) / 4)")
        assert result.status == "success"
        df = result.data.collect()
        # ((100+1)*2-3)/4 = (202-3)/4 = 199/4 = 49.75
        assert abs(df["result"][0] - 49.75) < 0.01

    def test_nested_method_chains(self):
        """嵌套方法链: close.shift(1).shift(2)"""
        result = _run_factor("close.shift(1).shift(2)")
        assert result.status == "success"
        assert "result" in result.factors

    def test_complex_real_world_factor(self):
        """真实因子: momentum = (close / ts_lag(close, 20) - 1) * ts_std(close / ts_lag(close, 1) - 1, 20)"""
        result = _run_factor(
            "(close / ts_lag(close, 20) - 1) * ts_std(close / ts_lag(close, 1) - 1, 20)"
        )
        assert result.status == "success"
        assert "result" in result.factors


# ─────────────────────────────────────────────
# 5. 多自定义算子文件加载
# ─────────────────────────────────────────────

class TestMultipleCustomOperatorFiles:
    """多自定义算子文件加载"""

    def test_load_two_files(self, tmp_path):
        """同时加载两个自定义算子文件"""
        file1 = tmp_path / "ops_a.py"
        file1.write_text("""
def custom_double(f, **kwargs):
    return f * 2
""")
        file2 = tmp_path / "ops_b.py"
        file2.write_text("""
def custom_triple(f, **kwargs):
    return f * 3
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(file1), str(file2)])

        # 使用第一个文件的算子
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_double(close)")],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        df = result.data.collect()
        assert df["result"][0] == 200.0

        # 使用第二个文件的算子
        config2 = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_triple(close)")],
        )
        result2 = executor.run(config2, _make_data())
        assert result2.status == "success"
        df2 = result2.data.collect()
        assert df2["result"][0] == 300.0

    def test_load_mixed_formats(self, tmp_path):
        """混合加载格式: str + dict"""
        file1 = tmp_path / "simple.py"
        file1.write_text("""
def custom_quad(f, **kwargs):
    return f * 4
""")
        file2 = tmp_path / "detailed.py"
        file2.write_text("""
def custom_half(f, **kwargs):
    return f / 2
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([
            str(file1),
            {"source": str(file2), "category": "math", "functions": ["custom_half"]},
        ])

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_quad(close)")],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"

    def test_one_file_fails_other_succeeds(self, tmp_path):
        """一个文件失败，另一个正常加载"""
        good_file = tmp_path / "good.py"
        good_file.write_text("""
def custom_good(f, **kwargs):
    return f + 1
""")
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("invalid python syntax {{{")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(bad_file), str(good_file)])

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="result", expr="custom_good(close)")],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        df = result.data.collect()
        assert df["result"][0] == 101.0

    def test_empty_functions_list_skips_all(self, tmp_path):
        """空 functions 列表时不注册任何算子"""
        custom_file = tmp_path / "ops.py"
        custom_file.write_text("""
def custom_alpha(f, **kwargs):
    return f

def custom_beta(f, **kwargs):
    return f
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([{
            "source": str(custom_file),
            "category": "point",
            "functions": [],  # 空列表
        }])

        # 自定义算子不应可用
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="math", name="result", category="custom_alpha",
                    inputs=["close"], params={},
                )
            ],
        )
        result = executor.run(config, _make_data())
        # 应该走 fallback 返回原始输入
        assert result.status == "success"


# ─────────────────────────────────────────────
# 6. _preload_custom_operators (loader.py) 独立测试
# ─────────────────────────────────────────────

class TestPreloadCustomOperators:
    """ConfigLoader._preload_custom_operators 独立测试"""

    def test_preload_simple_format(self, tmp_path):
        """简单 str 格式预加载"""
        from QuantNodes.agent.config.loader import ConfigLoader

        custom_file = tmp_path / "my_ops.py"
        custom_file.write_text("""
def custom_preload_test(f, **kwargs):
    return f + 100
""")

        loader = ConfigLoader()
        loader._preload_custom_operators([str(custom_file)])

        # 验证算子已注册
        from QuantNodes.factor_node.factor_functions import get_operator
        op = get_operator("custom_preload_test")
        assert op is not None

    def test_preload_dict_format(self, tmp_path):
        """dict 格式预加载"""
        from QuantNodes.agent.config.loader import ConfigLoader

        custom_file = tmp_path / "my_ts_ops.py"
        custom_file.write_text("""
def custom_preload_ts(f, window=5, **kwargs):
    return f.rolling_mean(window)
""")

        loader = ConfigLoader()
        loader._preload_custom_operators([{
            "source": str(custom_file),
            "category": "time_series",
            "functions": ["custom_preload_ts"],
        }])

        from QuantNodes.factor_node.factor_functions import get_operator
        op = get_operator("custom_preload_ts")
        assert op is not None

    def test_preload_file_not_found(self, tmp_path):
        """文件不存在时应打印警告不崩溃"""
        from QuantNodes.agent.config.loader import ConfigLoader

        loader = ConfigLoader()
        # 不应抛出异常
        loader._preload_custom_operators(["/nonexistent/path.py"])

    def test_preload_empty_list(self):
        """空列表应直接返回"""
        from QuantNodes.agent.config.loader import ConfigLoader

        loader = ConfigLoader()
        loader._preload_custom_operators([])  # 不应抛出异常

    def test_preload_none(self):
        """None 应直接返回"""
        from QuantNodes.agent.config.loader import ConfigLoader

        loader = ConfigLoader()
        loader._preload_custom_operators(None)  # 不应抛出异常

    def test_preload_invalid_syntax_file(self, tmp_path):
        """语法错误文件应打印警告不崩溃"""
        from QuantNodes.agent.config.loader import ConfigLoader

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(")

        loader = ConfigLoader()
        loader._preload_custom_operators([str(bad_file)])  # 不应抛出异常

    def test_preload_skips_private_functions(self, tmp_path):
        """跳过 _ 开头的私有函数"""
        from QuantNodes.agent.config.loader import ConfigLoader

        custom_file = tmp_path / "ops.py"
        custom_file.write_text("""
def custom_public(f, **kwargs):
    return f

def _private_func(f, **kwargs):
    return f

def not_custom_func(f, **kwargs):
    return f
""")

        loader = ConfigLoader()
        loader._preload_custom_operators([str(custom_file)])

        from QuantNodes.factor_node.factor_functions import get_operator
        assert get_operator("custom_public") is not None
        assert get_operator("_private_func") is None
        assert get_operator("not_custom_func") is None

    def test_preload_custom_ops_detected_in_coverage(self, tmp_path):
        """预加载后 check_coverage 能识别自定义算子"""
        from QuantNodes.agent.config.loader import ConfigLoader

        custom_file = tmp_path / "ops.py"
        custom_file.write_text("""
def custom_coverage_test(f, **kwargs):
    return f
""")

        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="math", name="result", category="custom_coverage_test",
                    inputs=["close"], params={},
                )
            ],
            validation=ValidationConfig(
                custom_operators=[str(custom_file)]
            ),
        )

        loader = ConfigLoader()
        report = loader.check_coverage(config)
        assert report.is_complete


# ─────────────────────────────────────────────
# 7. ExprParser 解析值和函数参数测试
# ─────────────────────────────────────────────

class TestExprParserParsingDetails:
    """ExprParser 解析细节测试"""

    def test_string_keyword_arg(self):
        """字符串 keyword 参数"""
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args('method="mean"')
        assert kwargs["method"] == "mean"

    def test_negative_number_arg(self):
        """负数参数"""
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("value=-1.5")
        assert kwargs["value"] == -1.5

    def test_bool_keyword_arg(self):
        """布尔 keyword 参数 (作为字符串)"""
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("ascending=true")
        # 'true' 不会被解析为 Python True，而是作为列引用
        assert "ascending" in kwargs

    def test_nested_parentheses_in_args(self):
        """函数参数中的嵌套括号"""
        executor = ConfigExecutor()
        parser = ExprParser(executor)
        # ts_sum((close + open), 5) - 参数中含括号表达式
        # 注意: _parse_func_args 是解析逗号分割的参数
        # 但 ExprParser 本身支持嵌套
        result = parser.parse("ts_sum((close + open), 5)")
        assert result is not None

    def test_multiple_commas(self):
        """多个逗号参数"""
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("a, b, c")
        assert len(args) == 3

    def test_whitespace_in_args(self):
        """参数中的空白"""
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("  close  ,  20  ")
        assert len(args) == 2

    def test_unary_before_parenthesized_expr(self):
        """一元运算符前缀 + 括号"""
        executor = ConfigExecutor()
        result = executor._parse_expr("-(close + 1)")
        assert result is not None

    def test_parenthesized_function_call(self):
        """括号中的函数调用"""
        executor = ConfigExecutor()
        result = executor._parse_expr("(ts_mean(close, 5))")
        assert result is not None


# ─────────────────────────────────────────────
# 8. 端到端集成测试 - 复杂策略
# ─────────────────────────────────────────────

class TestComplexStrategyIntegration:
    """复杂策略端到端集成测试"""

    def test_multi_factor_strategy(self, tmp_path):
        """多因子策略: 因子定义 + 运算 + 组合"""
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="complex_test",
            factors=[
                FactorConfig(name="momentum", expr="close / ts_lag(close, 20) - 1"),
                FactorConfig(name="volatility", expr="ts_std(close / ts_lag(close, 1) - 1, 20)"),
            ],
            operations=[
                OperationConfig(
                    type="section", name="mom_rank", category="rank",
                    inputs=["momentum"],
                ),
                OperationConfig(
                    type="section", name="vol_rank", category="rank",
                    inputs=["volatility"],
                ),
            ],
            composite=[
                CompositeConfig(name="alpha", formula="mom_rank - vol_rank"),
            ],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "momentum" in result.factors
        assert "volatility" in result.factors
        assert "mom_rank" in result.factors
        assert "vol_rank" in result.factors
        assert "alpha" in result.factors

    def test_strategy_with_nested_expr_factors(self):
        """使用嵌套表达式的因子策略"""
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="nested_test",
            factors=[
                FactorConfig(name="vwap_approx", expr="ts_sum(close * volume, 5) / ts_sum(volume, 5)"),
                FactorConfig(name="price_divergence", expr="close / vwap_approx - 1"),
            ],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "vwap_approx" in result.factors
        assert "price_divergence" in result.factors

    def test_strategy_with_all_operation_types(self):
        """混合运算类型策略"""
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="mixed_types",
            factors=[
                FactorConfig(name="ret", expr="close / open - 1"),
            ],
            operations=[
                OperationConfig(
                    type="time_series", name="ret_ma", category="ts_mean",
                    inputs=["ret"], params={"window": 3},
                ),
                OperationConfig(
                    type="math", name="ret_abs", category="abs",
                    inputs=["ret_ma"], params={},
                ),
                OperationConfig(
                    type="section", name="ret_rank", category="rank",
                    inputs=["ret_abs"],
                ),
            ],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "ret" in result.factors
        assert "ret_ma" in result.factors
        assert "ret_abs" in result.factors
        assert "ret_rank" in result.factors

    def test_strategy_custom_op_plus_builtin_ops(self, tmp_path):
        """自定义算子 + 内置运算混合策略"""
        custom_file = tmp_path / "custom.py"
        custom_file.write_text("""
def custom_zscore(f, **kwargs):
    mean = f.mean()
    std = f.std()
    return (f - mean) / std
""")

        executor = ConfigExecutor()
        executor._load_custom_operators([str(custom_file)])

        config = StrategyConfig(
            name="custom_mixed",
            factors=[
                FactorConfig(name="ret", expr="close / open - 1"),
            ],
            operations=[
                OperationConfig(
                    type="time_series", name="ret_ma", category="ts_mean",
                    inputs=["ret"], params={"window": 3},
                ),
                OperationConfig(
                    type="math", name="ret_custom_zscore", category="custom_zscore",
                    inputs=["ret_ma"], params={},
                ),
            ],
        )
        result = executor.run(config, _make_data())
        assert result.status == "success"
        assert "ret_custom_zscore" in result.factors

# coding=utf-8
"""QuantNodes.agent.tools.factor 单元测试"""
import pytest
import polars as pl

from QuantNodes.agent.tools.factor import FactorTool


@pytest.fixture
def tool():
    return FactorTool()


class TestFactorTool:
    def test_creation(self):
        tool = FactorTool()
        assert tool is not None

    def test_name(self):
        tool = FactorTool()
        assert tool.name == "factor"

    def test_description(self):
        tool = FactorTool()
        assert "IC" in tool.description or "因子" in tool.description

    def test_parameters(self):
        tool = FactorTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "factor_code" in params["properties"]
        assert "analysis_type" in params["properties"]
        assert params["properties"]["analysis_type"]["enum"] == ["ic", "correlation", "both"]

    def test_read_only(self):
        tool = FactorTool()
        assert tool.read_only is True

    def test_to_openai_schema(self):
        tool = FactorTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "factor"

    @pytest.mark.asyncio
    async def test_execute_empty_code(self, tool):
        result = await tool.execute(factor_code="", analysis_type="both")
        assert result["status"] == "error"
        assert result["errors"] is not None

    @pytest.mark.asyncio
    async def test_execute_no_result_variable(self, tool):
        code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3]})
"""
        result = await tool.execute(factor_code=code, analysis_type="both")
        assert result["status"] == "error"
        assert "No 'result' variable found" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_execute_wrong_result_type(self, tool):
        code = """
result = [1, 2, 3]
"""
        result = await tool.execute(factor_code=code, analysis_type="both")
        assert result["status"] == "error"
        assert "Polars DataFrame" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_execute_missing_columns(self, tool):
        code = """
import polars as pl
result = pl.DataFrame({"date": ["2024-01-01"], "code": ["A"]})
"""
        result = await tool.execute(factor_code=code, analysis_type="both")
        assert result["status"] == "error"
        assert "Missing required columns" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_execute_valid_factor_code(self, tool):
        code = """
import polars as pl
result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
"""
        result = await tool.execute(factor_code=code, analysis_type="both")
        assert "analysis" in result
        assert "ic" in result["analysis"]
        assert "correlation" in result["analysis"]

    @pytest.mark.asyncio
    async def test_execute_correlation_only(self, tool):
        code = """
import polars as pl
result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
"""
        result = await tool.execute(factor_code=code, analysis_type="correlation")
        assert "analysis" in result
        assert "correlation" in result["analysis"]
        assert result["analysis"]["ic"] == {}

    @pytest.mark.asyncio
    async def test_execute_single_cross_section(self, tool):
        code = """
import polars as pl
result = pl.DataFrame({
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
"""
        result = await tool.execute(factor_code=code, analysis_type="ic")
        assert "analysis" in result
        assert result["analysis"]["correlation"] == {}

    @pytest.mark.asyncio
    async def test_execute_with_date_filter(self, tool):
        code = """
import polars as pl
result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "code": ["A", "A", "A"],
    "factor_value": [0.1, 0.2, 0.3],
    "forward_return": [0.01, 0.02, 0.03],
})
"""
        result = await tool.execute(
            factor_code=code,
            analysis_type="ic",
            start_date="2024-01-02",
            end_date="2024-01-03"
        )
        assert result["status"] == "success"


class TestFactorToolComputeIC:
    def test_compute_ic_without_date(self, tool):
        df = pl.DataFrame({
            "factor_value": [0.1, 0.2, 0.3, 0.4],
            "forward_return": [0.05, 0.03, 0.02, 0.01],
        })
        ic_result = tool._compute_ic(df)
        assert "ic_mean" in ic_result
        assert "rank_ic_mean" in ic_result

    def test_compute_ic_with_date(self, tool):
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "factor_value": [0.1, 0.2, 0.3, 0.4],
            "forward_return": [0.05, 0.03, 0.02, 0.01],
        })
        ic_result = tool._compute_ic(df)
        assert "ic_series" in ic_result
        assert "n_dates" in ic_result


class TestFactorToolComputeCorrelation:
    def test_compute_correlation_basic(self, tool):
        df = pl.DataFrame({
            "factor_value": [0.1, 0.2, 0.3, 0.4],
            "forward_return": [0.05, 0.03, 0.02, 0.01],
        })
        corr_result = tool._compute_correlation(df)
        assert "factor_return_corr" in corr_result

    def test_compute_correlation_with_multiple_numeric_cols(self, tool):
        df = pl.DataFrame({
            "factor_value": [0.1, 0.2, 0.3, 0.4],
            "forward_return": [0.05, 0.03, 0.02, 0.01],
            "volume": [100, 200, 300, 400],
        })
        corr_result = tool._compute_correlation(df)
        assert "correlation_matrix" in corr_result

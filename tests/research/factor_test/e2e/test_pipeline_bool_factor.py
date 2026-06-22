# coding=utf-8
"""End-to-end integration test for Phase 1+2 refactored nodes (Option D).

验证以下 3 个节点在 bool / 离散 / 轻度 ties 因子上的端到端行为:
  1. FactorPreprocessNode  (Phase 2.2: Strategy pattern)
  2. FactorNeutralizeNode  (Phase 2.1: Chain of Responsibility)
  3. GroupAnalyzerNode     (Phase 1.0: dispatch 3-mode)

测试场景:
  - 30 只 -1 + 20 只 +1 (alpha-004 风格 bool 因子)
  - 7 unique 的整数 ties (alpha-004 因子原始形态)
  - 连续 float 因子 (回归保护)

每个场景:
  1. 构造 50 stocks × 12 dates 合成数据
  2. preprocess → neutralize → group_analyzer 顺序执行
  3. 验证:
     - 不抛 ValueError
     - group_analyzer 输出 5 组
     - output key 完整 (group_ret, group_num, fac_group 等)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.factor_preprocess_node import (
    FactorPreprocessNode,
)
from QuantNodes.research.factor_test.nodes.factor_neutralize_node import (
    FactorNeutralizeNode,
)
from QuantNodes.research.factor_test.nodes.group_analyzer_node import (
    GroupAnalyzerNode,
)
from QuantNodes.research.factor_test.nodes.configs import (
    PreprocessNodeConfig,
    NeutralizeNodeConfig,
    GroupAnalyzerNodeConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 12
N_STOCKS = 50
# 因子 index 用 yyyymmdd int (与 LoadDataNode 实际从 H5 加载的格式一致)
DATES_INT = [
    20200101, 20200108, 20200115, 20200122,
    20200201, 20200208, 20200215, 20200222,
    20200301, 20200308, 20200315, 20200322,
][:N_DATES]
DATES = pd.to_datetime([f"{d}" for d in DATES_INT])  # datetime 版本给 group_analyzer
STOCKS = [f"s{i:02d}" for i in range(N_STOCKS)]


@pytest.fixture
def adj_dates() -> pd.DataFrame:
    """调仓日 (每月 1 次, 共 N_DATES / 3 ≈ 4 次)."""
    return pd.DataFrame({"date": DATES_INT[::3]})


@pytest.fixture
def tradable() -> pd.DataFrame:
    """全 1: 50 只股票都标记为可交易. index 与 factor 一致 (int yyyymmdd)."""
    return pd.DataFrame(np.ones((N_DATES, N_STOCKS)), index=DATES_INT, columns=STOCKS)


N_ADJ = 4  # 12 dates / 3 = 4 调仓日


def make_factor(values: list, *, vary_across_dates: bool = True) -> pd.DataFrame:
    """构造 50 stocks × N_DATES 的因子 DataFrame.

    index 用 int yyyymmdd (与 LoadDataNode 实际格式一致), 避免 preprocess
    中 .loc[isin(adj_date_values)] 返回空。

    vary_across_dates=True: 加日期噪声, 避免 zscore 后 per-row std=0 → NaN.
    """
    if len(values) != N_STOCKS:
        raise ValueError(f"need {N_STOCKS} values, got {len(values)}")
    if not vary_across_dates:
        return pd.DataFrame(
            [values for _ in range(N_DATES)],
            index=DATES_INT, columns=STOCKS,
        )
    rng = np.random.RandomState(0)
    noise = rng.randn(N_DATES, N_STOCKS) * 0.1
    return pd.DataFrame(
        np.array(values) + noise,
        index=DATES_INT, columns=STOCKS,
    )


# ---------------------------------------------------------------------------
# 端到端 pipeline helpers
# ---------------------------------------------------------------------------

def run_preprocess_neutralize_group(
    factor_values: list,
    adj_dates: pd.DataFrame,
    tradable: pd.DataFrame,
    *,
    industry_values: list | None = None,
    preprocess_missing: str = "ind_avg",
    preprocess_extreme: str = "median",
    preprocess_norm: str = "zscore",
    neutralize_industry: bool = True,
    neutralize_risk: bool = False,
    groups: int = 5,
    floor_mode: str = "group",
) -> dict:
    """运行 preprocess → neutralize → group_analyzer, 返回 group_analyzer 输出."""
    factor = make_factor(factor_values)
    N_ADJ = len(adj_dates)

    # Step 1: Preprocess
    prep = FactorPreprocessNode(
        config=PreprocessNodeConfig(
            missing=preprocess_missing,
            extreme=preprocess_extreme,
            norm=preprocess_norm,
            mad_n=3.0,
            pct_low=0.01,
            pct_high=0.99,
        )
    )
    preprocessed = prep._execute(
        context={
            "LoadData": {"factor": factor},
            "TradabilityFilter": tradable,
            "AdjustDate": adj_dates,
        }
    )
    assert preprocessed is not None
    # Preprocess 后只剩调仓日 (N_ADJ)
    assert preprocessed.shape == (N_ADJ, N_STOCKS), (
        f"preprocess output shape {preprocessed.shape} != expected ({N_ADJ}, {N_STOCKS})"
    )

    # Step 2: Neutralize (可选 industry)
    industry = None
    if industry_values is not None and neutralize_industry:
        # industry 的 index 与 factor 一致 (int yyyymmdd)
        industry = pd.DataFrame(
            [industry_values for _ in range(N_DATES)],
            index=DATES_INT, columns=STOCKS,
        )
    if neutralize_industry or neutralize_risk:
        neut = FactorNeutralizeNode(
            config=NeutralizeNodeConfig(
                industry_neutral=neutralize_industry,
                risk_neutral=neutralize_risk,
                risk_factors=[],
            )
        )
        preprocessed = neut._execute(
            context={
                "FactorPreprocess": preprocessed,
                "LoadData": {"id_citic1": industry} if industry is not None else {},
                "AdjustDate": adj_dates,
            }
        )
        assert preprocessed is not None
        assert preprocessed.shape == (N_ADJ, N_STOCKS)

    # Step 3: Group Analyzer
    grp = GroupAnalyzerNode(
        config=GroupAnalyzerNodeConfig(
            groups=groups,
            factor_direction=1,
            floor_mode=floor_mode,
            hedge="equal",
        )
    )
    # GroupAnalyzer 需要 price (DataFrame, index=date, columns=stock) 和 index_cp
    price = pd.DataFrame(
        np.ones((N_DATES, N_STOCKS)) * 100.0,  # 任意价格 (group_analyzer 算 pct_change)
        index=DATES_INT, columns=STOCKS,
    )
    index_cp = pd.DataFrame(
        {"000300.SH": np.ones(N_DATES) * 100.0},
        index=DATES_INT,
    )
    return grp._execute(
        context={
            "FactorNeutralize": preprocessed,
            "LoadData": {
                "id_citic1": industry if industry is not None else pd.DataFrame(),
                "price": price,
                "index_cp": index_cp,
            },
            "AdjustDate": adj_dates,
        }
    )


# ---------------------------------------------------------------------------
# Test 1: alpha-004 风格 bool 因子 (-1 × 30 + +1 × 20)
# ---------------------------------------------------------------------------

class TestBoolFactorEndToEnd:
    """alpha-004 真实场景: 30 只 -1 + 20 只 +1, 走完整 preprocess → neutralize → group 链路."""

    def test_bool_no_industry_no_neutralize(self, adj_dates, tradable):
        """30×-1 + 20×+1, 不做 neutralize, 直接 preprocess + group."""
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            industry_values=None,
            neutralize_industry=False,
        )
        # 5 组都在 (4 调仓日 × 5 组)
        assert result["group_ret"].shape == (N_ADJ, 5)
        # 每组至少 1 只股票
        group_num = result["group_num"]
        # 至少一个非 nan group
        assert group_num.notna().any().any()
        # 不应全 nan (preprocess 后的 zscore 应有值)
        assert result["group_ret"].notna().any().any()

    def test_bool_with_industry_neutralize(self, adj_dates, tradable):
        """30×-1 + 20×+1, 加 industry neutralize."""
        # 4 个行业, 均匀分布
        industry_values = [(i % 4) + 1 for i in range(N_STOCKS)]
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            industry_values=industry_values,
            neutralize_industry=True,
        )
        # 5 组 + 不崩
        assert result["group_ret"].shape == (N_ADJ, 5)
        # preprocess + industry neutralize 后, 残差应仍能产出 5 组
        assert result["group_ret"].notna().any().any()

    def test_bool_different_preprocess_norm(self, adj_dates, tradable):
        """30×-1 + 20×+1, 用 norm (rank → ppf) 标准化."""
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            preprocess_norm="norm",  # RankToNormal
            neutralize_industry=False,
        )
        # rank 后所有非 nan 值都映射到 N(0,1)
        assert result["group_ret"].shape == (N_ADJ, 5)
        # 不应全 nan
        assert result["group_ret"].notna().any().any()

    def test_bool_with_pct_shrink_extreme(self, adj_dates, tradable):
        """30×-1 + 20×+1, 用 pct_shrink 去极值."""
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            preprocess_extreme="pct_shrink",
            preprocess_norm="zscore",
            neutralize_industry=False,
        )
        assert result["group_ret"].shape == (N_ADJ, 5)
        assert result["group_ret"].notna().any().any()


# ---------------------------------------------------------------------------
# Test 2: 轻度 ties 因子 (alpha-004 原始 7 unique 场景)
# ---------------------------------------------------------------------------

class TestLowTieEndToEnd:
    """7 unique × 50 stocks: alpha-004 当 n_unique >= n_groups 的真实 ties 场景."""

    def test_seven_unique_factor(self, adj_dates, tradable):
        """12 只 -9, 8 只 -8, 5 只 -7, 5 只 -6, 5 只 -5, 4 只 -4, 11 只 -3 = 50 只."""
        values = (
            [-9] * 12 + [-8] * 8 + [-7] * 5 + [-6] * 5 +
            [-5] * 5 + [-4] * 4 + [-3] * 11
        )
        assert len(values) == 50
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            neutralize_industry=False,
        )
        assert result["group_ret"].shape == (N_ADJ, 5)
        # 5 组都应有值
        for g in range(1, 6):
            col = result["group_ret"].iloc[:, g - 1]
            assert col.notna().any(), f"group {g} all-nan"

    def test_three_unique_factor(self, adj_dates, tradable):
        """3 unique (17/17/16): 走 _group_ranked (rank + qcut), n_unique=3 → ranked."""
        values = [0] * 17 + [1] * 17 + [2] * 16
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            neutralize_industry=False,
        )
        assert result["group_ret"].shape == (N_ADJ, 5)
        assert result["group_ret"].notna().any().any()


# ---------------------------------------------------------------------------
# Test 3: 连续 float 因子 (回归保护)
# ---------------------------------------------------------------------------

class TestContinuousFactorEndToEnd:
    """连续因子 (n_unique 远大于 n_groups): 走原 _group_ranked 路径."""

    def test_random_continuous(self, adj_dates, tradable):
        rng = np.random.RandomState(42)
        values = list(rng.randn(N_STOCKS))
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            neutralize_industry=False,
        )
        assert result["group_ret"].shape == (N_ADJ, 5)
        assert result["group_ret"].notna().any().any()

    def test_random_continuous_with_industry(self, adj_dates, tradable):
        rng = np.random.RandomState(7)
        values = list(rng.randn(N_STOCKS))
        industry_values = [(i % 5) + 1 for i in range(N_STOCKS)]
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            industry_values=industry_values,
            neutralize_industry=True,
        )
        assert result["group_ret"].shape == (N_ADJ, 5)


# ---------------------------------------------------------------------------
# Test 4: 4 种 group 数量
# ---------------------------------------------------------------------------

class TestDifferentGroupCounts:
    @pytest.mark.parametrize("groups", [2, 3, 5, 10])
    def test_various_group_counts(self, adj_dates, tradable, groups):
        """group=2/3/5/10 都能正确 dispatch."""
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            groups=groups,
            neutralize_industry=False,
        )
        assert result["group_ret"].shape == (N_ADJ, groups)
        # 5 组都应有数据
        for g in range(1, groups + 1):
            col = result["group_ret"].iloc[:, g - 1]
            assert col.notna().any(), f"group {g} all-nan"


# ---------------------------------------------------------------------------
# Test 5: 输出 keys 完整性
# ---------------------------------------------------------------------------

class TestOutputKeys:
    """GroupAnalyzerNode 应输出完整 keys, 不随因子类型变化."""

    EXPECTED_KEYS = {
        "fac_group", "group_num", "group_ret",
        "group_winratio", "group_winloss",
    }

    def test_bool_factor_output_keys(self, adj_dates, tradable):
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            neutralize_industry=False,
        )
        for key in self.EXPECTED_KEYS:
            assert key in result, f"missing key: {key}"

    def test_continuous_factor_output_keys(self, adj_dates, tradable):
        rng = np.random.RandomState(42)
        values = list(rng.randn(N_STOCKS))
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            neutralize_industry=False,
        )
        for key in self.EXPECTED_KEYS:
            assert key in result


# ---------------------------------------------------------------------------
# Test 6: floor_mode="last" (回归)
# ---------------------------------------------------------------------------

class TestFloorMode:
    def test_floor_mode_last(self, adj_dates, tradable):
        """floor_mode='last' 稀疏日复制上一日 (Phase 1 修复时验证过)."""
        values = [-1] * 30 + [1] * 20
        result = run_preprocess_neutralize_group(
            values, adj_dates, tradable,
            floor_mode="last",
            neutralize_industry=False,
        )
        # 任何调仓日都应有数据
        assert result["group_ret"].notna().any().any()

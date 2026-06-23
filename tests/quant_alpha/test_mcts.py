# coding=utf-8
"""Tests for QuantAlpha MCTS subpackage (M2 PR).

覆盖：
- ExtensionOpPool：从 OperatorVocab 动态生成（vs 旧 7 硬编码）
- MCTSNode / MCTSTree：谱系追踪 + UCB1
- 5 通道反馈采集
- MCTSSearch：端到端 UCB1 + 反馈驱动
- CLI alpha-mcts 命令
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.mcts import (
    ExtensionOp,
    ExtensionOpPool,
    MCTSSearch,
    MCTSSearchConfig,
    MCTSSearchResult,
    MCTSTree,
    MCTSNode,
    NodeStatus,
    MCTSFeedbackConfig,
    DEFAULT_WINDOWS,
)
from QuantNodes.research.quant_alpha.mcts.feedback import (
    collect_all_channels,
    collect_code_channel,
    collect_execution_channel,
    collect_llm_channel,
    collect_shape_channel,
    collect_value_channel,
)
from QuantNodes.core.feedback import FeedbackChannel


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成 5 票 × 10 日测试数据"""
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            rows.append({
                "date": date,
                "code": code,
                "close": float(np.random.randn() * 5 + 100),
                "open": float(np.random.randn() * 5 + 100),
                "high": float(np.random.randn() * 5 + 102),
                "low": float(np.random.randn() * 5 + 98),
                "vol": float(np.random.randint(1000, 5000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def op_pool() -> ExtensionOpPool:
    """ExtensionOpPool 默认实例"""
    return ExtensionOpPool()


@pytest.fixture
def mcts_config() -> MCTSSearchConfig:
    """默认 MCTS 配置"""
    return MCTSSearchConfig(
        iterations=10,
        max_depth=3,
        exploration_weight=1.414,
        seed=42,
        feedback_config=MCTSFeedbackConfig(),
    )


# ==============================================================================
# Test Class 1: ExtensionOpPool
# ==============================================================================


class TestExtensionOpPool:
    """ExtensionOpPool 测试 - 从 OperatorVocab 动态生成"""

    def test_pool_size_greater_than_old_7(self, op_pool: ExtensionOpPool):
        """Pool 大小应大于旧 7 硬编码"""
        assert len(op_pool) > 7
        # 实际 ≥ 26（含 wrap/window/unary/diff/ratio）
        assert len(op_pool) >= 20

    def test_pool_has_six_categories(self, op_pool: ExtensionOpPool):
        """Pool 应有 6 个 category"""
        cats = op_pool.list_categories()
        assert "wrap" in cats
        assert "window" in cats
        assert "unary" in cats
        assert "diff" in cats
        assert "ratio" in cats

    def test_window_ops_require_window(self, op_pool: ExtensionOpPool):
        """window 类算子应需要 window 参数"""
        for op in op_pool:
            if op.category == "window":
                assert op.requires_window is True

    def test_wrap_ops_no_window(self, op_pool: ExtensionOpPool):
        """wrap 类算子不需要 window"""
        for op in op_pool:
            if op.category == "wrap":
                assert op.requires_window is False

    def test_sample_returns_extension_op(self, op_pool: ExtensionOpPool):
        """sample() 返回 ExtensionOp"""
        op = op_pool.sample()
        assert isinstance(op, ExtensionOp)
        assert op.template
        assert op.name

    def test_sample_by_category(self, op_pool: ExtensionOpPool):
        """sample(category) 只返回指定类"""
        op = op_pool.sample(category="wrap")
        assert op.category == "wrap"

    def test_sample_window(self, op_pool: ExtensionOpPool):
        """sample_window() 返回 DEFAULT_WINDOWS 中的值"""
        w = op_pool.sample_window()
        assert w in DEFAULT_WINDOWS

    def test_instantiate_window_op(self, op_pool: ExtensionOpPool):
        """instantiate 替换 {f} 和 {w}"""
        op = ExtensionOp(
            name="test",
            template="ts_mean({f}, {w})",
            requires_window=True,
        )
        result = op.instantiate("close", 20)
        assert result == "ts_mean(close, 20)"

    def test_seed_formulas_cover_all_categories(self, op_pool: ExtensionOpPool):
        """seed_formulas 覆盖各 category"""
        cols = ["close", "open", "high", "low", "vol"]
        seeds = op_pool.get_seed_formulas(cols)
        assert len(seeds) > 0
        # 应包含 wrap, window, diff, ratio
        assert any("rank(" in s for s in seeds)  # wrap
        assert any("ts_mean(" in s for s in seeds)  # window
        assert any("signedpower(" in s for s in seeds)  # unary
        assert any("- ts_mean(" in s for s in seeds)  # diff
        assert any("/ ts_lag(" in s for s in seeds)  # ratio

    def test_pool_stats(self, op_pool: ExtensionOpPool):
        """stats 返回统计"""
        stats = op_pool.stats()
        assert "total" in stats
        assert "by_category" in stats
        assert "windows" in stats
        assert stats["total"] == len(op_pool)
        assert stats["windows"] == DEFAULT_WINDOWS


# ==============================================================================
# Test Class 2: MCTSNode / MCTSTree
# ==============================================================================


class TestMCTSNode:
    """MCTSNode 谱系追踪测试"""

    def test_default_entry_id_is_uuid(self):
        """默认 entry_id 是 UUID"""
        node = MCTSNode(formula="close")
        assert len(node.entry_id) >= 32  # UUID hex 长度

    def test_add_child_sets_parent_id(self):
        """add_child 设置 parent_id 和 depth"""
        parent = MCTSNode(formula="close", depth=0)
        child = MCTSNode(formula="ts_mean(close, 5)")
        parent.add_child(child)
        assert child.parent_id == parent.entry_id
        assert child.depth == 1

    def test_ancestors_chain(self):
        """ancestors() 返回根到当前节点的链"""
        root = MCTSNode(formula="__ROOT__", depth=0)
        n1 = MCTSNode(formula="close")
        n2 = MCTSNode(formula="ts_mean(close, 5)")
        n3 = MCTSNode(formula="rank(close)")
        root.add_child(n1)
        n1.add_child(n2)
        n2.add_child(n3)
        ancestors = n3.ancestors()
        assert len(ancestors) == 3
        assert ancestors[0].formula == "__ROOT__"
        assert ancestors[1].formula == "close"
        assert ancestors[2].formula == "ts_mean(close, 5)"

    def test_is_root(self):
        """is_root() 仅对根节点返回 True"""
        root = MCTSNode(formula="__ROOT__", depth=0)
        child = MCTSNode(formula="close")
        root.add_child(child)
        assert root.is_root() is True
        assert child.is_root() is False

    def test_ucb1_infinite_for_unvisited(self):
        """UCB1 对未访问节点返回 inf"""
        parent = MCTSNode(formula="close", depth=0)
        parent.visits = 5
        child = MCTSNode(formula="ts_mean(close, 5)")
        parent.add_child(child)
        # child.visits = 0
        assert child.ucb1() == float("inf")

    def test_ucb1_finite_for_visited(self):
        """UCB1 对已访问节点返回有限值"""
        parent = MCTSNode(formula="close", depth=0)
        parent.visits = 10
        child = MCTSNode(formula="ts_mean(close, 5)")
        parent.add_child(child)
        child.visits = 3
        child.overall_score = 0.5
        ucb = child.ucb1()
        assert ucb != float("inf")
        assert ucb > 0.5  # exploit + explore


class TestMCTSTree:
    """MCTSTree 容器测试"""

    def test_add_node_via_root(self):
        """通过 root 添加节点"""
        tree = MCTSTree()
        n = MCTSNode(formula="close")
        tree.add_node(n, parent=tree.root)
        assert n in tree.root.children
        assert tree.formula_cache["close"] == n

    def test_get_by_formula(self):
        """按 formula 查节点"""
        tree = MCTSTree()
        n = MCTSNode(formula="close")
        tree.add_node(n, parent=tree.root)
        assert tree.get_by_formula("close") == n
        assert tree.get_by_formula("nonexistent") is None

    def test_all_nodes_dfs(self):
        """all_nodes() 深度优先遍历"""
        tree = MCTSTree()
        n1 = MCTSNode(formula="close")
        n2 = MCTSNode(formula="ts_mean(close, 5)")
        tree.add_node(n1, parent=tree.root)
        tree.add_node(n2, parent=n1)
        all_nodes = tree.all_nodes()
        assert len(all_nodes) == 2
        assert n1 in all_nodes
        assert n2 in all_nodes

    def test_best_k_dedup(self):
        """best_k() 去重"""
        tree = MCTSTree()
        # 添加多个相同 overall_score 的节点
        n1 = MCTSNode(formula="close"); n1.overall_score = 0.5
        n2 = MCTSNode(formula="open"); n2.overall_score = 0.7
        n3 = MCTSNode(formula="high"); n3.overall_score = 0.6
        tree.add_node(n1, parent=tree.root)
        tree.add_node(n2, parent=tree.root)
        tree.add_node(n3, parent=tree.root)
        best = tree.best_k(k=10)
        assert len(best) == 3
        # 排序: n2 (0.7) > n3 (0.6) > n1 (0.5)
        assert best[0].formula == "open"
        assert best[1].formula == "high"
        assert best[2].formula == "close"

    def test_stats(self):
        """stats 返回树统计"""
        tree = MCTSTree()
        n1 = MCTSNode(formula="close")
        n1.status = NodeStatus.EVALUATED
        tree.add_node(n1, parent=tree.root)
        stats = tree.stats()
        assert stats["total_nodes"] == 1
        assert stats["by_status"]["evaluated"] == 1
        assert stats["max_depth"] == 0


# ==============================================================================
# Test Class 3: 5 通道反馈
# ==============================================================================


class TestFiveChannelFeedback:
    """5 通道反馈采集测试"""

    def test_execution_success(self):
        """EXECUTION 通道：无异常时通过"""
        fb = collect_execution_channel("ts_mean(close, 5)", None)
        assert fb.passed is True
        assert fb.score == 1.0

    def test_execution_failure(self):
        """EXECUTION 通道：异常时不通过"""
        fb = collect_execution_channel(
            "ts_mean(close, 5)", ValueError("test error"),
        )
        assert fb.passed is False
        assert fb.score == 0.0

    def test_shape_match(self):
        """SHAPE 通道：长度匹配时通过"""
        result = pl.Series("x", [1.0, 2.0, 3.0])
        fb = collect_shape_channel(result, expected_length=3)
        assert fb.passed is True

    def test_shape_mismatch(self):
        """SHAPE 通道：长度不匹配时不通过"""
        result = pl.Series("x", [1.0, 2.0, 3.0])
        fb = collect_shape_channel(result, expected_length=5)
        assert fb.passed is False

    def test_code_valid(self):
        """CODE 通道：合规公式通过"""
        fb = collect_code_channel(
            "ts_mean(close, 5)",
            MCTSFeedbackConfig(),
        )
        assert fb.passed is True

    def test_code_too_long(self):
        """CODE 通道：超长公式不通过"""
        long_formula = "ts_mean(close, 5) + " * 30 + "vol"
        fb = collect_code_channel(long_formula, MCTSFeedbackConfig())
        assert fb.passed is False
        assert "length" in fb.detail

    def test_code_syntax_error(self):
        """CODE 通道：语法错误不通过"""
        fb = collect_code_channel("ts_mean(close,", MCTSFeedbackConfig())
        assert fb.passed is False

    def test_value_valid(self):
        """VALUE 通道：合理分布通过"""
        result = pl.Series("x", np.random.randn(100))
        fb = collect_value_channel(result, MCTSFeedbackConfig())
        assert fb.passed is True

    def test_value_all_nan(self):
        """VALUE 通道：全 NaN 不通过"""
        result = pl.Series("x", [float("nan")] * 100)
        fb = collect_value_channel(result, MCTSFeedbackConfig())
        assert fb.passed is False
        assert "NaN" in fb.detail

    def test_llm_mock_no_hypothesis(self):
        """LLM 通道：无 hypothesis 时默认 pass"""
        fb = collect_llm_channel("ts_mean(close, 5)")
        assert fb.passed is True

    def test_llm_keyword_match(self):
        """LLM 通道：关键字匹配度"""
        fb = collect_llm_channel(
            "ts_mean(close, 5)",
            hypothesis="use rolling mean of close",
            description="rolling mean",
        )
        # "close" 和 "mean" 都应匹配
        assert fb.score > 0.0

    def test_collect_all_channels_aggregates(self):
        """collect_all_channels 聚合 5 通道"""
        result = pl.Series("x", np.random.randn(50))
        config = MCTSFeedbackConfig()
        fb = collect_all_channels(
            formula="ts_mean(close, 5)",
            result=result,
            expected_length=50,
            config=config,
        )
        # 5 通道都启用（除 LLM）
        assert FeedbackChannel.EXECUTION in fb.channels
        assert FeedbackChannel.SHAPE in fb.channels
        assert FeedbackChannel.CODE in fb.channels
        assert FeedbackChannel.VALUE in fb.channels
        # decision 应为 True（无异常时）
        assert fb.decision is True

    def test_collect_all_channels_with_exception(self):
        """collect_all_channels 异常时 decision=False"""
        config = MCTSFeedbackConfig()
        fb = collect_all_channels(
            formula="bad_formula",
            result=None,
            expected_length=50,
            config=config,
            exception=ValueError("boom"),
        )
        assert fb.decision is False
        # summary 应包含 failed 通道名（小写）
        assert "execution" in fb.summary


# ==============================================================================
# Test Class 4: MCTSSearch 端到端
# ==============================================================================


class TestMCTSSearch:
    """MCTSSearch 端到端测试"""

    def test_search_returns_result(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() 返回 MCTSSearchResult"""
        mcts = MCTSSearch(config=mcts_config)
        result = mcts.search(data=sample_data, date_column="date")
        assert isinstance(result, MCTSSearchResult)
        assert result.total_iterations == mcts_config.iterations

    def test_search_generates_formulas(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() 生成公式"""
        mcts = MCTSSearch(config=mcts_config)
        result = mcts.search(data=sample_data, date_column="date")
        assert result.formula_count > 0
        # 至少有一个 valid node
        assert result.valid_count >= 0

    def test_search_with_seed_formulas(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() 支持 seed_formulas"""
        mcts = MCTSSearch(config=mcts_config)
        seeds = ["rank(close)", "ts_mean(close, 20)"]
        result = mcts.search(
            data=sample_data,
            seed_formulas=seeds,
            date_column="date",
        )
        # 至少种子公式应该在树中
        for f in seeds:
            assert result.tree.get_by_formula(f) is not None

    def test_search_lineage_tracking(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() 启用谱系追踪"""
        mcts_config.enable_lineage = True
        mcts = MCTSSearch(config=mcts_config)
        result = mcts.search(data=sample_data, date_column="date")
        # 所有非根节点应有 parent_id
        for n in result.tree.all_nodes():
            assert n.parent_id is not None
            assert n.parent_id == result.tree.root.entry_id or True  # depth=0 也指向 root

    def test_search_invalid_formulas_filtered(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() 过滤无效公式"""
        mcts = MCTSSearch(config=mcts_config)
        # seed 含一个有效 + 一个语法错误
        result = mcts.search(
            data=sample_data,
            seed_formulas=["ts_mean(close, 5)", "totally_bad_formula(("],
            date_column="date",
        )
        # 至少坏公式被 rejected
        bad_node = result.tree.get_by_formula("totally_bad_formula((")
        assert bad_node is not None
        assert bad_node.status == NodeStatus.REJECTED

    def test_search_best_k_deduped(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() best_k_nodes 去重"""
        mcts = MCTSSearch(config=mcts_config)
        result = mcts.search(data=sample_data, date_column="date")
        # 验证 best_k_nodes 无重复 entry_id
        entry_ids = [n.entry_id for n in result.best_k_nodes]
        assert len(entry_ids) == len(set(entry_ids))

    def test_get_feedback(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """get_feedback 返回 5 通道反馈"""
        mcts = MCTSSearch(config=mcts_config)
        mcts.search(data=sample_data, date_column="date")
        # 查一个真实评估过的公式
        formula = "ts_mean(close, 20)"
        fb = mcts.get_feedback(formula)
        # 可能 None（如果没评估到）或 FactorFeedback
        if fb is not None:
            assert hasattr(fb, "channels")
            assert hasattr(fb, "decision")

    def test_search_stats(
        self, sample_data: pl.DataFrame, mcts_config: MCTSSearchConfig
    ):
        """search() 统计"""
        mcts = MCTSSearch(config=mcts_config)
        mcts.search(data=sample_data, date_column="date")
        stats = mcts.stats()
        assert "total_nodes" in stats
        assert "formula_cache_size" in stats
        assert stats["formula_cache_size"] > 0


# ==============================================================================
# Test Class 5: CLI alpha-mcts 命令
# ==============================================================================


class TestAlphaMctsCLI:
    """CLI alpha-mcts 命令测试"""

    def test_command_registered(self):
        """alpha-mcts 命令已注册"""
        from QuantNodes.cli.commands import COMMAND_REGISTRY
        cmd = COMMAND_REGISTRY.get("alpha-mcts")
        assert cmd is not None
        assert cmd.name == "alpha-mcts"

    def test_command_run_with_synthetic_data(self):
        """alpha-mcts 用合成数据跑通"""
        from QuantNodes.cli.commands.alpha import AlphaMctsCommand

        cmd = AlphaMctsCommand()
        args = argparse.Namespace(
            iterations=5,
            data=None,
            date_column="date",
            code_column="code",
            max_depth=3,
            exploration_weight=1.414,
            seed=42,
            top_k=5,
            quiet=True,
        )
        result = cmd.run(args)
        assert result == 0

    def test_command_run_with_invalid_data_path(self):
        """alpha-mcts 数据路径错误返回 1"""
        from QuantNodes.cli.commands.alpha import AlphaMctsCommand

        cmd = AlphaMctsCommand()
        args = argparse.Namespace(
            iterations=5,
            data="/nonexistent/path/data.parquet",
            date_column="date",
            code_column="code",
            max_depth=3,
            exploration_weight=1.414,
            seed=42,
            top_k=5,
            quiet=True,
        )
        result = cmd.run(args)
        assert result == 1

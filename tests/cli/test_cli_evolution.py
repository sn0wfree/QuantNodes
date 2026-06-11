"""QuantNodes CLI 演化命令测试。"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from QuantNodes.cli import (
    cmd_evolve,
    cmd_factor_best,
    cmd_factor_info,
    cmd_help,
)
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# 1. cmd_help
# ============================================================================

def test_cmd_help_mentions_evolution():
    """help 文本提及新命令。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_help(None)
    out = buf.getvalue()
    assert "evolve" in out
    assert "factor-info" in out
    assert "factor-best" in out


# ============================================================================
# 2. cmd_factor_info
# ============================================================================

def test_factor_info_empty_pool(tmp_path):
    """空 pool 也能正常显示 (不报错)。"""
    class Args:
        pool_dir = str(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_info(Args())
    assert rc == 0
    out = buf.getvalue()
    assert "size: 0" in out
    assert "passed: 0 / 0" in out


def test_factor_info_with_entries(tmp_path):
    """填充后正确显示 by_round / by_operation / passed。"""
    pool = TrajectoryPool(tmp_path)
    for i, (op, r, sharpe) in enumerate([
        ("original", 0, 1.0),
        ("original", 0, 1.5),
        ("mutation", 1, 2.0),
        ("crossover", 2, 1.8),
        ("original", 0, 0.0),  # failed
    ]):
        pool.add(TrajectoryEntry(
            entry_id=f"e{i}",
            round_idx=r,
            operation=op,
            feedback=FactorFeedback(
                factor_name=f"f{i}",
                decision=sharpe > 0,
                summary="ok" if sharpe > 0 else "fail",
            ),
            metrics={"sharpe": sharpe},
        ))

    class Args:
        pool_dir = str(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_info(Args())
    assert rc == 0
    out = buf.getvalue()
    assert "size: 5" in out
    assert "by_round: {0: 3, 1: 1, 2: 1}" in out
    assert "by_operation: {'original': 3, 'mutation': 1, 'crossover': 1}" in out
    assert "passed: 4 / 5" in out


def test_factor_info_missing_dir():
    """pool 目录不存在 → exit 1。"""
    class Args:
        pool_dir = "/nonexistent/path/xyz"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_info(Args())
    assert rc == 1


# ============================================================================
# 3. cmd_factor_best
# ============================================================================

def test_factor_best_orders_by_metric(tmp_path):
    """factor-best 按 metric 降序输出。"""
    pool = TrajectoryPool(tmp_path)
    for i, sharpe in enumerate([0.5, 2.0, 1.5, 0.8, 1.2]):
        pool.add(TrajectoryEntry(
            entry_id=f"e{i}",
            feedback=FactorFeedback(factor_name=f"f{i}", decision=True, summary="ok"),
            metrics={"sharpe": sharpe},
        ))

    class Args:
        pool_dir = str(tmp_path)
        top = 3
        metric = "sharpe"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_best(Args())
    assert rc == 0
    out = buf.getvalue()
    # Top 3 应该是 f1(2.0), f2(1.5), f4(1.2)
    pos_f1 = out.find("f1 [")
    pos_f2 = out.find("f2 [")
    pos_f4 = out.find("f4 [")
    pos_f3 = out.find("f3 [")
    assert pos_f1 < pos_f2 < pos_f4  # 顺序正确
    assert pos_f3 == -1  # top=3, 不显示 f3(0.8)


def test_factor_best_custom_metric(tmp_path):
    """自定义 metric (arr / ic_mean / 等)。"""
    pool = TrajectoryPool(tmp_path)
    for i, arr in enumerate([0.05, 0.20, 0.15]):
        pool.add(TrajectoryEntry(
            entry_id=f"e{i}",
            feedback=FactorFeedback(factor_name=f"f{i}", decision=True, summary="ok"),
            metrics={"sharpe": 0, "arr": arr},
        ))

    class Args:
        pool_dir = str(tmp_path)
        top = 2
        metric = "arr"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_best(Args())
    assert rc == 0
    out = buf.getvalue()
    pos_f1 = out.find("f1 [")
    pos_f2 = out.find("f2 [")
    assert pos_f1 < pos_f2  # arr=0.20 排第一


# ============================================================================
# 4. cmd_evolve (mock 数据, 验证 CLI 参数解析)
# ============================================================================

def test_evolve_missing_config():
    """缺 --config 仍能调用, 但缺文件应返回 1。"""
    class Args:
        config = "/nonexistent/config.yaml"
        directions = ""
        initial_json = None
        max_rounds = None
        early_stop = None
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_evolve(Args())
    assert rc == 1
    assert "配置文件不存在" in buf.getvalue()


def test_evolve_initial_json_parse_error():
    """--initial-json 解析失败 → exit 1。"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("""
factor:
  name: test
  factor_dir: x.h5
preprocess:
  adj_date_beg: 20240101
  adj_date_end: 20240301
evolution:
  enabled: false
""")
        cfg_path = f.name
    try:
        class Args:
            config = cfg_path
            directions = ""
            initial_json = "not json"
            max_rounds = None
            early_stop = None
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_evolve(Args())
        assert rc == 1
        assert "解析失败" in buf.getvalue()
    finally:
        Path(cfg_path).unlink()


def test_evolve_disabled_config_returns_error():
    """config.evolution.enabled=False → 演化失败。"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("""
factor:
  name: test
  factor_dir: x.h5
preprocess:
  adj_date_beg: 20240101
  adj_date_end: 20240301
evolution:
  enabled: false
""")
        cfg_path = f.name
    try:
        class Args:
            config = cfg_path
            directions = "momentum"
            initial_json = None
            max_rounds = None
            early_stop = None
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_evolve(Args())
        # evolution.enabled=False → run_evolution 抛 ValueError
        assert rc == 1
        assert "演化失败" in buf.getvalue()
    finally:
        Path(cfg_path).unlink()

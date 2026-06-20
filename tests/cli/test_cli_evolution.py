"""QuantNodes CLI 演化命令测试。"""
from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


from QuantNodes.cli import (
    cmd_evolve,
    cmd_factor_best,
    cmd_factor_info,
    cmd_factor_rag_show,
    cmd_factor_visual,
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
    assert "factor-visual" in out
    assert "factor-rag-show" in out


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


# ============================================================================
# 5. cmd_factor_visual (Week 6)
# ============================================================================

def _populate_pool_with_entries(pool: TrajectoryPool, n: int = 3) -> None:
    for i in range(n):
        pool.add(TrajectoryEntry(
            entry_id=f"e{i}",
            round_idx=i % 2,
            operation="original" if i == 0 else "mutation",
            feedback=FactorFeedback(factor_name=f"f{i}", decision=True, summary="ok"),
            metrics={"sharpe": 0.5 + i * 0.3},
        ))


def test_factor_visual_writes_html(tmp_path):
    """factor-visual 生成 HTML 报告。"""
    pool = TrajectoryPool(tmp_path / "pool")
    _populate_pool_with_entries(pool)
    out_html = tmp_path / "report.html"
    pool_dir = str(tmp_path / "pool")

    class Args:
        pass
    args = Args()
    args.pool_dir = pool_dir
    args.output = str(out_html)
    args.metric = "sharpe"
    args.title = "Test Report"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_visual(args)
    assert rc == 0
    assert out_html.exists()
    assert out_html.stat().st_size > 1000
    out = buf.getvalue()
    assert "HTML 报告已生成" in out


def test_factor_visual_empty_pool(tmp_path):
    """空 pool → exit 1。"""
    pool = TrajectoryPool(tmp_path / "pool")  # 空
    class Args:
        pass
    args = Args()
    args.pool_dir = str(tmp_path / "pool")
    args.output = None
    args.metric = "sharpe"
    args.title = None
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_visual(args)
    assert rc == 1
    assert "pool 为空" in buf.getvalue()


def test_factor_visual_missing_dir():
    """pool 目录不存在 → exit 1。"""
    class Args:
        pass
    args = Args()
    args.pool_dir = "/nonexistent/path/xyz"
    args.output = None
    args.metric = "sharpe"
    args.title = None
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_visual(args)
    assert rc == 1
    assert "pool 目录不存在" in buf.getvalue()


def test_factor_visual_default_output_path(tmp_path):
    """output=None 时默认写到 <pool-dir>_report.html。"""
    pool_dir_path = tmp_path / "mypool"
    pool = TrajectoryPool(pool_dir_path)
    _populate_pool_with_entries(pool)

    class Args:
        pass
    args = Args()
    args.pool_dir = str(pool_dir_path)
    args.output = None
    args.metric = "sharpe"
    args.title = None
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_visual(args)
    assert rc == 0
    # 默认: <tmp_path>/mypool_report.html
    expected = tmp_path / "mypool_report.html"
    assert expected.exists()


# ============================================================================
# 6. cmd_factor_rag_show (Week 7)
# ============================================================================

def _populate_pool_with_factors(pool: TrajectoryPool) -> None:
    """填充带 hypothesis/description 的 entry 用于 RAG 检索。"""
    pool.add(TrajectoryEntry(
        entry_id="e1", round_idx=0, operation="original",
        config_snapshot={"factor": {
            "name": "momentum_20d",
            "expression": "(close-close.shift(20))/close.shift(20)",
            "hypothesis": "momentum effect",
            "description": "20-day price momentum factor",
        }},
        feedback=FactorFeedback(factor_name="momentum_20d", decision=True, summary="ok"),
        metrics={"sharpe": 1.5},
    ))
    pool.add(TrajectoryEntry(
        entry_id="e2", round_idx=0, operation="original",
        config_snapshot={"factor": {
            "name": "reversal_5d",
            "expression": "close - close.shift(5)",
            "hypothesis": "reversal effect",
            "description": "5-day mean reversal factor",
        }},
        feedback=FactorFeedback(factor_name="reversal_5d", decision=True, summary="ok"),
        metrics={"sharpe": 1.0},
    ))


def test_factor_rag_show_basic(tmp_path):
    """factor-rag-show 检索并显示 Top-K。"""
    pool = TrajectoryPool(tmp_path / "pool")
    _populate_pool_with_factors(pool)

    class Args:
        pass
    args = Args()
    args.pool_dir = str(tmp_path / "pool")
    args.query = "momentum effect"
    args.top = 2
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_rag_show(args)
    assert rc == 0
    out = buf.getvalue()
    assert "Top 2" in out
    assert "momentum_20d" in out
    assert "score=" in out


def test_factor_rag_show_missing_dir():
    """pool 目录不存在 → exit 1。"""
    class Args:
        pass
    args = Args()
    args.pool_dir = "/nonexistent/path/xyz"
    args.query = "anything"
    args.top = 5
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_rag_show(args)
    assert rc == 1
    assert "pool 目录不存在" in buf.getvalue()


def test_factor_rag_show_empty_pool(tmp_path):
    """空 pool → exit 1。"""
    pool = TrajectoryPool(tmp_path / "pool")  # 空
    class Args:
        pass
    args = Args()
    args.pool_dir = str(tmp_path / "pool")
    args.query = "momentum"
    args.top = 5
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_rag_show(args)
    assert rc == 1
    assert "pool 为空" in buf.getvalue()


def test_factor_rag_show_no_match(tmp_path):
    """query 不匹配时返回 0 (但仍 exit 0)。"""
    pool = TrajectoryPool(tmp_path / "pool")
    _populate_pool_with_factors(pool)
    class Args:
        pass
    args = Args()
    args.pool_dir = str(tmp_path / "pool")
    args.query = "xyz_zzz_unrelated_quantum_xyzzy"
    args.top = 5
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_rag_show(args)
    assert rc == 0
    assert "无匹配结果" in buf.getvalue()

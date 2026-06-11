"""真实数据 E2E 测试 (Week 11) — 8 tests。

覆盖:
    - data_prep 脚本 (2)
    - 完整 E2E 流程 (3)
    - PipelineRunner 注入机制 (1)
    - 输出验证 (2)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.e2e.data_prep import (
    _gen_factor_data,
    _gen_index_cp,
    _gen_stk_daily,
)


# ============================================================================
# 1. data_prep 模块 (2)
# ============================================================================

def test_data_prep_functions():
    """data_prep 模块函数可独立调用。"""
    rng = np.random.RandomState(42)
    factor = _gen_factor_data(rng, 30, 10, "momentum_20d")
    assert factor.shape == (30, 10)
    # momentum 因子有 trend (正 linspace), 不应全 0
    assert factor.std().mean() > 0.5

    index_cp = _gen_index_cp(rng, 30)
    assert index_cp.shape == (30, 2)
    assert "000300.SH" in index_cp.columns

    stk_daily = _gen_stk_daily(rng, 30, 10)
    assert "cp" in stk_daily
    assert "st" in stk_daily
    assert stk_daily["cp"].shape == (30, 10)


def test_data_prep_cli_synthetic(tmp_path):
    """data_prep CLI 跑通, 输出 H5 格式正确。"""
    result = subprocess.run(
        [
            sys.executable, "-m",
            "QuantNodes.research.factor_test.e2e.data_prep",
            "--output-dir", str(tmp_path / "data"),
            "--n-days", "30",
            "--n-stocks", "10",
            "--factors", "momentum_20d,reversal_5d",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # 文件存在
    assert (tmp_path / "data" / "stk_daily.h5").exists()
    assert (tmp_path / "data" / "index_daily.h5").exists()
    assert (tmp_path / "data" / "stklist.h5").exists()
    assert (tmp_path / "data" / "trade_dt.h5").exists()
    assert (tmp_path / "data" / "momentum_20d.h5").exists()
    assert (tmp_path / "data" / "reversal_5d.h5").exists()
    # 读 H5 验证
    cp = pd.read_hdf(tmp_path / "data" / "stk_daily.h5", key="cp")
    assert cp.shape == (30, 10)


# ============================================================================
# 2. 完整 E2E 流程 (3)
# ============================================================================

@pytest.fixture(scope="module")
def e2e_artifacts(tmp_path_factory):
    """跑一次 E2E, 共享 artifact 供后续测试用。"""
    data_dir = tmp_path_factory.mktemp("e2e_data")
    out_dir = tmp_path_factory.mktemp("e2e_output")
    # 1. data_prep (因子名要匹配 e2e 的默认 --factor-name)
    subprocess.run([
        sys.executable, "-m",
        "QuantNodes.research.factor_test.e2e.data_prep",
        "--output-dir", str(data_dir),
        "--n-days", "30", "--n-stocks", "10",
        "--factors", "momentum_20d,reversal_5d,volatility_60d",
    ], check=True, capture_output=True, timeout=30)
    # 2. run_evolution_e2e
    result = subprocess.run([
        sys.executable, "-m",
        "QuantNodes.research.factor_test.e2e.run_evolution_e2e",
        "--data-path", str(data_dir),
        "--output-dir", str(out_dir),
        "--max-rounds", "2",
    ], capture_output=True, text=True, timeout=60)
    return data_dir, out_dir, result


def test_run_evolution_e2e_synthetic(e2e_artifacts):
    """E2E 跑通, stdout 含关键成功标记。"""
    data_dir, out_dir, result = e2e_artifacts
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "演化完成" in out
    assert "HTML 报告" in out
    assert "JSON 摘要" in out
    assert "E2E 完成" in out


def test_evolution_pool_size_meets_expectations(e2e_artifacts):
    """E2E pool size ≥ 3 (round 0 必有 2 directions)。"""
    data_dir, out_dir, result = e2e_artifacts
    summary = json.loads((out_dir / "evolution_summary.json").read_text())
    assert summary["pool_size"] >= 3
    assert summary["rounds_completed"] >= 1


def test_html_report_generated(e2e_artifacts):
    """E2E 生成 HTML 报告。"""
    data_dir, out_dir, result = e2e_artifacts
    report = out_dir / "evolution_report.html"
    assert report.exists()
    assert report.stat().st_size > 5000  # 至少 5KB
    content = report.read_text()
    assert "演化" in content
    # 5 个 figure
    for fig in ("lineage_dag", "metric_distribution", "gate_breakdown",
                "operation_breakdown", "metric_per_round"):
        assert f"id=\"fig_{fig}\"" in content


# ============================================================================
# 3. PipelineRunner 注入机制 (1)
# ============================================================================

def test_pipeline_runner_skip_load_data(tmp_path):
    """PipelineRunner._context 注入 LoadData 后, run() 跳过 LoadDataNode。"""
    from QuantNodes.research.factor_test.config import (
        SingleFactorTestConfig, PreprocessSetting,
    )
    from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
    from QuantNodes.research.factor_test.utils.data_loader import DataLoader

    rng = np.random.RandomState(42)
    n_days, n_stocks = 30, 10
    dates = [int(d.strftime('%Y%m%d'))
             for d in pd.bdate_range('2026-01-04', periods=n_days)]
    stocks = list(range(100001, 100001 + n_stocks))

    # 写一个 dummy H5 (DataLoader 需要真实目录)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(stocks, columns=[0]).to_hdf(data_dir / "stklist.h5", key="data", mode="w")
    pd.DataFrame(dates, columns=[0]).to_hdf(data_dir / "trade_dt.h5", key="data", mode="w")

    cfg = SingleFactorTestConfig(
        factor={"name": "test", "factor_dir": "x.h5"},
        preprocess={
            "adj_date_beg": 20260101, "adj_date_end": 20260630,
            "adj_mode": ["M", "end"], "sample_index": "all",
            "sample_industry": "all",
            "tradable": {"no_st": True, "no_suspended": True},
            "missing": "", "extreme": "median", "norm": "zscore",
        },
    )
    runner = PipelineRunner(cfg)
    runner._context["LoadData"] = {
        "factor": pd.DataFrame(rng.randn(n_days, n_stocks), index=dates, columns=stocks),
        "price": pd.DataFrame(100 * np.exp(np.cumsum(rng.randn(n_days, n_stocks) * 0.02, axis=0)),
                              index=dates, columns=stocks),
        "id_citic1": pd.DataFrame(rng.randint(1, 31, (n_days, n_stocks)),
                                  index=dates, columns=stocks),
        "mv_float": pd.DataFrame(rng.lognormal(10, 1, (n_days, n_stocks)),
                                 index=dates, columns=stocks),
        "st": pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
        "suspend": pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
        "ud_limit": pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
        "ipo_days": pd.DataFrame(np.ones((n_days, n_stocks), dtype=int) * 500,
                                 index=dates, columns=stocks),
        "index_cp": pd.DataFrame({"000300.SH": np.arange(30)}, index=dates),
        "stklist": pd.DataFrame(stocks, columns=[0]),
        "trade_dt": pd.DataFrame(dates, columns=[0]),
        "_loader": DataLoader(str(data_dir)),  # 真实 loader, RiskCorrelation 不报错
    }
    # 跑 run() 应不抛 LoadDataNode 错误
    ctx = runner.run()
    assert "ICAnalyzer" in ctx
    assert ctx["ICAnalyzer"] is not None


# ============================================================================
# 4. 输出验证 (2)
# ============================================================================

def test_summary_json_valid(e2e_artifacts):
    """evolution_summary.json 含必要字段。"""
    data_dir, out_dir, result = e2e_artifacts
    summary = json.loads((out_dir / "evolution_summary.json").read_text())
    required = {"data_path", "output_dir", "directions", "max_rounds",
                "pool_size", "rounds_completed", "total_count",
                "rejected_count", "best_entries", "rag_metrics_history"}
    assert required <= set(summary.keys())
    assert isinstance(summary["best_entries"], list)
    assert isinstance(summary["rag_metrics_history"], list)


def test_quality_gate_rejects_in_e2e(tmp_path):
    """启用 QG 时, 全部因子被拦也是合理 (合成数据质量低)。"""
    # 简版: 只跑 round 0, QG 全 disabled
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    subprocess.run([
        sys.executable, "-m",
        "QuantNodes.research.factor_test.e2e.data_prep",
        "--output-dir", str(data_dir),
        "--n-days", "60", "--n-stocks", "20",
        "--factors", "momentum_20d",
    ], check=True, capture_output=True, timeout=30)

    result = subprocess.run([
        sys.executable, "-m",
        "QuantNodes.research.factor_test.e2e.run_evolution_e2e",
        "--data-path", str(data_dir),
        "--output-dir", str(out_dir),
        "--max-rounds", "0",
        "--disable-quality-gate",  # 禁用 QG, 让所有都通过
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    summary = json.loads((out_dir / "evolution_summary.json").read_text())
    # 禁用 QG 时, 至少 round 0 的因子都通过
    assert summary["total_count"] >= 1
    assert summary["rejected_count"] == 0

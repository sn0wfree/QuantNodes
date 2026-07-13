# coding=utf-8
"""Tests for validation.py (抗过拟合检验)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.backtest import BacktestConfig
from QuantNodes.strategy.momentum_etf_rotation.portfolio import (
    DiversificationCaps,
    RotationConfig,
)
from QuantNodes.strategy.momentum_etf_rotation.universe import DEFAULT_POOL
from QuantNodes.strategy.momentum_etf_rotation.validation import (
    ValidationConfig,
    ValidationReport,
    ablation,
    run_full_validation,
    validate_parameter_perturbation,
    validate_rebalance_offsets,
    validate_starting_points,
)


def _make_nav(n_days: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n_days)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.Series(prices, index=idx, name="nav")


def _make_panel(n_days: int = 1200, n_codes: int = 10, seed: int = 42) -> pd.DataFrame:
    """合成 ETF 面板 (mock for validation testing)."""
    rng = np.random.default_rng(seed)
    codes = [m.code for m in DEFAULT_POOL.members[:n_codes]]
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, (n_days, n_codes))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=codes)


def _small_pool(n_codes: int = 5):
    """小池 (含足够类别) 用于快速测试."""
    from QuantNodes.strategy.momentum_etf_rotation.universe import (
        Category, ETFMeta, ETFPool,
    )
    members = tuple(
        ETFMeta(
            code=f"E{i:03d}", name=f"E{i:03d}",
            category=Category.A_BROAD, index_code=f"I{i % 3}",
            liquidity_rank=i % 3 + 1,
        )
        for i in range(n_codes)
    )
    return ETFPool(members=members)


class TestValidationConfig:
    def test_default_values(self) -> None:
        """默认值应来自 validation_fix_report.md."""
        vcfg = ValidationConfig()
        assert len(vcfg.start_points) == 4
        assert "2019-01-01" in vcfg.start_points
        assert vcfg.calmar_cv_threshold == 0.25
        assert vcfg.perturb_lookbacks == (80, 100, 120)
        assert vcfg.min_calmar == 0.4


class TestValidateStartingPoints:
    def test_returns_validation_result(self) -> None:
        """应返回 ValidationResult 含 name/passed/summary/table."""
        panel = _make_panel(n_days=1500)
        cfg = BacktestConfig(rotation=RotationConfig(
            lookback=60, top_n=5, min_history=60,
            diversification=DiversificationCaps(
                a_share=5, a_share_broad=2, a_share_sector=2,
                require_commodity=False, require_overseas=False,
            ),
        ))
        pool = _small_pool(5)
        result = validate_starting_points(panel, pool, cfg)
        assert result.name == "起点依赖"
        # passed 应为 bool 类型
        passed = result.passed
        assert passed in (True, False), f"passed = {passed} ({type(passed)})"
        assert isinstance(result.table, pd.DataFrame)
        assert len(result.table) == 4  # 4 个起点
        assert "Calmar" in result.table.columns


class TestValidateRebalanceOffsets:
    def test_returns_validation_result(self) -> None:
        """应返回 ValidationResult, 5 个偏移 (当前实现 CV=0 placeholder)."""
        panel = _make_panel(n_days=500)
        cfg = BacktestConfig(rotation=RotationConfig(
            lookback=60, top_n=3,
            diversification=DiversificationCaps(
                a_share=5, a_share_broad=2, a_share_sector=2,
                require_commodity=False, require_overseas=False,
            ),
        ))
        pool = _small_pool(5)
        result = validate_rebalance_offsets(panel, pool, cfg)
        assert result.name == "调仓日偏移"
        assert result.passed in (True, False)
        assert len(result.table) == 5


class TestValidateParameterPerturbation:
    def test_returns_validation_result(self) -> None:
        """应返回 ValidationResult 含 3 类扰动."""
        panel = _make_panel(n_days=1000)
        cfg = BacktestConfig(rotation=RotationConfig(
            lookback=90, top_n=3,
            diversification=DiversificationCaps(
                a_share=5, a_share_broad=2, a_share_sector=2,
                require_commodity=False, require_overseas=False,
            ),
        ))
        pool = _small_pool(5)
        result = validate_parameter_perturbation(panel, pool, cfg)
        assert result.name == "参数扰动"
        assert result.passed in (True, False)
        assert "扰动" in result.table.columns
        assert "值" in result.table.columns
        assert "Calmar" in result.table.columns


class TestAblation:
    def test_returns_validation_result(self) -> None:
        """应返回 ValidationResult 含 5 行 (基线 + 4 规则)."""
        panel = _make_panel(n_days=1000)
        cfg = BacktestConfig(rotation=RotationConfig(
            lookback=60, top_n=3,
            diversification=DiversificationCaps(
                a_share=5, a_share_broad=2, a_share_sector=2,
                require_commodity=False, require_overseas=False,
            ),
        ))
        pool = _small_pool(5)
        result = ablation(panel, pool, cfg)
        assert result.name == "消融实验"
        assert result.passed in (True, False)
        assert len(result.table) == 5  # 基线 + 4 规则


class TestRunFullValidation:
    def test_returns_report_with_4_actions(self) -> None:
        """应返回 ValidationReport 含 4 个 action."""
        panel = _make_panel(n_days=1000)
        cfg = BacktestConfig(rotation=RotationConfig(
            lookback=60, top_n=3,
            diversification=DiversificationCaps(
                a_share=5, a_share_broad=2, a_share_sector=2,
                require_commodity=False, require_overseas=False,
            ),
        ))
        pool = _small_pool(5)
        report = run_full_validation(panel, pool, cfg)
        assert isinstance(report, ValidationReport)
        assert len(report.actions) == 4
        assert report.passed + report.failed == 4

    def test_report_markdown(self) -> None:
        """应生成包含 4 个 action 的 markdown."""
        panel = _make_panel(n_days=800)
        cfg = BacktestConfig(rotation=RotationConfig(
            lookback=60, top_n=3,
            diversification=DiversificationCaps(
                a_share=5, a_share_broad=2, a_share_sector=2,
                require_commodity=False, require_overseas=False,
            ),
        ))
        pool = _small_pool(5)
        report = run_full_validation(panel, pool, cfg)
        md = report.to_markdown()
        assert "起点依赖" in md
        assert "调仓日偏移" in md
        assert "参数扰动" in md
        assert "消融实验" in md
        assert "总结" in md
        # report.markdown 字段应同步
        assert report.markdown == md
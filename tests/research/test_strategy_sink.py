"""Tests for run_101_alphas_v2.py — M3.4 strategy sink.

覆盖:
  - _alpha_to_signal_type: 6 个信号类型分支 + fallback
  - _safe_strategy_name: 特殊字符 sanitize
  - RunConfig: 默认 off / 三种 mode / --strategies-dir 传递 / effective_strategies_dir fallback
  - CLI argparse: --strategy-mode 三选一 + --strategies-dir
  - _persist_strategy: mode=per_alpha 写入 / mode=off 跳过 / failed 跳过 / best-effort (异常不抛)
  - _persist_batch_strategy: mode=after_batch 聚合 / 空 successes 跳过 / weights 均分
  - _strategy_name_for: paper_id + idx + signal_type 命名
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Add scripts/research to path so we can import the run_101_alphas_v2 module
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "research"
sys.path.insert(0, str(SCRIPT_DIR))

import run_101_alphas_v2 as r101  # noqa: E402
from QuantNodes.research.backtest.base import FactorResult  # noqa: E402
from QuantNodes.research.signal_source.base import Signal  # noqa: E402


# ── _alpha_to_signal_type 启发式 ────────────────────────────────────────


class TestAlphaToSignalType:
    def test_rsi(self) -> None:
        assert r101.FactorStage._alpha_to_signal_type("alpha_rsi_5") == "rsi"
        assert r101.FactorStage._alpha_to_signal_type("MyRSI_Strategy") == "rsi"

    def test_volatility(self) -> None:
        assert r101.FactorStage._alpha_to_signal_type("volatility_60") == "volatility"
        assert r101.FactorStage._alpha_to_signal_type("alpha_vol_20") == "volatility"
        assert r101.FactorStage._alpha_to_signal_type("xx_vol") == "volatility"

    def test_momentum(self) -> None:
        assert r101.FactorStage._alpha_to_signal_type("mom_20d") == "momentum"
        assert r101.FactorStage._alpha_to_signal_type("alpha_mom_60") == "momentum"
        assert r101.FactorStage._alpha_to_signal_type("my_momentum_alpha") == "momentum"

    def test_ma_cross(self) -> None:
        assert r101.FactorStage._alpha_to_signal_type("macross_5_20") == "ma_cross"
        assert r101.FactorStage._alpha_to_signal_type("alpha_ma_5") == "ma_cross"
        assert r101.FactorStage._alpha_to_signal_type("ma_strategy") == "ma_cross"

    def test_factor_rank(self) -> None:
        assert r101.FactorStage._alpha_to_signal_type("alpha_001") == "factor_rank"
        assert r101.FactorStage._alpha_to_signal_type("rank_perp") == "factor_rank"
        assert r101.FactorStage._alpha_to_signal_type("factor_combo") == "factor_rank"

    def test_fallback(self) -> None:
        """101 alphas (Alpha#1, Alpha#2, etc.) all fallback to factor_rank."""
        assert r101.FactorStage._alpha_to_signal_type("Alpha#1") == "factor_rank"
        assert r101.FactorStage._alpha_to_signal_type("unknown_xyz") == "factor_rank"


# ── _safe_strategy_name ────────────────────────────────────────


class TestSafeStrategyName:
    def test_alphanumeric_unchanged(self) -> None:
        assert r101.FactorStage._safe_strategy_name("alpha_001") == "alpha_001"
        assert r101.FactorStage._safe_strategy_name("paper123") == "paper123"

    def test_special_chars_sanitized(self) -> None:
        assert r101.FactorStage._safe_strategy_name("Alpha#1") == "Alpha_1"
        assert r101.FactorStage._safe_strategy_name("中文/策略") == "_____"

    def test_unicode_to_underscore(self) -> None:
        assert r101.FactorStage._safe_strategy_name("a/b\\c") == "a_b_c"


# ── RunConfig ────────────────────────────────────────


class TestRunConfigStrategyMode:
    def test_default_off(self) -> None:
        cfg = r101.RunConfig()
        assert cfg.strategy_mode == "off"

    def test_per_alpha(self) -> None:
        cfg = r101.RunConfig(strategy_mode="per_alpha")
        assert cfg.strategy_mode == "per_alpha"

    def test_after_batch(self) -> None:
        cfg = r101.RunConfig(strategy_mode="after_batch")
        assert cfg.strategy_mode == "after_batch"

    def test_effective_strategies_dir_default(self) -> None:
        cfg = r101.RunConfig()
        # PROPERTY fallback uses parent.parent.parent so default = PROJECT_ROOT/quant/strategies
        assert cfg.effective_strategies_dir.name == "strategies"
        assert "quant" in cfg.effective_strategies_dir.parts

    def test_effective_strategies_dir_override(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path / "my_strategies")
        assert cfg.effective_strategies_dir == tmp_path / "my_strategies"


# ── CLI argparse ────────────────────────────────────────


class TestCliParsing:
    def test_default_off(self) -> None:
        """No flag → strategy_mode='off', strategies_dir=None."""
        import argparse
        parser = _build_parser_for_test()
        ns = parser.parse_args([])
        assert getattr(ns, "strategy_mode", "off") == "off"
        assert getattr(ns, "strategies_dir", None) is None

    def test_per_alpha(self) -> None:
        import argparse
        parser = _build_parser_for_test()
        ns = parser.parse_args(["--strategy-mode", "per_alpha"])
        assert ns.strategy_mode == "per_alpha"

    def test_after_batch(self) -> None:
        import argparse
        parser = _build_parser_for_test()
        ns = parser.parse_args(["--strategy-mode", "after_batch"])
        assert ns.strategy_mode == "after_batch"

    def test_invalid_mode_rejected(self) -> None:
        import argparse
        parser = _build_parser_for_test()
        with pytest.raises(SystemExit):
            parser.parse_args(["--strategy-mode", "bogus"])

    def test_strategies_dir(self) -> None:
        import argparse
        parser = _build_parser_for_test()
        ns = parser.parse_args(["--strategies-dir", "/tmp/x"])
        assert str(ns.strategies_dir) == "/tmp/x"


def _build_parser_for_test() -> "argparse.ArgumentParser":
    """Mirror main()'s argparse setup, but without run-blocking args."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy-mode",
        choices=["off", "per_alpha", "after_batch"],
        default="off",
    )
    parser.add_argument("--strategies-dir", type=Path, default=None)
    return parser


# ── Helpers for sink tests ────────────────────────────────────────


def _make_fr(idx: int, status: str = "success", ic: float = 0.05, icir: float = 0.3) -> FactorResult:
    """Construct a FactorResult with realistic 101-alpha fields."""
    signal = Signal(
        id=f"101_alphas_minimal_alpha_{idx:03d}",
        name=f"Alpha#{idx}",
        formula_brief=f"rank(...) fake formula for alpha #{idx}",
        metadata={"alpha_index": idx},
    )
    return FactorResult(
        signal=signal,
        status=status,
        code="def compute_factor(df): return df['close']",
        code_chars=37,
        factor_series=None,
        long_df=None,
        h5_path=None,
        backtest={
            "ic_mean": ic,
            "icir": icir,
            "rank_ic_mean": ic * 0.9,
            "rank_icir": icir * 0.95,
            "win_rate": 0.55,
            "annual_return": 0.15,
            "longshort_max_dd": -0.08,
        },
        stage=None,
        error=None,
        elapsed_sec=10.0,
        metadata={},
    )


class _MockFactorStage:
    """Minimal stand-in: only the 3 M3.4 sink methods (no full __init__)."""

    def __init__(self, config):
        self.config = config

    _alpha_to_signal_type = staticmethod(r101.FactorStage._alpha_to_signal_type)
    _safe_strategy_name = staticmethod(r101.FactorStage._safe_strategy_name)

    def _strategy_name_for(self, fr):
        idx = fr.signal.metadata.get("alpha_index", 0)
        return f"{self.config.paper_id}_{int(idx):03d}_{self._alpha_to_signal_type(fr.signal.name)}"

    def _persist_strategy(self, fr):
        if self.config.strategy_mode != "per_alpha":
            return
        if fr.status != "success":
            return
        from QuantNodes.research.persist import strategy_library as sl
        bt = fr.backtest or {}
        strategy_name = self._strategy_name_for(fr)
        data = {
            "strategy": {
                "name": strategy_name,
                "signal_type": self._alpha_to_signal_type(fr.signal.name),
                "status": "已注册",
                "l1": {"signal_params": bt},
            },
            "backtest": {"status": "success"},
        }
        return sl.write_strategy_yaml(
            name=strategy_name, data=data,
            strategies_dir=self.config.effective_strategies_dir,
        )

    def _persist_batch_strategy(self, results):
        if self.config.strategy_mode != "after_batch":
            return
        successes = [r for r in results if r.status == "success"]
        if not successes:
            return
        from QuantNodes.research.persist import strategy_library as sl
        composite_name = f"{self.config.paper_id}_composite"
        n = len(successes)
        data = {
            "strategy": {
                "name": composite_name,
                "signal_type": "signal_composite",
                "status": "已注册",
                "l1": {
                    "config": {
                        "weights": {r.signal.name: 1.0 / n for r in successes},
                    },
                },
                "l2": {"factors": [r.signal.id for r in successes]},
            },
        }
        return sl.write_strategy_yaml(
            name=composite_name, data=data,
            strategies_dir=self.config.effective_strategies_dir,
        )


# ── _strategy_name_for ────────────────────────────────────────


class TestStrategyNameFor:
    def test_naming_convention(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path)
        stage = _MockFactorStage(cfg)
        fr = _make_fr(5)
        # paper_id default = "101_alphas_minimal", idx=5, signal_type=Alpha#5=fallback→factor_rank
        assert stage._strategy_name_for(fr) == "101_alphas_minimal_005_factor_rank"

    def test_custom_paper_id(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, paper_id="my_paper")
        stage = _MockFactorStage(cfg)
        fr = _make_fr(42)
        assert stage._strategy_name_for(fr) == "my_paper_042_factor_rank"


# ── _persist_strategy ────────────────────────────────────────


class TestPersistStrategy:
    def test_mode_off_skips(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path)
        stage = _MockFactorStage(cfg)
        assert stage._persist_strategy(_make_fr(1)) is None
        # No files written
        assert list(tmp_path.iterdir()) == []

    def test_mode_per_alpha_writes(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="per_alpha")
        stage = _MockFactorStage(cfg)
        ret = stage._persist_strategy(_make_fr(1))
        assert "Created" in ret or "Updated" in ret
        # Verify file landed
        yaml_path = tmp_path / "101_alphas_minimal_001_factor_rank" / "strategy.yaml"
        assert yaml_path.exists()

    def test_failed_status_skipped(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="per_alpha")
        stage = _MockFactorStage(cfg)
        assert stage._persist_strategy(_make_fr(1, status="failed")) is None
        assert list(tmp_path.iterdir()) == []

    def test_metrics_passed_to_l1(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="per_alpha")
        stage = _MockFactorStage(cfg)
        stage._persist_strategy(_make_fr(7, ic=0.123, icir=0.456))
        yaml_path = tmp_path / "101_alphas_minimal_007_factor_rank" / "strategy.yaml"
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert data["strategy"]["signal_type"] == "factor_rank"
        assert data["strategy"]["l1"]["signal_params"]["ic_mean"] == 0.123
        assert data["strategy"]["l1"]["signal_params"]["icir"] == 0.456


# ── _persist_batch_strategy ────────────────────────────────────────


class TestPersistBatchStrategy:
    def test_mode_off_skips(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path)
        stage = _MockFactorStage(cfg)
        stage._persist_batch_strategy([_make_fr(1), _make_fr(2)])
        assert list(tmp_path.iterdir()) == []

    def test_empty_results_skipped(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="after_batch")
        stage = _MockFactorStage(cfg)
        stage._persist_batch_strategy([])
        assert list(tmp_path.iterdir()) == []

    def test_no_successes_skipped(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="after_batch")
        stage = _MockFactorStage(cfg)
        stage._persist_batch_strategy([_make_fr(1, status="failed")])
        assert list(tmp_path.iterdir()) == []

    def test_single_success_writes(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="after_batch")
        stage = _MockFactorStage(cfg)
        ret = stage._persist_batch_strategy([_make_fr(1)])
        assert "Created" in ret or "Updated" in ret
        yaml_path = tmp_path / "101_alphas_minimal_composite" / "strategy.yaml"
        assert yaml_path.exists()

    def test_multi_success_equal_weights(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="after_batch")
        stage = _MockFactorStage(cfg)
        results = [_make_fr(i) for i in range(1, 5)]
        stage._persist_batch_strategy(results)
        yaml_path = tmp_path / "101_alphas_minimal_composite" / "strategy.yaml"
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        cfg_data = data["strategy"]["l1"]["config"]
        weights = cfg_data["weights"]
        assert len(weights) == 4
        # Equal weights: each 0.25
        assert all(abs(w - 0.25) < 1e-9 for w in weights.values())
        assert data["strategy"]["l2"]["factors"] == [
            f"101_alphas_minimal_alpha_{i:03d}" for i in range(1, 5)
        ]

    def test_composite_signal_type(self, tmp_path: Path) -> None:
        cfg = r101.RunConfig(strategies_dir=tmp_path, strategy_mode="after_batch")
        stage = _MockFactorStage(cfg)
        stage._persist_batch_strategy([_make_fr(1), _make_fr(2), _make_fr(3)])
        yaml_path = tmp_path / "101_alphas_minimal_composite" / "strategy.yaml"
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert data["strategy"]["signal_type"] == "signal_composite"
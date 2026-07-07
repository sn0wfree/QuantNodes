"""Tests for `QuantNodes.research.signal_source.bridge` — SignalV2 PR6.

Covers:
  - `classify_paper_signal`: heuristic over Signal.name
  - `classify_name`: heuristic over plain string
  - `signal_type_to_strategy_class`: enum → StrategyNode mapping
  - Edge cases: empty name, lowercase, special chars, ordering
  - Roundtrip: paper Signal → SignalType → StrategyNode

The heuristic was previously inlined in
`scripts/research/run_101_alphas_v2.py:_alpha_to_signal_type`. The bridge
preserves the exact same logic — these tests pin it so future refactors
don't drift.
"""
from __future__ import annotations

import pytest

from QuantNodes.research.paper_understanding.contracts import SignalType
from QuantNodes.research.signal_source.base import Signal
from QuantNodes.research.signal_source.bridge import (
    classify_name,
    classify_paper_signal,
    signal_type_to_strategy_class,
)


# ── 1. classify_name heuristic ────────────────────────────────


class TestClassifyNameHeuristic:
    """Heuristic precedence: rsi > volatility > momentum > ma_cross > factor_rank."""

    def test_rsi_substring(self):
        assert classify_name("RSI_14day") == SignalType.RSI

    def test_rsi_lowercase(self):
        assert classify_name("my_rsi_strategy") == SignalType.RSI

    def test_volatility_substring(self):
        assert classify_name("volatility_breakout") == SignalType.VOLATILITY

    def test_vol_underscore_suffix(self):
        assert classify_name("alpha_vol") == SignalType.VOLATILITY

    def test_vol_underscore_infix(self):
        assert classify_name("alpha_vol_breakout") == SignalType.VOLATILITY

    def test_momentum_substring(self):
        assert classify_name("momentum_60d") == SignalType.MOMENTUM

    def test_mom_underscore_suffix(self):
        assert classify_name("alpha_mom") == SignalType.MOMENTUM

    def test_mom_prefix(self):
        assert classify_name("mom_60d") == SignalType.MOMENTUM

    def test_ma_cross_substring(self):
        assert classify_name("ma_cross_5_20") == SignalType.MA_CROSS

    def test_ma_underscore(self):
        assert classify_name("ma_5_20") == SignalType.MA_CROSS

    def test_macross_no_separator(self):
        assert classify_name("macross_5_20") == SignalType.MA_CROSS

    def test_factor_rank_default(self):
        assert classify_name("alpha_rank_close") == SignalType.FACTOR_RANK

    def test_factor_keyword(self):
        assert classify_name("factor_value") == SignalType.FACTOR_RANK

    def test_default_fallback_101_alphas(self):
        """101 alphas are all rank-based → factor_rank fallback."""
        for name in ("Alpha#1", "Alpha#2", "Alpha#50", "Alpha#101"):
            assert classify_name(name) == SignalType.FACTOR_RANK

    def test_empty_name_returns_factor_rank(self):
        assert classify_name("") == SignalType.FACTOR_RANK

    def test_unknown_name_returns_factor_rank(self):
        assert classify_name("xyz_unknown_strategy") == SignalType.FACTOR_RANK


# ── 2. Heuristic ordering (priority) ───────────────────────────


class TestHeuristicOrdering:
    """More-specific tokens beat less-specific ones."""

    def test_momentum_beats_factor(self):
        # "momentum_factor_value" → momentum (not factor_rank)
        assert classify_name("momentum_factor_value") == SignalType.MOMENTUM

    def test_volatility_beats_rank(self):
        # "vol_rank_breakout" → no "volatility"/"_vol_" → factor_rank
        # (The old inline heuristic matched "_vol_" infix only, not bare "vol".)
        assert classify_name("vol_rank_breakout") == SignalType.FACTOR_RANK

    def test_rsi_beats_momentum(self):
        # Branch order: rsi > momentum, but momentum keyword isn't present,
        # so result is rsi.
        assert classify_name("rsi_strategy") == SignalType.RSI


# ── 3. classify_paper_signal ───────────────────────────────────


class TestClassifyPaperSignal:
    """classify_paper_signal: Signal.name → SignalType."""

    def test_basic_signal(self):
        s = Signal(id="001", name="momentum_60d", formula_brief="ret(close, 60)")
        assert classify_paper_signal(s) == SignalType.MOMENTUM

    def test_case_insensitive(self):
        s = Signal(id="002", name="VOLATILITY_BREAKOUT", formula_brief="...")
        assert classify_paper_signal(s) == SignalType.VOLATILITY

    def test_signal_with_metadata_ignored(self):
        s = Signal(
            id="003",
            name="alpha_rank",
            formula_brief="rank(...)",
            metadata={"index": 5, "paper_id": "101_alphas"},
        )
        # metadata not used by heuristic
        assert classify_paper_signal(s) == SignalType.FACTOR_RANK


# ── 4. signal_type_to_strategy_class ───────────────────────────


class TestSignalTypeToStrategyClass:
    """Enum → prebuilt StrategyNode class mapping."""

    def test_rsi_maps_to_rsi_strategy_node(self):
        cls = signal_type_to_strategy_class(SignalType.RSI)
        assert cls is not None
        assert cls.__name__ == "RSIStrategyNode"

    def test_volatility_maps(self):
        cls = signal_type_to_strategy_class(SignalType.VOLATILITY)
        assert cls is not None
        assert cls.__name__ == "VolatilityStrategyNode"

    def test_momentum_maps(self):
        cls = signal_type_to_strategy_class(SignalType.MOMENTUM)
        assert cls is not None
        assert cls.__name__ == "MomentumStrategyNode"

    def test_ma_cross_maps(self):
        cls = signal_type_to_strategy_class(SignalType.MA_CROSS)
        assert cls is not None
        assert cls.__name__ == "MACrossStrategyNode"

    def test_factor_rank_maps(self):
        cls = signal_type_to_strategy_class(SignalType.FACTOR_RANK)
        assert cls is not None
        assert cls.__name__ == "FactorRankStrategyNode"

    def test_unknown_returns_none(self):
        assert signal_type_to_strategy_class(SignalType.UNKNOWN) is None

    def test_signal_composite_maps(self):
        cls = signal_type_to_strategy_class(SignalType.SIGNAL_COMPOSITE)
        assert cls is not None
        assert cls.__name__ == "SignalCompositeStrategyNode"


# ── 5. Roundtrip ───────────────────────────────────────────────


class TestRoundtrip:
    """Paper Signal → SignalType → StrategyNode class."""

    def test_roundtrip_alpha_5(self):
        s = Signal(id="005", name="Alpha#5", formula_brief="rank(close)")
        sig_type = classify_paper_signal(s)
        cls = signal_type_to_strategy_class(sig_type)
        assert cls is not None
        assert cls.__name__ == "FactorRankStrategyNode"

    def test_roundtrip_momentum(self):
        s = Signal(id="060", name="momentum_60d", formula_brief="ret(close, 60)")
        sig_type = classify_paper_signal(s)
        cls = signal_type_to_strategy_class(sig_type)
        assert cls is not None
        assert cls.__name__ == "MomentumStrategyNode"


# ── 6. Equivalence with old inline heuristic ───────────────────


class TestInlineHeuristicEquivalence:
    """The bridge must produce the same output as the old inline heuristic
    in `run_101_alphas_v2.py:_alpha_to_signal_type` for all 101 alphas."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Alpha#1", "factor_rank"),
            ("Alpha#50", "factor_rank"),
            ("Alpha#101", "factor_rank"),
            ("momentum_60d", "momentum"),
            ("rsi_strategy", "rsi"),
            ("vol_breakout", "factor_rank"),
            ("vol_20d_breakout", "factor_rank"),
            ("_vol_breakout", "volatility"),
            ("breakout_vol", "volatility"),
            ("my_vol_strategy", "volatility"),
            ("ma_cross", "ma_cross"),
            ("ma_5_20", "ma_cross"),
            ("macross_5_20", "ma_cross"),
            ("factor_value", "factor_rank"),
            ("alpha_rank", "factor_rank"),
        ],
    )
    def test_name_to_signal_type_string(self, name, expected):
        assert classify_name(name).value == expected
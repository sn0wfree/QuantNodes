"""Cross-layer Signal bridge — SignalV2 PR6.

Connects the three canonical Signal-like dataclasses:

  - `Signal`       (paper layer)   — `QuantNodes.research.signal_source.base`
  - `SignalType`   (classifier)    — `QuantNodes.research.paper_understanding.contracts`
  - `TradeSignal`  (trade layer)   — `QuantNodes.backtest.strategy_node`

This module encapsulates the heuristic that maps a paper-extracted Signal
to a SignalType enum value (and optionally to a StrategyNode class).

Why a separate bridge module?
  - The heuristic (`rsi > volatility > momentum > ma_cross > factor_rank`)
    was previously inlined in `run_101_alphas_v2.py:_alpha_to_signal_type`.
    Promoting it to a bridge module:
      (a) makes the layer crossing explicit
      (b) lets tests pin the heuristic behavior in one place
      (c) allows other consumers (CLI, UI, agent tools) to reuse the
          same classification without re-implementing it

Branch order matters: more specific tokens first, falling through to
`SignalType.FACTOR_RANK` (the default for 101 alphas, which are all
rank-based cross-sectional factors).

Usage:
    from QuantNodes.research.signal_source import Signal, TrackBSignalSource
    from QuantNodes.research.signal_source.bridge import (
        classify_paper_signal, classify_name, signal_type_to_strategy_class,
    )

    src = TrackBSignalSource(Path("quant/papers/101_alphas_minimal/track_b.json"))
    for signal in src.iter_signals():
        sig_type = classify_paper_signal(signal)
        strategy_cls = signal_type_to_strategy_class(sig_type)
        if strategy_cls is not None:
            node = strategy_cls()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from QuantNodes.research.signal_source.base import Signal
from QuantNodes.research.paper_understanding.contracts import SignalType

if TYPE_CHECKING:
    from QuantNodes.backtest.strategy_node import StrategyNode


# ── Heuristic table ────────────────────────────────────────────
#
# Branch order matters: more specific tokens first. Each entry maps
# (substring match) → SignalType. Order preserved for transparency.
_HEURISTIC: list[tuple[str, SignalType]] = [
    ("rsi", SignalType.RSI),
    ("volatility", SignalType.VOLATILITY),
    ("_vol_", SignalType.VOLATILITY),
    ("_vol", SignalType.VOLATILITY),
    ("momentum", SignalType.MOMENTUM),
    ("_mom_", SignalType.MOMENTUM),
    ("_mom", SignalType.MOMENTUM),
    ("mom_", SignalType.MOMENTUM),
    ("_ma_", SignalType.MA_CROSS),
    ("ma_", SignalType.MA_CROSS),
    ("_ma", SignalType.MA_CROSS),
    ("macross", SignalType.MA_CROSS),
    ("factor", SignalType.FACTOR_RANK),
    ("rank", SignalType.FACTOR_RANK),
]


def classify_paper_signal(signal: Signal) -> SignalType:
    """Heuristic: paper `Signal` → `SignalType` enum.

    Args:
        signal: Paper-extracted Signal (id, name, formula_brief, metadata).

    Returns:
        SignalType enum value. Defaults to `SignalType.FACTOR_RANK` when
        no heuristic token matches — correct for 101 alphas (rank-based).
    """
    return classify_name(signal.name)


def classify_name(name: str) -> SignalType:
    """Heuristic: plain string name → `SignalType` enum.

    Used both by `classify_paper_signal()` and directly by callers that
    have only a name (e.g. test fixtures, run_101 inline heuristic).

    Args:
        name: Factor or signal name (case-insensitive substring match).

    Returns:
        SignalType enum value. Defaults to `SignalType.FACTOR_RANK`.
    """
    if not name:
        return SignalType.FACTOR_RANK
    n = name.lower()
    for token, sig_type in _HEURISTIC:
        if token in n:
            return sig_type
    return SignalType.FACTOR_RANK


def signal_type_to_strategy_class(
    signal_type: SignalType,
) -> "type[StrategyNode] | None":
    """Map `SignalType` enum → prebuilt `StrategyNode` class.

    Lazy import `SIGNAL_NODE_REGISTRY` to avoid circular dependency with
    `QuantNodes.research.backtest.strategies` (which imports from
    `backtest.strategy_node`).

    Args:
        signal_type: SignalType enum value.

    Returns:
        StrategyNode subclass, or `None` if no mapping (e.g.
        `SignalType.UNKNOWN`, `SignalType.SIGNAL_COMPOSITE`).
    """
    from QuantNodes.research.backtest.strategies import SIGNAL_NODE_REGISTRY

    return SIGNAL_NODE_REGISTRY.get(signal_type.value)


__all__ = [
    "classify_paper_signal",
    "classify_name",
    "signal_type_to_strategy_class",
]
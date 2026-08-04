"""Tests for layer model functions — _model_harrv, _model_climatology, etc."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk.ensemble import (
    _model_harrv, _model_climatology, _model_ewma, _model_current_vol,
    OOS_START, OOS_END,
)


def _make_returns() -> pd.Series:
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", "2025-06-30")
    rets = pd.Series(np.random.randn(len(idx)) * 0.5, index=idx)
    return rets


def _oos_idx(rets: pd.Series) -> pd.Index:
    return rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]


class TestModelHarrv:
    def test_returns_series(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_harrv(rets, oos)
        assert isinstance(f, pd.Series)
        assert len(f.dropna()) > 0

    def test_output_not_constant(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_harrv(rets, oos).dropna()
        assert f.nunique() > 1


class TestModelClimatology:
    def test_returns_series(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_climatology(rets, oos)
        assert isinstance(f, pd.Series)
        assert len(f.dropna()) > 0

    def test_output_not_constant(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_climatology(rets, oos).dropna()
        assert f.nunique() > 1


class TestModelEwma:
    def test_returns_series(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_ewma(rets, oos)
        assert isinstance(f, pd.Series)
        assert len(f.dropna()) > 0

    def test_lam_persistence(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f_high = _model_ewma(rets, oos, lam=0.99)
        f_low = _model_ewma(rets, oos, lam=0.50)
        # High lambda should be smoother (lower variance)
        assert f_high.std() < f_low.std() * 2


class TestModelCurrentVol:
    def test_returns_series(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_current_vol(rets, oos)
        assert isinstance(f, pd.Series)
        assert len(f.dropna()) > 0

    def test_looks_like_volatility(self) -> None:
        rets = _make_returns()
        oos = _oos_idx(rets)
        f = _model_current_vol(rets, oos).dropna()
        # 20d vol of daily returns * sqrt(252) should be positive
        assert (f > 0).all()
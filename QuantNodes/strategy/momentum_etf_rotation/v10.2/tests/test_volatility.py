"""Tests for volatility estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ca_gcp.core.volatility import estimate_volatility


def test_estimate_volatility_shape():
    df = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (100, 5)))
    sigma = estimate_volatility(df)
    assert sigma.shape == df.shape
    assert (sigma > 0).all().all()
    assert not sigma.isna().any().any()


def test_estimate_volatility_stable_low_vol():
    df = pd.DataFrame(np.zeros((100, 3)))
    sigma = estimate_volatility(df)
    assert (sigma > 0).all().all()
    assert sigma.values.max() < 1.0
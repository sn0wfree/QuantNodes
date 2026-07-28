"""Tests for modulator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ca_gcp.core.modulator import apply_modulator, compute_systemic_stress


def test_stress_in_unit_interval():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (200, 5)))
    sigma = pd.DataFrame(rng.uniform(0.005, 0.02, (200, 5)))
    s = compute_systemic_stress(df, sigma)
    assert (s >= 0).all() and (s <= 1).all()


def test_apply_modulator_scales_width():
    hw = pd.DataFrame(np.ones((10, 3)) * 0.02)
    stress = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = apply_modulator(hw, stress, eta=0.5)
    assert (out.iloc[0].values == hw.iloc[0].values).all()
    assert out.iloc[1, 0] > hw.iloc[1, 0]
"""Tests for end-to-end CA-GCP pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ca_gcp.core.pipeline import CAGCPConfig, CAGCPipeline


def test_pipeline_smoke():
    rng = np.random.default_rng(0)
    n, t = 8, 400
    df = pd.DataFrame(rng.normal(0, 0.01, (t, n)))
    df.index = pd.date_range("2020-01-01", periods=t, freq="B")
    train = df.iloc[:200]
    calib = df.iloc[150:300]
    test = df.iloc[300:]
    pipe = CAGCPipeline(CAGCPConfig(k=3, alpha=0.05))
    pipe.fit(train)
    out = pipe.predict(calib, test)
    assert out["lower"].shape == test.shape
    assert out["upper"].shape == test.shape
    assert (out["upper"] >= out["lower"]).all().all()


def test_pipeline_no_inf_intervals():
    rng = np.random.default_rng(1)
    n, t = 5, 300
    df = pd.DataFrame(rng.normal(0, 0.005, (t, n)))
    df.iloc[::50] = rng.normal(0, 0.05, (6, n))
    df.index = pd.date_range("2020-01-01", periods=t, freq="B")
    pipe = CAGCPipeline(CAGCPConfig(k=3))
    pipe.fit(df.iloc[:200])
    out = pipe.predict(df.iloc[200:250], df.iloc[250:])
    assert not np.isinf(out["half_width"].values).any()
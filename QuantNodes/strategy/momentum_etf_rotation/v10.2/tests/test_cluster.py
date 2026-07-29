"""Tests for sector-clustered CA-GCP."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (
    SectorCAGCPResult,
    build_sector_groups,
    fit_sector_ca_gcp,
    fit_sector_hybrid_ca_gcp,
    load_sector_map,
    predict_sector_ca_gcp,
    CAGCPConfig,
)


@pytest.fixture
def sector_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = 250
    dates = pd.date_range("2020-01-01", periods=t, freq="B")

    tech = rng.normal(0, 0.02, (t, 4))
    tech_corr = rng.uniform(0.5, 0.9, (4, 4))
    tech_corr = (tech_corr + tech_corr.T) / 2
    np.fill_diagonal(tech_corr, 1.0)
    tech_L = np.linalg.cholesky(tech_corr)
    tech = tech @ tech_L.T

    fin = rng.normal(0, 0.015, (t, 3))
    fin_corr = rng.uniform(0.4, 0.8, (3, 3))
    fin_corr = (fin_corr + fin_corr.T) / 2
    np.fill_diagonal(fin_corr, 1.0)
    fin_L = np.linalg.cholesky(fin_corr)
    fin = fin @ fin_L.T

    df = pd.DataFrame(
        np.concatenate([tech, fin], axis=1),
        index=dates,
        columns=["T1", "T2", "T3", "T4", "F1", "F2", "F3"],
    )
    return df


@pytest.fixture
def sectors_dict(sector_returns: pd.DataFrame) -> dict[str, list[str]]:
    return {"tech": ["T1", "T2", "T3", "T4"], "fin": ["F1", "F2", "F3"]}


def test_load_sector_map_writes_and_reads(tmp_path: Path):
    csv = tmp_path / "map.csv"
    csv.write_text("code,sector\nT1,tech\nF1,fin\n")
    mapping = load_sector_map(csv)
    assert mapping == {"T1": "tech", "F1": "fin"}


def test_build_sector_groups_filters_small(sector_returns):
    mapping = {"T1": "tech", "T2": "tech", "T3": "tech", "T4": "tech", "F1": "fin", "F2": "fin", "F3": "fin"}
    groups = build_sector_groups(list(sector_returns.columns), mapping, min_size=3)
    assert set(groups.keys()) == {"tech", "fin"}
    assert len(groups["tech"]) == 4
    assert len(groups["fin"]) == 3


def test_fit_sector_ca_gcp_returns_pipelines_per_sector(sector_returns, sectors_dict):
    pipes = fit_sector_ca_gcp(sector_returns, sectors_dict, CAGCPConfig(k=3))
    assert set(pipes.keys()) == {"tech", "fin"}
    assert pipes["tech"].codes == ["T1", "T2", "T3", "T4"]
    assert pipes["fin"].codes == ["F1", "F2", "F3"]


def test_predict_sector_ca_gcp_concatenates_outputs(sector_returns, sectors_dict):
    pipes = fit_sector_ca_gcp(sector_returns, sectors_dict, CAGCPConfig(k=2))
    train = sector_returns.iloc[:150]
    calib = sector_returns.iloc[100:200]
    test = sector_returns.iloc[200:]
    out = predict_sector_ca_gcp(pipes, calib, test)
    assert isinstance(out, SectorCAGCPResult)
    assert out.lower.shape == test.shape
    assert out.upper.shape == test.shape
    assert out.half_width.shape == test.shape
    assert set(out.sector_stress.keys()) == {"tech", "fin"}
    assert not (out.upper < out.lower).any().any()


def test_k_auto_adjusts_for_tiny_sector(sector_returns):
    sectors = {"tech": ["T1", "T2", "T3", "T4"], "fin": ["F1", "F2", "F3"]}
    pipes = fit_sector_ca_gcp(sector_returns, sectors, CAGCPConfig(k=10))
    assert pipes["fin"].config.k == 2
    assert pipes["tech"].config.k == 3


def test_fit_sector_hybrid_ca_gcp_keeps_sectors_independent(sector_returns, sectors_dict):
    pipes = fit_sector_hybrid_ca_gcp(sector_returns, sectors_dict, CAGCPConfig(k=3), cross_sector_threshold=0.99)
    assert set(pipes.keys()) == {"tech", "fin"}
    assert "T1" in pipes["tech"].codes
    assert "F1" in pipes["fin"].codes
"""Shared fixtures for CA-GCP early-warning tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CA_GCP_EW = _ROOT
if str(_CA_GCP_EW) not in sys.path:
    sys.path.insert(0, str(_CA_GCP_EW))

from ca_gcp_ew_eval import KNOWN_EVENTS, get_asset_sectors  # noqa: E402


def _select_real_etf_codes(n_per_sector: int = 5,
                            target_sectors: tuple[str, ...] = ("a_broad", "overseas", "bond",
                                                                "commodity")) -> list[str]:
    sectors = get_asset_sectors()
    by_sector: dict[str, list[str]] = {}
    for code, sec in sectors.items():
        by_sector.setdefault(sec, []).append(code)
    selected: list[str] = []
    for sec in target_sectors:
        selected.extend(by_sector.get(sec, [])[:n_per_sector])
    if len(selected) < n_per_sector * len(target_sectors):
        for s_list in by_sector.values():
            for c in s_list:
                if c not in selected:
                    selected.append(c)
                if len(selected) >= n_per_sector * len(target_sectors):
                    break
            if len(selected) >= n_per_sector * len(target_sectors):
                break
    return selected


_REAL_CODES = _select_real_etf_codes()


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-02", periods=400, freq="B")
    cols = _REAL_CODES
    rets = rng.normal(0.0005, 0.012, (len(dates), len(cols)))
    return pd.DataFrame(rets, index=dates, columns=cols)


@pytest.fixture
def bullish_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-02", periods=120, freq="B")
    cols = _REAL_CODES
    rets = rng.normal(0.01, 0.005, (len(dates), len(cols)))
    return pd.DataFrame(rets, index=dates, columns=cols)


@pytest.fixture
def bearish_returns() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    dates = pd.date_range("2020-01-02", periods=120, freq="B")
    cols = _REAL_CODES
    rets = rng.normal(-0.01, 0.025, (len(dates), len(cols)))
    return pd.DataFrame(rets, index=dates, columns=cols)


@pytest.fixture
def trend_ok_series(sample_returns: pd.DataFrame) -> pd.Series:
    n = len(sample_returns)
    vals = [True] * (n // 2) + [False] * (n - n // 2)
    return pd.Series(vals, index=sample_returns.index, name="trend_ok")


@pytest.fixture
def sample_hw(sample_returns: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.uniform(0.005, 0.05, sample_returns.shape),
        index=sample_returns.index,
        columns=sample_returns.columns,
    )


@pytest.fixture
def sample_stress(sample_returns: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(2)
    return pd.Series(
        rng.uniform(0.0, 0.4, len(sample_returns)),
        index=sample_returns.index,
        name="stress",
    )


@pytest.fixture
def sample_lower_upper(sample_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    lower = pd.DataFrame(
        rng.uniform(-0.04, -0.01, sample_returns.shape),
        index=sample_returns.index,
        columns=sample_returns.columns,
    )
    upper = pd.DataFrame(
        rng.uniform(0.01, 0.04, sample_returns.shape),
        index=sample_returns.index,
        columns=sample_returns.columns,
    )
    return lower, upper


def _make_alerts_from_indices(
    index: pd.DatetimeIndex,
    wz_vals: list[float],
    sv_vals: list[float],
    fired_vals: list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "width_z": wz_vals,
            "stress": sv_vals,
            "fired": fired_vals,
        },
        index=index,
    )


@pytest.fixture
def sample_alerts_with_fired(sample_returns: pd.DataFrame) -> pd.DataFrame:
    n = len(sample_returns)
    rng = np.random.default_rng(5)
    wz = rng.uniform(0.0, 5.0, n).tolist()
    sv = rng.uniform(0.0, 1.0, n).tolist()
    fired = (np.array(wz) > 3.0).astype(int).tolist()
    return _make_alerts_from_indices(sample_returns.index, wz, sv, fired)


@pytest.fixture
def sample_sector_map() -> dict[str, str]:
    return {
        "510300": "宽基", "510500": "宽基", "518880": "黄金",
        "513100": "海外", "511260": "债券", "512760": "半导体",
        "512480": "半导体", "515030": "新能源", "515790": "光伏",
    }


@pytest.fixture
def known_events() -> list[dict]:
    return KNOWN_EVENTS

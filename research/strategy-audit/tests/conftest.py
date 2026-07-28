"""Shared pytest fixtures."""
from pathlib import Path

import pytest


@pytest.fixture
def pkg_root() -> Path:
    """Path to quantnodes-strategy-audit project root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def lessons_dir(pkg_root: Path) -> Path:
    return pkg_root / "lessons"


@pytest.fixture
def rules_path(pkg_root: Path) -> Path:
    return pkg_root / "rules" / "simple_rules.yaml"


@pytest.fixture
def bad_code() -> str:
    """Sample code with multiple look-ahead and NaN-safe violations."""
    return '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def bad_pct_change(nav):
    """Bare pct_change without NaN-safe wrapper."""
    return nav.pct_change()


def bad_standardize(X):
    """Full-sample standardization (look-ahead bias)."""
    mean = X.mean()
    std = X.std()
    return (X - mean) / std


def bad_standardscaler(X):
    """sklearn StandardScaler default = full sample fit."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def bad_shift_future(prices):
    """Using .shift(-1) to reference future."""
    return prices.shift(-1)


def bad_fillna_zero(returns):
    """fillna(0) on returns produces pseudo-zero returns."""
    return returns.pct_change().fillna(0)
'''


@pytest.fixture
def good_code() -> str:
    """Sample code with NaN-safe and rolling patterns."""
    return '''
import pandas as pd
import numpy as np


def good_pct_change(nav):
    """NaN-safe pct_change with .where."""
    return nav.pct_change().where(
        nav.shift(1).notna() & nav.notna()
    )


def good_rolling_standardize(X):
    """Rolling standardization (no look-ahead)."""
    mean = X.rolling(252).mean()
    std = X.rolling(252).std()
    return (X - mean) / std


def good_expanding_standardize(X):
    """Expanding standardization."""
    mean = X.expanding(min_periods=252).mean()
    std = X.expanding(min_periods=252).std()
    return (X - mean) / std


def good_shift_past(prices):
    """shift(1) is OK (uses past data)."""
    return prices.shift(1)
'''

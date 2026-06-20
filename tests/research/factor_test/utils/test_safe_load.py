"""Tests for utils/safe_load.py — Phase J3+J4."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.safe_load import (
    safe_load_custom,
    safe_load_factor,
    safe_load_h5,
    try_load_panels,
)


class TestSafeLoadH5:
    def test_returns_dataframe_on_success(self) -> None:
        loader = MagicMock()
        loader.load_h5.return_value = pd.DataFrame({"a": [1, 2]})
        loader.add_index.return_value = pd.DataFrame({"a": [1, 2]})
        result = safe_load_h5(loader, "stk_daily.h5", "cp")
        assert result is not None
        loader.load_h5.assert_called_once_with("stk_daily.h5", "cp")

    def test_returns_none_on_keyerror(self) -> None:
        loader = MagicMock()
        loader.load_h5.side_effect = KeyError("missing")
        result = safe_load_h5(loader, "stk_daily.h5", "missing")
        assert result is None

    def test_returns_none_on_filenotfound(self) -> None:
        loader = MagicMock()
        loader.load_h5.side_effect = FileNotFoundError("no file")
        result = safe_load_h5(loader, "missing.h5", "cp")
        assert result is None

    def test_logs_at_debug_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        loader = MagicMock()
        loader.load_h5.side_effect = KeyError("missing")
        with caplog.at_level(logging.DEBUG, logger="QuantNodes.research.factor_test.utils.safe_load"):
            safe_load_h5(loader, "stk_daily.h5", "missing")
        assert "safe_load_h5" in caplog.text


class TestTryLoadPanels:
    def test_returns_first_hit(self) -> None:
        loader = MagicMock()
        loader.load_h5.side_effect = KeyError("not here")  # first fails
        loader.add_index.return_value = pd.DataFrame({"x": [1]})
        # Override for second call (using a different mock side_effect would be cleaner
        # but MagicMock can't do this declaratively; use direct method swap)
        def fake_load(filename, key):
            if filename == "stk_daily.h5":
                raise KeyError("not here")
            return pd.DataFrame({"x": [1]})

        loader.load_h5.side_effect = fake_load
        result = try_load_panels(loader, "cp")
        assert result is not None

    def test_returns_none_when_no_candidates_match(self) -> None:
        loader = MagicMock()
        loader.load_h5.side_effect = KeyError("missing")
        result = try_load_panels(loader, "nope")
        assert result is None


class TestSafeLoadCustom:
    def test_returns_none_on_failure(self) -> None:
        loader = MagicMock()
        loader.load_custom.side_effect = ValueError("bad path")
        result = safe_load_custom(loader, ("/", "bad.csv"))
        assert result is None


class TestSafeLoadFactor:
    def test_returns_dataframe_on_success(self) -> None:
        loader = MagicMock()
        loader.load_factor.return_value = pd.DataFrame({"a": [1]})
        result = safe_load_factor(loader, "factor_dir", "mom_20")
        assert result is not None

    def test_returns_none_on_failure(self) -> None:
        loader = MagicMock()
        loader.load_factor.side_effect = FileNotFoundError("no factor")
        result = safe_load_factor(loader, "factor_dir", "mom_20")
        assert result is None

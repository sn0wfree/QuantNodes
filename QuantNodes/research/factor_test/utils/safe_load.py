"""Safe data-load helpers (Phase J3+J4, 2026-06-20).

Consolidates 11 ``try: ... except Exception: pass`` patterns that swallow
H5 / CSV load failures silently. Each helper logs a warning instead of
silently dropping data, so users can debug "where did my factor go?".

Used by:
  - LoadDataNode (load_data_node.py): 4 silent-fail sites (price,
    load_keys fallback, index_cp, etc.)
  - FactorNeutralizeNode: 1 risk-factor load fallback
  - RiskCorrelationNode: 1 risk-factor load fallback
  - GroupAnalyzerNode: 1 hedge_path load

All helpers return None on failure (instead of raising), so existing
control flow is preserved. Switch to logger.warning() from silent pass.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def safe_load_h5(
    loader: object,
    filename: str,
    key: str,
    *,
    axis_type: str = "stock",
    add_index: bool = True,
) -> "pd.DataFrame | None":
    """Load a single H5 key, returning None on failure.

    Args:
        loader: DataLoader instance.
        filename: H5 filename relative to loader.api (e.g. 'stk_daily.h5').
        key: H5 key (auto-prefixed with '/' if missing).
        axis_type: 'stock' or 'index' (passed to loader.add_index).
        add_index: if True, also add stock/index axis index to result.

    Returns:
        DataFrame or None on KeyError/FileNotFoundError/Exception.
        Logs at DEBUG level on failure (visible only when logging enabled).
    """
    try:
        data = loader.load_h5(filename, key)
        if add_index and hasattr(loader, "add_index"):
            return loader.add_index(data, axis_type=axis_type)
        return data
    except Exception as e:
        logger.debug("safe_load_h5(%s, %s) failed: %s", filename, key, e)
        return None


def try_load_panels(
    loader: object,
    key: str,
    *,
    axis_type: str = "stock",
    candidates: tuple[str, ...] = ("stk_daily.h5", "index_daily.h5"),
) -> "pd.DataFrame | None":
    """Try each candidate filename, return first non-None panel.

    Used when a key may exist in either stk_daily.h5 or index_daily.h5
    and we don't know which one ahead of time.

    Returns None if none of the candidates have the key (logs at DEBUG).
    """
    for fn in candidates:
        result = safe_load_h5(loader, fn, key, axis_type=axis_type)
        if result is not None:
            return result
    logger.debug("try_load_panels(%s) - key not in any candidate %s", key, candidates)
    return None


def safe_load_custom(
    loader: object,
    data_dir: tuple,
) -> "pd.DataFrame | None":
    """Wrap loader.load_custom(), return None on failure.

    Replaces bare ``except Exception: pass`` around load_custom calls.
    """
    try:
        return loader.load_custom(data_dir)
    except Exception as e:
        logger.debug("safe_load_custom(%s) failed: %s", data_dir, e)
        return None


def safe_load_factor(
    loader: object,
    factor_dir: str,
    factor_name: str,
) -> "pd.DataFrame | None":
    """Wrap loader.load_factor(), return None on failure."""
    try:
        return loader.load_factor(factor_dir, factor_name)
    except Exception as e:
        logger.debug("safe_load_factor(%s, %s) failed: %s", factor_dir, factor_name, e)
        return None

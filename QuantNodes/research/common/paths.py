"""Centralized Wiki path management.

All Wiki path operations MUST go through this module.
This is the single source of truth for Wiki directory structure.

Usage:
    from QuantNodes.research.paths import (
        WIKI_DIR_FACTOR,
        WIKI_DIR_STRATEGY,
        page_path,
        result_path,
    )

    # Get path for a factor definition page
    path = page_path(wiki, WIKI_DIR_FACTOR, "momentum")

    # Get path for a factor backtest result
    path = result_path(wiki, WIKI_DIR_FACTOR, "momentum", "20240101-20241231")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import config

# ─── Wiki directory constants ────────────────────────────────

def _get_wiki_dirs() -> dict[str, str]:
    """Get Wiki directory names from config."""
    return {
        "factor": config.get("wiki.factor_dir", "factor"),
        "strategy": config.get("wiki.strategy_dir", "strategy"),
        "sources": config.get("wiki.sources_dir", "sources"),
        "reproduction": config.get("wiki.reproduction_dir", "reproduction"),
    }


# Lazy initialization
_WIKI_DIRS: dict[str, str] | None = None


def _ensure_dirs() -> dict[str, str]:
    """Ensure wiki dirs are loaded."""
    global _WIKI_DIRS
    if _WIKI_DIRS is None:
        _WIKI_DIRS = _get_wiki_dirs()
    return _WIKI_DIRS


# Public constants (lazy)
WIKI_DIR_FACTOR: str = ""
WIKI_DIR_STRATEGY: str = ""
WIKI_DIR_SOURCES: str = ""
WIKI_DIR_REPRODUCTION: str = ""


def _init_constants() -> None:
    """Initialize constants (called once)."""
    global WIKI_DIR_FACTOR, WIKI_DIR_STRATEGY, WIKI_DIR_SOURCES, WIKI_DIR_REPRODUCTION
    dirs = _ensure_dirs()
    WIKI_DIR_FACTOR = dirs["factor"]
    WIKI_DIR_STRATEGY = dirs["strategy"]
    WIKI_DIR_SOURCES = dirs["sources"]
    WIKI_DIR_REPRODUCTION = dirs["reproduction"]


# ─── Path helper functions ───────────────────────────────────


def page_path(wiki: Any, dir_name: str, slug: str) -> Path:
    """Get path for a Wiki definition page.

    Args:
        wiki: Wiki instance with wiki_dir attribute.
        dir_name: Directory name (e.g., WIKI_DIR_FACTOR).
        slug: Page slug (e.g., "momentum").

    Returns:
        Path to the Wiki page (e.g., wiki/factor/momentum.md).
    """
    return wiki.wiki_dir / dir_name / f"{slug}.md"


def result_path(wiki: Any, dir_name: str, slug: str, run_id: str) -> Path:
    """Get path for a backtest result page.

    Args:
        wiki: Wiki instance with wiki_dir attribute.
        dir_name: Directory name (e.g., WIKI_DIR_FACTOR).
        slug: Factor/strategy slug.
        run_id: Run identifier (e.g., "20240101-20241231").

    Returns:
        Path to the result page (e.g., wiki/factor/momentum/results/20240101-20241231.md).
    """
    return wiki.wiki_dir / dir_name / slug / "results" / f"{run_id}.md"


def result_dir(wiki: Any, dir_name: str, slug: str) -> Path:
    """Get directory for backtest results.

    Args:
        wiki: Wiki instance with wiki_dir attribute.
        dir_name: Directory name (e.g., WIKI_DIR_FACTOR).
        slug: Factor/strategy slug.

    Returns:
        Path to the results directory (e.g., wiki/factor/momentum/results/).
    """
    return wiki.wiki_dir / dir_name / slug / "results"


def list_pages(wiki: Any, dir_name: str) -> list[dict[str, Any]]:
    """List all pages in a Wiki directory.

    Args:
        wiki: Wiki instance with wiki_dir attribute.
        dir_name: Directory name (e.g., WIKI_DIR_FACTOR).

    Returns:
        List of dicts with keys: slug, path, frontmatter.
    """
    from .utils import parse_frontmatter

    page_dir = wiki.wiki_dir / dir_name
    if not page_dir.is_dir():
        return []

    results = []
    for md in sorted(page_dir.glob("*.md")):
        try:
            content = md.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            if fm:
                fm["_slug"] = md.stem
                fm["_path"] = str(md)
                results.append(fm)
        except OSError as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("could not read %s: %s", md, exc)

    return results


def list_results(wiki: Any, dir_name: str, slug: str) -> list[dict[str, Any]]:
    """List all results for a factor/strategy.

    Args:
        wiki: Wiki instance with wiki_dir attribute.
        dir_name: Directory name (e.g., WIKI_DIR_FACTOR).
        slug: Factor/strategy slug.

    Returns:
        List of dicts with keys: run_id, path, created_at.
    """
    res_dir = result_dir(wiki, dir_name, slug)
    if not res_dir.is_dir():
        return []

    results = []
    for md in sorted(res_dir.glob("*.md")):
        results.append({
            "run_id": md.stem,
            "path": str(md),
        })

    return results


# Initialize constants on import
_init_constants()


__all__ = [
    "WIKI_DIR_FACTOR",
    "WIKI_DIR_STRATEGY",
    "WIKI_DIR_SOURCES",
    "WIKI_DIR_REPRODUCTION",
    "page_path",
    "result_path",
    "result_dir",
    "list_pages",
    "list_results",
    # M4.2 (PR6.7): canonical ~/.quantnodes paths
    "QUANTNODES_HOME",
    "quantnodes_path",
    "ensure_migrated",
]


# ─── M4.2 (PR6.7): canonical ~/.quantnodes paths ─────────────────────────
#
# Above is the legacy Wiki paths module (unchanged).
# Below adds helpers for ~/.quantnodes/ canonical paths + transparent
# symlink migration. See scripts/migrate_llmwikify_paths.py.

import logging
import os
from pathlib import Path as _P

logger = logging.getLogger(__name__)

QUANTNODES_HOME: _P = _P(os.path.expanduser("~/.quantnodes"))


def quantnodes_path(subpath: str) -> _P:
    """Return canonical ``~/.quantnodes/<subpath>``.

    Subpath may contain slashes: ``quantnodes_path("akshare_cache/quantnodes_h5")``.
    Path is created on demand by callers (e.g. via ``mkdir(parents=True)``).
    """
    return QUANTNODES_HOME / subpath


def ensure_migrated(subpath: str) -> _P:
    """Ensure ``~/.quantnodes/<subpath>`` is accessible.

    Idempotent: if symlink/dir already exists, returns as-is.
    Intended for module-level use; failures are logged, not raised.

    NOTE: This function does NOT automatically create symlinks — that is
    handled by ``scripts/migrate_llmwikify_paths.py``. This function
    only ensures the parent directory exists.
    """
    target = quantnodes_path(subpath)
    parent = target.parent if target.suffix else target
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "[paths] cannot create %s: %s (caller will handle)",
                parent, exc,
            )
    return target


# __all__ is defined above (single source of truth, M4.2 merged)

"""One-shot migration: ~/.llmwikify/* → ~/.quantnodes/* (symlink or copy).

M4.2 (PR6.7): after this PR, all code paths hardcode ~/.quantnodes/*.
Users with existing data in ~/.llmwikify/* must run this script ONCE
to create transparent symlinks (zero-copy, recommended) or physically
copy data.

Usage:
    python -m scripts.migrate_llmwikify_paths           # symlink mode (default)
    python -m scripts.migrate_llmwikify_paths --copy    # physical copy
    python -m scripts.migrate_llmwikify_paths --dry-run # show plan only

Idempotent: existing symlinks/dirs are skipped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

LEGACY_HOME = Path.home() / ".llmwikify"
NEW_HOME = Path.home() / ".quantnodes"

# Mapping: (new_subpath, legacy_subpath, kind)
# kind = "dir" or "file"
MIGRATION_PLAN: tuple[tuple[str, str, str], ...] = (
    # Single config files (legacy → new with rename)
    ("llm.json", "llmwikify.json", "file"),
    # Cache directories (direct symlink/copy)
    ("akshare_cache", "akshare_cache", "dir"),
    ("akshare_cache/quantnodes_h5", "akshare_cache/quantnodes_h5", "dir"),
    ("akshare_cache/quantnodes_h5_long", "akshare_cache/quantnodes_h5_long", "dir"),
    ("ifind_cache", "ifind_cache", "dir"),
    ("etf_cache", "etf_cache", "dir"),
    ("cache", "cache", "dir"),
    # Config files (legacy → new same name)
    ("ifind_http.yaml", "ifind_http.yaml", "file"),
    ("clickhouse.yaml", "clickhouse.yaml", "file"),
    ("semantic_registry.yaml", "semantic_registry.yaml", "file"),
    ("factor_cache", "factor_cache", "dir"),
    ("extract_output", "extract_output", "dir"),
    ("mcts_cache", "mcts_cache", "dir"),
    # DB files
    ("monitor.db", "monitor.db", "file"),
    ("reproduction.db", "agent/reproduction.db", "file"),
    # Strategies dir
    ("strategies", "strategies", "dir"),
    # Papers (large dir, typically symlink)
    ("papers", "papers", "dir"),
)


def _plan_entry(new_sub: str, legacy_sub: str, kind: str) -> tuple[Path, Path, str]:
    return (NEW_HOME / new_sub, LEGACY_HOME / legacy_sub, kind)


def _needs_migration(target: Path, source: Path, kind: str) -> bool:
    if target.exists() or target.is_symlink():
        return False
    if not source.exists():
        return False
    return True


def _do_symlink(target: Path, source: Path, kind: str, dry_run: bool) -> str:
    if dry_run:
        return f"[DRY] symlink {target} → {source}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "dir":
        target.symlink_to(source, target_is_directory=True)
    else:
        target.symlink_to(source)
    return f"symlink {target} → {source}"


def _do_copy(target: Path, source: Path, kind: str, dry_run: bool) -> str:
    if dry_run:
        return f"[DRY] copy {source} → {target}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "dir":
        shutil.copytree(source, target, symlinks=False)
    else:
        shutil.copy2(source, target)
    return f"copy {source} → {target}"


def migrate(mode: str = "symlink", dry_run: bool = False) -> int:
    """Run migration. Returns count of items migrated."""
    if mode not in ("symlink", "copy"):
        raise ValueError(f"unknown mode: {mode!r} (use 'symlink' or 'copy')")

    if not LEGACY_HOME.exists():
        print(f"[skip] legacy {LEGACY_HOME} does not exist; nothing to migrate")
        return 0

    print(f"[plan] mode={mode}, dry_run={dry_run}")
    print(f"       legacy: {LEGACY_HOME}")
    print(f"       new:    {NEW_HOME}")
    print()

    count = 0
    for new_sub, legacy_sub, kind in MIGRATION_PLAN:
        target, source, _ = _plan_entry(new_sub, legacy_sub, kind)
        if not _needs_migration(target, source, kind):
            if target.exists() or target.is_symlink():
                pass  # already migrated or new path exists
            continue

        if mode == "symlink":
            msg = _do_symlink(target, source, kind, dry_run)
        else:
            msg = _do_copy(target, source, kind, dry_run)
        print(f"  {msg}")
        count += 1

    print()
    if count == 0:
        print("[done] nothing to migrate (already migrated or no legacy data)")
    else:
        action = "would migrate" if dry_run else "migrated"
        print(f"[done] {action} {count} item(s)")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate ~/.llmwikify/* → ~/.quantnodes/* (M4.2 PR6.7)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Physically copy data (default: symlink, zero-copy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without modifying filesystem",
    )
    args = parser.parse_args(argv)

    mode = "copy" if args.copy else "symlink"
    migrate(mode=mode, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
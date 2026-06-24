#!/usr/bin/env python3
# coding=utf-8
"""Migrate workspace from .quant_agent/ (v2.x) to .agent/ (v3.0.0+ nanobot).

Usage::

    python scripts/migrate_workspace.py [--src .quant_agent] [--dst .agent]
                                        [--backup-keep-days 7] [--dry-run]

What it does
------------
1. Copies files from ``--src`` to ``--dst`` (recursive).
2. Migrates the ``MEMORY.md`` format (v2.x uses a single personality file;
   v3.0.0 splits into ``SOUL.md`` (personality) + ``memory/MEMORY.md``
   (facts)). If the v2 MEMORY.md doesn't exist, no split is done.
3. Renames ``memory/history.jsonl`` → ``memory/history.jsonl`` (unchanged,
   same path under new layout).
4. Preserves ``sessions/*.json`` and ``settings.json`` unchanged.
5. Moves ``.quant_agent/.backup/`` snapshots if present.
6. Records a migration manifest in ``.agent/.migration_manifest.json``.
7. Keeps the source for ``--backup-keep-days`` (default 7) before suggesting
   deletion. By default, the source is NOT deleted — the operator must do
   it manually after verifying the migration succeeded.

Why split MEMORY.md
-------------------
HKUDS/nanobot 0.2.1 distinguishes:
- ``SOUL.md``  — personality / persona (read every turn, never updated by Dream)
- ``USER.md``  — user-specific preferences (tweak prompts)
- ``memory/MEMORY.md``  — long-term facts (consolidated by Dream)

v2.x had a single MEMORY.md that mixed all three. We do a simple heuristic
split: lines starting with ``# `` (headers) and first 100 lines become
SOUL.md (treat as personality), everything else becomes memory/MEMORY.md.

This is a best-effort migration; users should review both files after.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DEFAULT_SRC = ".quant_agent"
DEFAULT_DST = ".agent"
DEFAULT_BACKUP_DAYS = 7

FILES_TO_SKIP = {
    ".migration_manifest.json",
    "MEMORY.md",  # Handled separately (split into SOUL.md + memory/MEMORY.md)
}


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _should_skip(rel: Path) -> bool:
    """Skip migration manifest markers and hidden dotfiles."""
    if rel.name in FILES_TO_SKIP:
        return True
    if rel.name.startswith("."):
        return True
    return False


def _split_memory_md(content: str) -> tuple[str, str]:
    """Split v2.x MEMORY.md content into SOUL.md + memory/MEMORY.md.

    Heuristic: first 100 lines / lines starting with ``# User`` go to SOUL.md,
    the rest goes to memory/MEMORY.md. Best-effort.
    """
    lines = content.splitlines(keepends=True)
    soul_lines: List[str] = []
    fact_lines: List[str] = []
    in_soul = True
    soul_limit = 100
    for line in lines:
        if in_soul and (line.startswith("# ") or len(soul_lines) < soul_limit):
            soul_lines.append(line)
            if line.startswith("# ") and "user" in line.lower():
                in_soul = False
        else:
            in_soul = False
            fact_lines.append(line)
    if not fact_lines and soul_lines:
        fact_lines = ["# Long-term Memory\n\n", "(migrated from v2.x MEMORY.md)\n"]
        soul_lines = ["# Soul\n\n", "(migrated from v2.x MEMORY.md; review and split)\n"]
    return "".join(soul_lines), "".join(fact_lines)


def migrate(src: Path, dst: Path, dry_run: bool = False) -> Dict[str, List[str]]:
    """Copy ``src`` → ``dst`` with MEMORY.md split.

    Returns a dict mapping action → list of affected paths:
    ``{"copied": [...], "skipped": [...], "split": [...]}``.
    """
    if not src.exists():
        logger.warning("Source %s does not exist; nothing to migrate", src)
        return {"copied": [], "skipped": [], "split": []}

    dst.mkdir(parents=True, exist_ok=True)
    actions = {"copied": [], "skipped": [], "split": []}

    for src_file in sorted(src.rglob("*")):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src)

        if _should_skip(rel):
            actions["skipped"].append(str(rel))
            continue

        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if rel.name == "memory" and rel.parts == ("memory",):
            continue
        if rel.parts and rel.parts[0] == "memory" and rel.name == "history.jsonl":
            target = dst / "memory" / "history.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)

        if rel.name == "MEMORY.md":
            content = src_file.read_text(encoding="utf-8", errors="replace")
            soul, facts = _split_memory_md(content)
            (dst / "SOUL.md").write_text(soul, encoding="utf-8")
            (dst / "memory" / "MEMORY.md").write_text(facts, encoding="utf-8")
            (dst / "memory").mkdir(parents=True, exist_ok=True)
            actions["split"].append(str(rel))
            logger.info("Split %s → SOUL.md + memory/MEMORY.md", rel)
            continue

        if not dry_run:
            shutil.copy2(src_file, target)
        actions["copied"].append(str(rel))
        logger.info("Copied %s", rel)

    manifest = {
        "migrated_at": _now_iso(),
        "source": str(src.resolve()),
        "destination": str(dst.resolve()),
        "actions": actions,
        "dry_run": dry_run,
    }
    if not dry_run:
        (dst / ".migration_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    logger.info(
        "Migration done: %d copied, %d skipped, %d split",
        len(actions["copied"]),
        len(actions["skipped"]),
        len(actions["split"]),
    )
    return actions


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate QuantNodes workspace from .quant_agent/ to .agent/"
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"source dir (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"dest dir (default: {DEFAULT_DST})")
    parser.add_argument("--backup-keep-days", type=int, default=DEFAULT_BACKUP_DAYS,
                        help=f"days to keep .quant_agent/ backup (default: {DEFAULT_BACKUP_DAYS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be copied without writing anything")
    parser.add_argument("--delete-src", action="store_true",
                        help=f"delete {DEFAULT_SRC} after migration (only if --backup-keep-days passed)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()

    actions = migrate(src, dst, dry_run=args.dry_run)

    if args.delete_src and not args.dry_run:
        backup_marker = src / ".migrated_at"
        backup_marker.write_text(_now_iso(), encoding="utf-8")
        logger.warning(
            "Source marked as migrated at %s. Manually delete %s after %d days.",
            backup_marker, src, args.backup_keep_days,
        )

    print(json.dumps({
        "src": str(src.resolve()),
        "dst": str(dst.resolve()),
        "dry_run": args.dry_run,
        "copied": len(actions["copied"]),
        "skipped": len(actions["skipped"]),
        "split": len(actions["split"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

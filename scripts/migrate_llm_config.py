#!/usr/bin/env python3
"""Migrate LLM config from legacy ~/.llmwikify/llmwikify.json
to new canonical ~/.quantnodes/llm.json (M3.2).

Usage:
    # Dry run (show what would be written)
    python scripts/migrate_llm_config.py --dry-run

    # Actually write ~/.quantnodes/llm.json
    python scripts/migrate_llm_config.py

    # Overwrite if exists
    python scripts/migrate_llm_config.py --force

The legacy file is NOT deleted — it remains as a Tier-2 fallback.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LEGACY_PATH = Path.home() / ".llmwikify" / "llmwikify.json"
NEW_PATH = Path.home() / ".quantnodes" / "llm.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written")
    parser.add_argument("--force", action="store_true", help="Overwrite existing new config")
    args = parser.parse_args()

    if not LEGACY_PATH.exists():
        print(f"LEGACY: {LEGACY_PATH} not found — nothing to migrate.")
        return

    try:
        data = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: failed to parse {LEGACY_PATH}: {exc}")
        return

    llm_section = data.get("llm")
    if not isinstance(llm_section, dict) or not llm_section:
        print(f"LEGACY: {LEGACY_PATH} has no [llm] section — nothing to migrate.")
        return

    new_data = {"llm": llm_section}

    if NEW_PATH.exists() and not args.force:
        print(f"NEW: {NEW_PATH} already exists (use --force to overwrite).")
        print(f"  Would write: {json.dumps(new_data, indent=2)[:200]}...")
        return

    if args.dry_run:
        print(f"DRY RUN: would write to {NEW_PATH}:")
        print(json.dumps(new_data, indent=2, ensure_ascii=False))
        return

    NEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEW_PATH.write_text(json.dumps(new_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"MIGRATED: {LEGACY_PATH} → {NEW_PATH}")
    print(f"  provider: {llm_section.get('provider')}")
    print(f"  model:    {llm_section.get('model')}")
    print(f"  enabled:  {llm_section.get('enabled')}")
    print(f"  (legacy file preserved as Tier-2 fallback)")


if __name__ == "__main__":
    main()

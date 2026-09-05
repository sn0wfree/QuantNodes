# coding: utf-8
"""build_2026_market_p2.py — 临时库衍生层: option_iv_daily + vrp_daily.

依赖 P1 (原始表) + derived_tables (hv) + derived_factors (pcr) 已完成.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from QuantNodes.strategy.cta.data_loader import CtaDataLoader  # noqa: E402
from QuantNodes.strategy.cta.options.iv_panel import (  # noqa: E402
    build_iv_panel,
    save_iv_panel,
)
from build_vrp_panel import build_vrp_panel, save_vrp_panel  # noqa: E402

TMP_DB = PROJECT_ROOT / "export" / "_tmp_2026.duckdb"


def main() -> int:
    loader = CtaDataLoader(duckdb_path=str(TMP_DB))

    print("== building option_iv_daily ==", flush=True)
    panel = build_iv_panel(loader=loader)
    if panel.empty:
        print("IV panel 为空!")
        return 1
    save_iv_panel(panel, duckdb_path=str(TMP_DB))

    print("== building vrp_daily ==", flush=True)
    df = build_vrp_panel(duckdb_path=str(TMP_DB))
    if df is None or df.empty:
        print("VRP panel 为空!")
        return 1
    save_vrp_panel(df, duckdb_path=str(TMP_DB))
    return 0


if __name__ == "__main__":
    sys.exit(main())

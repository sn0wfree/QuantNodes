# coding=utf-8
"""主 CTA 方向面板快照 — 期权凸性执行层的输入 (A 路线).

复用 v1_2026_oos.py 同管线 (FactorModuleComboStrategy + FINAL_LUMP_CFG),
取 _signals() 方向面板 (-1/0/+1, 品种×日) 存 parquet.

用法: python3 scripts/cta_xsec_option/snapshot_main_cta_signals.py
输出: reports/cta/xsec_option/main_cta_signals.parquet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import fields
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_final_baseline import FINAL_LUMP_CFG  # noqa: E402
from cta_overfit_final_check import CALIBRATED_COSTS  # noqa: E402
from QuantNodes.strategy.cta.data_loader import CtaDataLoader  # noqa: E402
from QuantNodes.strategy.cta.strategies.base import (  # noqa: E402
    DEFAULT_PRODUCTS,
)
from QuantNodes.strategy.cta.strategies.combo import ComboConfig  # noqa: E402
from QuantNodes.strategy.cta.strategies.factor_modules import (  # noqa: E402
    FactorModuleComboStrategy,
)

TMP_DB = "/home/ll/Public/QuantNodes/export/_tmp_2026.duckdb"
BASE_LAMBDA_TV = 0.5
OUT = Path("/home/ll/Public/QuantNodes/reports/cta/xsec_option/main_cta_signals.parquet")


def snapshot_signals(duckdb_path: str, start: str, end: str) -> pd.DataFrame:
    cfg_kw = dict(FINAL_LUMP_CFG)
    cfg_kw["tvpr_lambda_tv"] = BASE_LAMBDA_TV
    cfg_kw["momentum_filter_window"] = 60
    cfg_kw.update(CALIBRATED_COSTS)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    loader = CtaDataLoader(duckdb_path=duckdb_path)
    s = FactorModuleComboStrategy(
        config=cfg, data_loader=loader, products=list(DEFAULT_PRODUCTS),
        start=start, end=end,
    )
    s.load_panel()
    sig = s._signals()
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=TMP_DB)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    sig = snapshot_signals(args.db, args.start, args.end)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(OUT, index=True)
    n_active = (sig != 0).sum(axis=1)
    print(f"信号面板: {sig.shape}, 存在信号日 {(n_active > 0).mean():.1%}, "
          f"日均活跃品种 {n_active.mean():.1f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

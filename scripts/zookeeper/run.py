# coding: utf-8
"""Zookeeper 复现 CLI 入口.

用法:
    python -m scripts.zookeeper.run features       # P2 构造 38 变量面板
    python -m scripts.zookeeper.run train-test     # P3-P4 完整回测
    python -m scripts.zookeeper.run spanning       # P5 vs 全部子策略
    python -m scripts.zookeeper.run all            # 一键全跑

⚠️ P1 脚手架: 仅实装 CLI 框架, 各阶段逻辑在子模块 raise NotImplementedError.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger("zookeeper")


def cmd_features(args) -> int:
    """P2: 构造 38 变量面板 → features_panel.parquet."""
    import pandas as pd
    from scripts.zookeeper import features_c, features_m
    from scripts.zookeeper.config import OUT_DIR

    start = args.start or "2018-01-01"
    end = args.end or "2025-07-21"  # full 库 spot_basis/term_structure 截止日

    LOG.info(f"P2: 构造 Group C/M 变量面板 (38 个, 跳过 S 族), {start} ~ {end}")
    products = features_c.select_top_products(start, end)

    panel_c = features_c.build_group_c(products, start, end)
    panel_m = features_m.build_group_m(start, end, pit_lag=args.pit_lag)

    n_feat = len(panel_c.columns.levels[1].unique())
    LOG.info(f"Group C 面板: {panel_c.shape} (品种={len(products)}, 特征={n_feat})")
    LOG.info(f"Group M 面板: {panel_m.shape}")

    panel_c.to_parquet(OUT_DIR / "features_c.parquet")
    panel_m.to_parquet(OUT_DIR / "features_m.parquet")
    pd.Series(products, name="product").to_csv(OUT_DIR / "product_pool.csv", index=False)
    LOG.info(f"输出: {OUT_DIR}/features_c.parquet, features_m.parquet, product_pool.csv")
    return 0


def cmd_train_test(_args) -> int:
    """P3-P4: 完整 L-En 训练 + 月度回测 → factor_nav.csv."""
    LOG.info("P3-P4: L-En 5 线性模型集成 + 月度截面回测")
    raise SystemExit("P3-P4 待实装")


def cmd_spanning(_args) -> int:
    """P5: Spanning Test vs 全部子策略 → spanning_alpha.csv."""
    LOG.info("P5: Spanning Test vs 全部子策略 + 主 CTA")
    raise SystemExit("P5 待实装")


def cmd_all(args) -> int:
    """一键全跑."""
    cmd_features(args)
    cmd_train_test(args)
    cmd_spanning(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zookeeper 因子复现 (Fan et al. 2026)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-07-21")
    parser.add_argument(
        "--pit-lag", type=int, default=1,
        help="宏观变量滞后月数 (P2 防御性, Spanning 鲁棒性加 2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features", help="P2 构造变量面板")
    sub.add_parser("train-test", help="P3-P4 训练+回测")
    sub.add_parser("spanning", help="P5 Spanning Test")
    sub.add_parser("all", help="一键全跑")
    args = parser.parse_args(argv)
    return {
        "features": cmd_features,
        "train-test": cmd_train_test,
        "spanning": cmd_spanning,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

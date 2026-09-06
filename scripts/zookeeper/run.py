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


def cmd_features(_args) -> int:
    """P2: 构造 38 变量面板 → features_panel.parquet."""
    LOG.info("P2: 构造 Group C/M 变量面板 (38 个, 跳过 S 族)")
    raise SystemExit("P2 待实装 — 见 features_c.py / features_m.py")


def cmd_train_test(_args) -> int:
    """P3-P4: 完整 L-En 训练 + 月度回测 → factor_nav.csv."""
    LOG.info("P3-P4: L-En 5 线性模型集成 + 月度截面回测")
    raise SystemExit("P3-P4 待实装")


def cmd_spanning(_args) -> int:
    """P5: Spanning Test vs 全部子策略 → spanning_alpha.csv."""
    LOG.info("P5: Spanning Test vs 全部子策略 + 主 CTA")
    raise SystemExit("P5 待实装")


def cmd_all(_args) -> int:
    """一键全跑."""
    cmd_features(_args)
    cmd_train_test(_args)
    cmd_spanning(_args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zookeeper 因子复现 (Fan et al. 2026)")
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

# coding=utf-8
"""脚本共享样板: 路径设置 + 参数解析 + 数据加载.

消除所有 v7.x 脚本中的重复样板代码.

用法:
    from _common import setup_path, default_argparser, load_base_data
    REPO = setup_path()
    parser = default_argparser("v7.XX 因子测试")
    X, Y, codes, daily = load_base_data()
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def setup_path() -> Path:
    """设置项目根目录到 sys.path, 返回 REPO 路径."""
    # 向上找 QuantNodes 目录 (支持 scripts/ 或 scripts/research/ 下的脚本)
    repo = Path(__file__).resolve().parent
    while repo.name != "QuantNodes" and repo != repo.parent:
        repo = repo.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def default_argparser(description: str = "v7.x 因子测试") -> argparse.ArgumentParser:
    """创建标准参数解析器 (支持 --fast, --ic-only, --step)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--fast", action="store_true", help="快速模式 (减少资产数)")
    parser.add_argument("--ic-only", action="store_true", help="只计算 IC, 不保存")
    parser.add_argument("--step", type=int, default=20, help="Beta 重估频率 (周)")
    return parser


def load_base_data() -> tuple[np.ndarray, pd.DataFrame, list[str], pd.DataFrame]:
    """加载 v7.10 基础数据 + 日频收益.

    Returns:
        X_v710: (T, N, 36) 周频因子面板
        Y: (T, N) 周频收益 DataFrame
        codes: 资产代码列表
        daily: (T_daily, N) 日频收益 DataFrame
    """
    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
        load_v7_10_data,
        load_daily_etf_returns,
    )
    X, Y, codes = load_v7_10_data()
    daily = load_daily_etf_returns()
    daily = daily[[c for c in codes if c in daily.columns]]
    return X, Y, codes, daily

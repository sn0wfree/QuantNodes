"""OHLCV 前复权单测 (§5.45): 跨 NaN 缺口的水平跳变检测."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from fix_ohlcv_adjust import detect_corp_actions, forward_adjust  # noqa: E402


def _panel_with_jump(gap: bool):
    idx = pd.bdate_range("2021-01-04", periods=20)
    close = np.linspace(2.0, 2.2, 20)
    close[10:] = close[10:] * 0.25          # 事件后水平下移 75%
    if gap:
        close[9] = np.nan                   # 事件前一天缺口 (旧代码漏检)
    data = {f: close.copy() for f in
            ["open", "high", "low", "close", "volume"]}
    cols = pd.MultiIndex.from_product([["512010"],
                                       ["open", "high", "low", "close", "volume"]])
    return pd.DataFrame(
        np.column_stack([data[f] for f in
                         ["open", "high", "low", "close", "volume"]]),
        index=idx, columns=cols)


def test_detect_catches_gap_jump():
    """§5.45: 缺口前的水平跳变也应被检测 (旧代码漏检)."""
    p = _panel_with_jump(gap=True)
    acts = detect_corp_actions(p)
    assert len(acts) == 1
    assert abs(acts[0]["ratio"] - 0.25) < 0.01


def test_detect_catches_normal_jump():
    """无缺口时检测正常."""
    p = _panel_with_jump(gap=False)
    acts = detect_corp_actions(p)
    assert len(acts) == 1


def test_forward_adjust_removes_jump_with_gap():
    """前复权后跨缺口跳变消除 (收益连续)."""
    p = _panel_with_jump(gap=True)
    adj = forward_adjust(p, detect_corp_actions(p))
    c = adj[("512010", "close")].dropna()
    r = c / c.shift(1) - 1
    assert r.abs().max() < 0.5


def test_no_jump_no_action():
    """无跳变 → 无事件."""
    idx = pd.bdate_range("2021-01-04", periods=20)
    cols = pd.MultiIndex.from_product(
        [["510300"], ["open", "high", "low", "close", "volume"]])
    v = np.linspace(1.0, 1.1, 20)
    p = pd.DataFrame(np.column_stack([v] * 5), index=idx, columns=cols)
    assert len(detect_corp_actions(p)) == 0

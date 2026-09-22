"""cash_proxy 延长单测 (§5.47)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


def _splice_returns(old: pd.Series, new: pd.Series) -> pd.Series:
    """与 extend() 同逻辑的纯函数 (供单测)."""
    old_last_date, old_last_val = old.index.max(), old.iloc[-1]
    add = new.loc[new.index > old_last_date]
    rets = new.pct_change(fill_method=None).loc[add.index]
    chained = old_last_val * (1.0 + rets.fillna(0.0)).cumprod()
    m = pd.concat([old, chained])
    return m[~m.index.duplicated(keep="last")]


def test_splice_no_jump_and_preserves_history():
    idx_o = pd.bdate_range("2026-05-01", periods=5)
    old = pd.Series(np.linspace(200.0, 200.2, 5), index=idx_o)
    idx_n = pd.bdate_range("2026-04-01", periods=40)   # 延伸到旧末端之后
    new = pd.Series(np.linspace(300.0, 302.0, 40), index=idx_n)  # 不同水平
    m = _splice_returns(old, new)
    # 历史保留
    pd.testing.assert_series_equal(m.loc[idx_o], old)
    # 断点无跳变 (首日 = 旧末端 × (1+r))
    d1 = m.index[m.index > idx_o.max()][0]
    r = new.pct_change().loc[d1]
    assert abs(m.loc[d1] - old.iloc[-1] * (1 + r)) < 1e-9
    # 收益率无异常
    rr = m.pct_change(fill_method=None).dropna()
    assert rr.abs().max() < 0.05


def test_splice_level_independent():
    """水平基准不同不影响链接后的收益 (常数缩放不变性)."""
    idx_o = pd.bdate_range("2026-05-01", periods=5)
    old = pd.Series([100.0] * 5, index=idx_o)
    idx_n = pd.bdate_range("2026-05-01", periods=10)
    new = pd.Series(np.linspace(50.0, 50.5, 10), index=idx_n)
    m1 = _splice_returns(old, new)
    m2 = _splice_returns(old * 3.0, new)
    r1 = m1.pct_change(fill_method=None).fillna(0)
    r2 = m2.pct_change(fill_method=None).fillna(0)
    pd.testing.assert_series_equal(r1, r2)

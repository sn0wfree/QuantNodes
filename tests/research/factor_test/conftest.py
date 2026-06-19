# coding: utf-8
"""Shared fixtures for factor_test tests.

历史来源: 迁移自 ``QuantNodes/research/factor_test/tests/conftest.py`` (C2 收敛, 2026-06-19).
提供:
- ``synthetic_data``: 120 天 × 30 股票的完整合成数据 (固定日期 2026-01 起, 用于节点单元测试)
- ``synthetic_rng``: session 级 np RandomState
- ``stub_fetcher``: IFindFetcherStub 实例 (无网络)
- ``stub_ifind_db``: 基于 stub 的 IFinDDatabase (单元测试用)
- ``node_data``: 节点 fixture 别名 = synthetic_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── 合成数据 Fixtures ──────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_rng():
    """Session 级随机种子 (np RandomState)."""
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_data(synthetic_rng):
    """120 天 × 30 股票的完整合成数据, 固定日期 2026-01 起.

    含: factor / price / id_citic1 / mv_float / st / suspend / ud_limit /
        ipo_days / index_cp / stklist / trade_dt

    注: 日期固定为 2026-01-04 起的 120 个工作日 (与 production 无关,
    仅为节点单元测试提供一致的输入).
    """
    n_days, n_stocks = 120, 30
    dates = [
        int(d.strftime('%Y%m%d'))
        for d in pd.bdate_range('2026-01-04', periods=n_days)
    ]
    stocks = list(range(100001, 100001 + n_stocks))

    factor = synthetic_rng.randn(n_days, n_stocks) + \
        np.linspace(0, 0.5, n_days).reshape(-1, 1)
    price = 100 * np.exp(np.cumsum(
        synthetic_rng.randn(n_days, n_stocks) * 0.02, axis=0))
    industry = synthetic_rng.randint(1, 31, (n_days, n_stocks))
    mv = synthetic_rng.lognormal(10, 1, (n_days, n_stocks))
    st = np.zeros((n_days, n_stocks), dtype=int)
    st[:, :2] = 1
    suspend = np.zeros((n_days, n_stocks), dtype=int)
    suspend[5:8, 3] = 1
    ud_limit = np.zeros((n_days, n_stocks), dtype=int)
    ipo_days = np.ones((n_days, n_stocks), dtype=int) * 500
    index_cp = pd.DataFrame({
        '000300.SH': 3500 + np.cumsum(synthetic_rng.randn(n_days) * 10),
        '000905.SH': 6000 + np.cumsum(synthetic_rng.randn(n_days) * 15),
    }, index=dates)

    return {
        'dates': dates,
        'stocks': stocks,
        'factor': pd.DataFrame(factor, index=dates, columns=stocks),
        'price': pd.DataFrame(price, index=dates, columns=stocks),
        'id_citic1': pd.DataFrame(industry, index=dates, columns=stocks),
        'mv_float': pd.DataFrame(mv, index=dates, columns=stocks),
        'st': pd.DataFrame(st, index=dates, columns=stocks),
        'suspend': pd.DataFrame(suspend, index=dates, columns=stocks),
        'ud_limit': pd.DataFrame(ud_limit, index=dates, columns=stocks),
        'ipo_days': pd.DataFrame(ipo_days, index=dates, columns=stocks),
        'index_cp': index_cp,
        'stklist': pd.DataFrame(stocks, columns=[0]),
        'trade_dt': pd.DataFrame(dates, columns=[0]),
        '_loader': None,
    }


@pytest.fixture
def synthetic_context(synthetic_data):
    """基于合成数据的标准 pipeline context 别名."""
    return dict(synthetic_data)


@pytest.fixture
def node_data(synthetic_data):
    """为节点 fixture 提供标准化输入 (别名)."""
    return synthetic_data


# ── iFinD Stub Fixtures ───────────────────────────────────────


@pytest.fixture
def stub_fetcher():
    """测试用 stub fetcher (无网络, 默认空响应)."""
    from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcherStub
    return IFindFetcherStub()


@pytest.fixture
def stub_ifind_db(stub_fetcher):
    """基于 stub 的 IFinDDatabase, 用于单元测试."""
    from QuantNodes.research.factor_test.ifind_db.ifind_database import IFinDDatabase
    return IFinDDatabase(
        date_beg='20260101',
        date_end='20260630',
        universe='沪深300',
        fetcher=stub_fetcher,
    )


# 注: 真实 iFinD fetcher/db fixtures 在 ``tests/research/factor_test/ifind_db/test_ifind_integration.py``
# (需要 API key, 默认 skip).
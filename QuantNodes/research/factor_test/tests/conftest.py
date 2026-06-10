# coding: utf-8
"""共享测试 fixtures - 合成数据 + iFinD 真实数据"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保 factor_test 在 path 中
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcherStub
from QuantNodes.research.factor_test.ifind_db.ifind_database import IFinDDatabase


# ── 合成数据 Fixtures ──────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_rng():
    """Session 级随机种子"""
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_data(synthetic_rng):
    """120 天 × 30 股票的完整合成数据"""
    n_days, n_stocks = 120, 30
    dates = [int(d.strftime('%Y%m%d'))
             for d in pd.bdate_range('2026-01-04', periods=n_days)]
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
    """基于合成数据的标准 pipeline context"""
    ctx = dict(synthetic_data)
    return ctx


# ── iFinD 数据 Fixtures ───────────────────────────────────────

@pytest.fixture(scope="session")
def ifind_fetcher():
    """Session 级 iFinD Fetcher (真实 API)"""
    from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcher
    try:
        fetcher = IFindFetcher()
        return fetcher
    except (FileNotFoundError, ValueError):
        pytest.skip("iFinD API key 未配置")


@pytest.fixture(scope="session")
def ifind_db(ifind_fetcher):
    """Session 级 iFinD 数据库 (真实 API, 带缓存)"""
    db = IFinDDatabase(
        date_beg='20260101',
        date_end='20260630',
        universe='沪深300',
        fetcher=ifind_fetcher,
    )
    return db


@pytest.fixture
def stub_fetcher():
    """测试用 stub fetcher"""
    return IFindFetcherStub()


@pytest.fixture
def stub_ifind_db(stub_fetcher):
    """基于 stub 的 IFinDDatabase, 用于单元测试"""
    return IFinDDatabase(
        date_beg='20260101',
        date_end='20260630',
        universe='沪深300',
        fetcher=stub_fetcher,
    )


# ── Node 执行 Fixtures ────────────────────────────────────────

@pytest.fixture
def node_data(synthetic_data):
    """为每个节点提供标准化输入"""
    return synthetic_data

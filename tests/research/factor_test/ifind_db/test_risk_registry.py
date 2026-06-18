# coding: utf-8
"""P-4: IFinDDatabase risk_registry 构造参数化 (Phase 3.1)

风险因子注册表从 get_apikeys() 内部硬编码 list 改为
构造参数 risk_registry, None=默认 10 Barra 风格因子.
"""
from __future__ import annotations

import pytest

from QuantNodes.research.factor_test.ifind_db import (
    IFinDDatabase, IFindFetcherStub,
)


@pytest.fixture
def stub_db():
    """最小 stub IFinDDatabase (无网络调用)"""
    return IFinDDatabase(
        date_beg='20260101', date_end='20260630',
        universe='all', fetcher=IFindFetcherStub(),
    )


class TestRiskRegistryDefault:
    """P-4: 默认 10 Barra 风格风险因子"""

    def test_default_10_factors(self, stub_db):
        """默认 get_apikeys 返回 10 个 Barra 风格因子"""
        keys = stub_db.get_apikeys('risk_factor.h5')
        assert len(keys) == 10
        # 包含核心因子
        for expected in ['/beta', '/momentum', '/size', '/volatility',
                         '/value', '/quality', '/growth', '/leverage',
                         '/liquidity', '/non_linear_size']:
            assert expected in keys

    def test_default_keys_have_leading_slash(self, stub_db):
        """默认 key 保持 '/xxx' 格式 (与 iFinD API 一致)"""
        keys = stub_db.get_apikeys('risk_factor.h5')
        for k in keys:
            assert k.startswith('/'), f"Risk factor key {k} missing leading /"


class TestRiskRegistryCustom:
    """P-4: 自定义 risk_registry 覆盖默认"""

    def test_custom_single_factor(self, stub_db_kwargs=None):
        """自定义 1 个因子"""
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            risk_registry=['/custom_factor'],
        )
        keys = db.get_apikeys('risk_factor.h5')
        assert keys == ['/custom_factor']

    def test_custom_multiple_factors(self):
        """自定义多个因子"""
        custom = ['/a', '/b', '/c', '/d']
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            risk_registry=custom,
        )
        keys = db.get_apikeys('risk_factor.h5')
        assert keys == custom

    def test_empty_registry(self):
        """空列表 → get_apikeys 返回 []"""
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            risk_registry=[],
        )
        keys = db.get_apikeys('risk_factor.h5')
        assert keys == []


class TestRiskRegistryIsolation:
    """P-4: 自定义 registry 不影响默认实例"""

    def test_custom_does_not_mutate_default(self):
        """多个实例 registry 独立 (避免 mutable default 陷阱)"""
        custom1 = ['/x1']
        custom2 = ['/y1', '/y2']
        db1 = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            risk_registry=custom1,
        )
        db2 = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            risk_registry=custom2,
        )
        # db1 / db2 互不影响
        assert db1.get_apikeys('x.h5') == ['/x1']
        assert db2.get_apikeys('x.h5') == ['/y1', '/y2']
        # 修改外部 list 不影响实例
        custom1.append('/mutated')
        assert '/mutated' not in db1.get_apikeys('x.h5'), (
            "Constructor must copy list to prevent external mutation"
        )

    def test_get_apikeys_returns_copy(self):
        """get_apikeys() 返回 list copy (防止外部修改影响内部)"""
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
        )
        keys = db.get_apikeys('x.h5')
        keys.append('/hacked')
        # 内部 _risk_registry 不应被污染
        keys2 = db.get_apikeys('x.h5')
        assert '/hacked' not in keys2

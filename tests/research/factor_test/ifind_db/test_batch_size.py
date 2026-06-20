# coding: utf-8
"""Phase 3.2 M10: IFinDDatabase batch_size 参数化"""
from __future__ import annotations


from QuantNodes.research.factor_test.ifind_db import IFinDDatabase, IFindFetcherStub


class TestBatchSizeParameter:
    """M10: batch_size 可自定义，默认 50"""

    def test_default_batch_size(self):
        """默认 50"""
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
        )
        assert db._batch_size == 50

    def test_custom_batch_size(self):
        """自定义 20"""
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            batch_size=20,
        )
        assert db._batch_size == 20

    def test_custom_batch_size_100(self):
        """自定义 100 (大批次)"""
        db = IFinDDatabase(
            date_beg='20260101', date_end='20260630',
            universe='all', fetcher=IFindFetcherStub(),
            batch_size=100,
        )
        assert db._batch_size == 100

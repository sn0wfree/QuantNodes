# coding: utf-8
"""Test that rate_limit and cache_ttl are configurable on IFindFetcher (H17/H18 HIGH)"""
from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcher


def test_default_values():
    """Default values match audit expectation"""
    fetcher = IFindFetcher()
    assert fetcher.DEFAULT_RATE_LIMIT_S == 0.5
    assert fetcher.DEFAULT_CACHE_TTL_S == 7 * 86400
    assert fetcher.rate_limit_s == 0.5
    assert fetcher.cache_ttl_s == 7 * 86400


def test_custom_rate_limit():
    """Custom rate limit is accepted"""
    fetcher = IFindFetcher(rate_limit_s=0.1)
    assert fetcher.rate_limit_s == 0.1


def test_custom_cache_ttl():
    """Custom cache TTL is accepted"""
    fetcher = IFindFetcher(cache_ttl_s=24 * 3600)
    assert fetcher.cache_ttl_s == 24 * 3600


def test_both_custom():
    """Both rate and cache can be customized"""
    fetcher = IFindFetcher(rate_limit_s=0.25, cache_ttl_s=14 * 86400)
    assert fetcher.rate_limit_s == 0.25
    assert fetcher.cache_ttl_s == 14 * 86400

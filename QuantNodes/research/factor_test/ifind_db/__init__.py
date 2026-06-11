# coding: utf-8
"""iFinD Database - 包装同花顺 iFinD API 为 DataLoader 兼容接口"""

from .ifind_database import IFinDDatabase
from .fetcher import IFindFetcher, IFindFetcherStub

__all__ = ['IFinDDatabase', 'IFindFetcher', 'IFindFetcherStub']

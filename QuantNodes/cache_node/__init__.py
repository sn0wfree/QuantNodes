# coding=utf-8
"""
cache_node - 行情数据缓存节点

提供 Parquet 文件缓存能力, 支持:
- 透明代理模式: ConfigBacktestTool 自动调用
- 独立 Pipeline 节点: YAML 配置驱动
"""

from QuantNodes.cache_node.base import MarketDataCacheNode
from QuantNodes.cache_node.cache_store import ParquetCacheStore
from QuantNodes.cache_node.metadata import CacheMetadata, CacheMeta

__all__ = [
    'MarketDataCacheNode',
    'ParquetCacheStore',
    'CacheMetadata',
    'CacheMeta',
]

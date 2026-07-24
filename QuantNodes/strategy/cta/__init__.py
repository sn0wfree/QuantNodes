# coding=utf-8
"""CTA 策略研究框架 (Strategy / Commodity Trading Advisor).

数据源:
    ~/Public/DataCache/futures_options_daily.duckdb
    - 91 个产品 × {futures, options} × 147 张表
    - 统一视图 v_all_futures / v_all_options
    - contract_specs (multiplier / tick / price_unit)

提供"原子抽取"函数, 不做面板聚合 (策略代码自行决定):
    from QuantNodes.strategy.cta import CtaDataLoader
    dl = CtaDataLoader()
    df = dl.load_futures_daily('IF')

版本:
    - v0 (2026-07-23): 初始 — 仅数据加载与合约参数访问
"""
from __future__ import annotations

from .contract_specs import (
    ContractSpec,
    contract_multiplier,
    contract_spec_dataframe,
    load_contract_specs,
    tick_size,
)
from .data_loader import (
    CtaDataLoader,
    DEFAULT_DUCKDB_PATH,
    ProductKind,
    ProductTable,
)

__all__ = [
    # data_loader
    "CtaDataLoader",
    "DEFAULT_DUCKDB_PATH",
    "ProductKind",
    "ProductTable",
    # contract_specs
    "ContractSpec",
    "contract_multiplier",
    "contract_spec_dataframe",
    "load_contract_specs",
    "tick_size",
]

# coding=utf-8
"""产业链组 — 盘面利润/利润动量/比价/上下游传导.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.industrial_chain import (
    ChainMarginFactor, LeadLagFactor, MarginMomentumFactor, PriceRatioFactor,
)


class IndustrialChainModuleStrategy(_ModuleStrategy):
    """产业链组: 盘面利润/利润动量/比价/上下游传导.

    信号 → 品种映射 (CHAIN_MAP):
        chain_margin_rb  → RB   (螺纹盘面利润, 交易下游成品)
        chain_margin_hc  → HC   (热卷盘面利润)
        chain_margin_pp  → PP   (MTO 利润)
        margin_momentum_m → M   (压榨利润动量)
        price_ratio_au_cu → AU-CU 配对 (比价高估 → 空 AU 多 CU)
        lead_lag_i_rb    → RB   (铁矿传导顺畅 → 做多下游螺纹)
        chain_margin_j   → J    (焦化利润)
        chain_margin_ta  → TA   (PTA加工费)
        chain_margin_lh  → LH   (生猪饲料成本)
    """

    FACTORS = [
        # 钢铁链: 铁矿→焦煤→焦炭→螺纹/热卷
        ("chain_margin_rb", lambda: ChainMarginFactor("steel_profit_rb"), "chain"),
        ("chain_margin_hc", lambda: ChainMarginFactor("steel_profit_hc"), "chain"),
        # 化工链: 石化→化工
        ("chain_margin_pp", lambda: ChainMarginFactor("mto_profit"), "chain"),
        # 农产品链: 大豆压榨
        ("margin_momentum_m", lambda: MarginMomentumFactor("crush_profit"), "chain"),
        # 跨品种
        ("price_ratio_au_cu", lambda: PriceRatioFactor("AU", "CU"), "chain"),
        ("lead_lag_i_rb", lambda: LeadLagFactor("I", "RB"), "chain"),
        # 新增: 焦化链
        ("chain_margin_j", lambda: ChainMarginFactor("coke_profit"), "chain"),
        # 新增: PTA加工费
        ("chain_margin_ta", lambda: ChainMarginFactor("pta_fee"), "chain"),
        # 新增: 生猪饲料成本
        ("chain_margin_lh", lambda: ChainMarginFactor("hog_feed_cost"), "chain"),
    ]

    CHAIN_MAP = {
        "chain_margin_rb": "RB",       # 螺纹盘面利润 → RB
        "chain_margin_hc": "HC",       # 热卷盘面利润 → HC
        "chain_margin_pp": "PP",       # MTO利润 → PP
        "margin_momentum_m": "M",      # 大豆压榨利润 → M
        "price_ratio_au_cu": "AU-CU",  # 比价 → AU和CU
        "lead_lag_i_rb": "RB",         # 铁矿传导 → RB
        "chain_margin_j": "J",         # 焦化利润 → J
        "chain_margin_ta": "TA",       # PTA加工费 → TA
        "chain_margin_lh": "LH",       # 生猪饲料成本 → LH
    }

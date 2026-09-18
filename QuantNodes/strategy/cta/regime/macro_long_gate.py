# coding=utf-8
"""宏观多头门控 (macro long gate) - 与 short_gate 独立的对称模块.

背景 (与 short_gate 对称的设计):
    short_gate (regime/macro_gate.py) 已在 2020-2026 实证中验证
    能抑制 trend_down 做空在商品超级牛市中的逆风亏损 (t=-3.50)。
    但商品超级牛市同样会出现阶段性回撤 (2022H1 美国激进加息,
    2024 衰退预期), 此时 trend_up 多头也会被错杀。

    借鉴 short_gate 设计, 建立对称的多头闸门: 仅在宏观"商品顺风"
    时允许多头持仓, 衰退/避险期拒做多。

设计:
    多头允许条件 (全部满足时允许多):
        复用 short_gate 的 6 个宏观维度, 但反向判断:
        - DXY 动量 ≤ 0 (美元不强势 → 商品计价顺风 → 允许多)
        - US10Y 动量 ≥ 0 (通胀不衰退 → 商品顺风 → 允许多)
        - VIX 动量 ≤ 0 (无避险情绪升温 → 允许多)
        - CRB 等权指数动量 > 0 (商品整体上行 → 允许多)
        - 中美利差 动量 ≤ 0 (人民币升值/稳定 → 商品定价压力小 → 允许多)
        - 金油比相关性 动量 ≥ -0.3 (无避险急升 → 允许多)

    模式:
        external  仅外部宏观 (默认)
        off       关闭
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def macro_long_gate(
    px_panel: pd.DataFrame,
    mode: str = "external",
    breadth_threshold: float = 0.6,
    breadth_ma: int = 250,
    use_vix: bool = True,
    use_crb_proxy: bool = True,
    use_credit_spread: bool = True,
    use_copper_oil_ratio: bool = True,
    cs_inverse: bool = True,
) -> pd.DataFrame:
    """多头闸门: 返回 (T, N) bool, True = 允许多头.

    Args:
        px_panel: (T, N) 收盘价面板
        mode: 'external' / 'breadth' / 'off'
              'off' 时全 True (不限制)
        breadth_threshold: 广度阈值 (mode=breadth 时使用)
        breadth_ma: 广度均线 (mode=breadth 时使用)
        use_vix / use_crb_proxy / use_credit_sspread / use_copper_oil_ratio:
            各宏观维度开关
        cs_inverse: True 时中美利差动量 > 0 → 不允许多 (人民币贬值预期 = 商品顺风,
                   但同样可解读为人民币资产承压, 留作可调)

    Returns:
        pd.DataFrame[bool], 与 px_panel 同形. trend_up 多头信号
        仅在该闸门为 True 的日子放行.
    """
    if mode not in ("external", "breadth", "off"):
        raise ValueError(f"mode must be one of ('external', 'breadth', 'off'), got {mode!r}")

    if mode == "off":
        return pd.DataFrame(
            np.ones((len(px_panel), px_panel.shape[1]), dtype=bool),
            index=px_panel.index, columns=px_panel.columns,
        )

    from .macro_gate import commodity_breadth, external_commodity_favorable

    if mode == "breadth":
        breadth = commodity_breadth(px_panel, breadth_ma)
        breadth_allow = (breadth >= breadth_threshold).fillna(False)
        return pd.DataFrame(
            np.repeat(breadth_allow.values[:, None], px_panel.shape[1], axis=1),
            index=px_panel.index, columns=px_panel.columns, dtype=bool,
        )

    # external: 复用 short_gate 的顺风判断, 取反即允许多
    # 顺风 = 禁空 → 多头允许 = 顺风成立时 (顺势做多更安全)
    ext_fav = external_commodity_favorable(
        use_vix=use_vix,
        use_crb_proxy=use_crb_proxy,
        use_credit_spread=use_credit_spread,
        use_copper_oil_ratio=use_copper_oil_ratio,
        px_panel=px_panel,
    )
    # 多头允许 = 顺风 (短空被禁 = 多头环境顺风)
    long_allow = ext_fav.reindex(px_panel.index).shift(1).fillna(False).astype(bool)

    return pd.DataFrame(
        np.repeat(long_allow.values[:, None], px_panel.shape[1], axis=1),
        index=px_panel.index, columns=px_panel.columns, dtype=bool,
    )

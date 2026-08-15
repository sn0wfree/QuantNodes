# coding=utf-8
"""Options Factor Filter: 期权因子 → 期货仓位过滤层.

用期权因子 (Skew/VRP/GEX/PCVR) 修改 CTA 仓位系数:
    - Skew 极端 → 减仓 (反转风险)
    - VRP 同向 → 加仓 (期权市场确认)
    - GEX 极端 → 减仓 (gamma squeeze 风险)
    - PCVR 极端 → 反向调整 (情绪反转)

集成方式: 在 FactorModuleComboStrategy 的 vol_target 之后调用.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..options.option_factors import (
    skew_zscore,
    iv_zscore,
    compute_gex,
    gex_zscore,
    put_call_volume_ratio,
)
from ..options.hedge_data import load_vrp_panel


class OptionsFactorFilter:
    """期权因子过滤层: 修改 CTA 仓位系数."""

    def __init__(
        self,
        skew_window: int = 60,
        skew_threshold: float = 1.5,
        vrp_boost: float = 1.2,
        gex_threshold: float = 1.5,
        gex_reduce: float = 0.7,
        pcvr_threshold: float = 1.3,
        pcvr_reduce: float = 0.8,
    ):
        self.skew_window = skew_window
        self.skew_threshold = skew_threshold
        self.vrp_boost = vrp_boost
        self.gex_threshold = gex_threshold
        self.gex_reduce = gex_reduce
        self.pcvr_threshold = pcvr_threshold
        self.pcvr_reduce = pcvr_reduce

    def filter(
        self,
        weights: pd.DataFrame,
        iv_panel: pd.DataFrame | None = None,
        skew_panel: pd.DataFrame | None = None,
        vrp_panel: pd.DataFrame | None = None,
        gex_series: pd.Series | None = None,
        pcvr_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """过滤 CTA 仓位系数.

        Args:
            weights: CTA vol_target 输出 (date × product), 值为仓位系数
            iv_panel: IV 面板 (date × product)
            skew_panel: Skew 面板 (date × product)
            vrp_panel: VRP 面板 (date × product)
            gex_series: GEX 时间序列
            pcvr_series: P/C Volume Ratio 时间序列

        Returns:
            调整后的权重矩阵 (同 shape)
        """
        result = weights.copy()
        factors = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)

        for product in weights.columns:
            # 1. Skew 过滤: |skew_z| > threshold → 减仓
            if skew_panel is not None and product in skew_panel.columns:
                skew_z = skew_zscore(skew_panel[[product]], window=self.skew_window)
                if product in skew_z.columns:
                    extreme = skew_z[product].abs() > self.skew_threshold
                    # 极端 skew 时减仓 50%
                    factors[product] = np.where(
                        extreme.reindex(factors.index, fill_value=False),
                        0.5,
                        factors[product],
                    )

            # 2. VRP 确认: VRP 与 CTA 同向 → 加仓
            if vrp_panel is not None and product in vrp_panel.columns:
                vrp = vrp_panel[product].reindex(factors.index)
                cta_sign = np.sign(weights[product])
                vrp_sign = np.sign(vrp)
                aligned = (vrp_sign == cta_sign) & (vrp_sign != 0) & vrp.notna()
                factors[product] = np.where(aligned, self.vrp_boost, factors[product])

            # 3. GEX 过滤: |gex_z| > threshold → 减仓
            if gex_series is not None and not gex_series.empty:
                gex_z = gex_zscore(gex_series, window=60)
                extreme_gex = gex_z.abs() > self.gex_threshold
                # GEX 影响所有品种 (系统性风险)
                for col in factors.columns:
                    factors[col] = np.where(
                        extreme_gex.reindex(factors.index, fill_value=False),
                        self.gex_reduce,
                        factors[col],
                    )
                break  # GEX 是系统性信号, 只应用一次

            # 4. PCVR 过滤: 极端 P/C 比率 → 反向调整
            if pcvr_series is not None and not pcvr_series.empty:
                extreme_pcvr = (pcvr_series > self.pcvr_threshold) | (pcvr_series < 1.0 / self.pcvr_threshold)
                for col in factors.columns:
                    factors[col] = np.where(
                        extreme_pcvr.reindex(factors.index, fill_value=False),
                        self.pcvr_reduce,
                        factors[col],
                    )
                break  # PCVR 是系统性信号

        return result * factors


def build_filter_data(
    products: list[str],
    start: str,
    end: str,
    option_data: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """构建 filter 所需数据.

    Returns:
        dict 含 iv_panel/skew_panel/vrp_panel/gex_series/pcvr_series
    """
    result = {}

    # IV 面板
    try:
        from ..options.iv_panel import load_iv_panel
        result["iv_panel"] = load_iv_panel(start=start, end=end)
    except Exception:
        pass

    # VRP 面板
    try:
        result["vrp_panel"] = load_vrp_panel(products=products, start=start, end=end)
    except Exception:
        pass

    # Skew 面板
    if option_data:
        for prod, df in option_data.items():
            if not df.empty:
                df2 = df.copy()
                if "dt" not in df2.columns and "trade_date" in df2.columns:
                    df2["dt"] = df2["trade_date"]
                if "k" not in df2.columns and "strike_price" in df2.columns:
                    df2["k"] = df2["strike_price"]
                if "bs_iv" not in df2.columns and "close" in df2.columns:
                    df2["bs_iv"] = df2["close"]
                result["skew_panel"] = skew_rolling_panel(df2, window=20)
                break

    # P/C Volume Ratio
    if option_data:
        for prod, df in option_data.items():
            if not df.empty and "volume" in df.columns:
                df2 = df.copy()
                if "dt" not in df2.columns and "trade_date" in df2.columns:
                    df2["dt"] = df2["trade_date"]
                if "cp" not in df2.columns and "option_type" in df2.columns:
                    df2["cp"] = df2["option_type"]
                result["pcvr_series"] = put_call_volume_ratio(df2)
                break

    return result


# 导入 (避免循环)
from ..options.option_factors import skew_rolling_panel

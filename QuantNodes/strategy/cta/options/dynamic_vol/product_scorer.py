"""品种评分器

根据VRP、流动性、IV适配度、数据历史长度为品种打分
动态调整最多品种数（基于账户规模）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .regime_detector import Regime

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """品种评分配置"""
    # 评分权重
    weight_vrp: float = 0.3
    weight_liquidity: float = 0.2
    weight_iv_fit: float = 0.3
    weight_history: float = 0.2

    # 最低入选门槛
    min_score: float = 0.3

    # 账户规模→最大品种数映射
    max_products_by_equity: Dict[float, int] = field(default_factory=lambda: {
        3_000_000: 3,
        8_000_000: 5,
        20_000_000: 8,
        float('inf'): 10,
    })

    # 最少需要的历史天数
    min_history_days: int = 60


class ProductScorer:
    """品种评分器

    输入：日期、市场状态、IV面板、VRP面板、成交量数据
    输出：按评分排序的品种列表
    """

    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()

    def get_max_products(self, equity: float) -> int:
        """根据账户规模返回最大品种数"""
        for threshold, max_n in self.config.max_products_by_equity.items():
            if equity <= threshold:
                return max_n
        return 5  # 默认

    def score_products(
        self,
        date: pd.Timestamp,
        regime: Regime,
        equity: float,
        products: List[str],
        iv_panel: pd.DataFrame,
        vrp_panel: Optional[pd.DataFrame] = None,
        volume_data: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """评分并选择Top N品种

        Parameters
        ----------
        date : 当前日期
        regime : 当前市场状态
        equity : 账户净值
        products : 候选品种列表
        iv_panel : IV面板 (date x product)
        vrp_panel : VRP面板 (date x product)
        volume_data : {product: volume} 成交量数据

        Returns
        -------
        评分排序后的品种列表（最多max_products个）
        """
        max_products = self.get_max_products(equity)
        scores = {}

        for product in products:
            score = self._score_single(
                product, date, regime, iv_panel, vrp_panel, volume_data
            )
            if score >= self.config.min_score:
                scores[product] = score

        # 按评分降序排序
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        selected = [p for p, s in sorted_products[:max_products]]
        logger.debug(
            f"Date {date}: selected {len(selected)}/{len(products)} products "
            f"(max={max_products}): {selected}"
        )
        return selected

    def _score_single(
        self,
        product: str,
        date: pd.Timestamp,
        regime: Regime,
        iv_panel: pd.DataFrame,
        vrp_panel: Optional[pd.DataFrame],
        volume_data: Optional[Dict[str, float]],
    ) -> float:
        """计算单个品种的综合评分"""
        cfg = self.config

        # 1. VRP评分
        vrp_score = self._score_vrp(product, date, vrp_panel)

        # 2. 流动性评分
        liq_score = self._score_liquidity(product, volume_data)

        # 3. IV适配度评分
        iv_fit_score = self._score_iv_fit(product, date, regime, iv_panel)

        # 4. 数据历史评分
        history_score = self._score_history(product, iv_panel)

        score = (vrp_score * cfg.weight_vrp
                 + liq_score * cfg.weight_liquidity
                 + iv_fit_score * cfg.weight_iv_fit
                 + history_score * cfg.weight_history)

        return score

    def _score_vrp(
        self, product: str, date: pd.Timestamp,
        vrp_panel: Optional[pd.DataFrame]
    ) -> float:
        """VRP评分：VRP越高越适合卖方

        VRP = IV - HV20
        归一化到 [0, 1]：[-0.1, 0.1] -> [0, 1]
        """
        if vrp_panel is None or product not in vrp_panel.columns:
            return 0.5
        if date not in vrp_panel.index:
            return 0.5
        vrp = vrp_panel.loc[date, product]
        if pd.isna(vrp):
            return 0.5
        # 归一化：[-0.1, 0.1] -> [0, 1]
        return float(np.clip((vrp + 0.1) / 0.2, 0.0, 1.0))

    def _score_liquidity(
        self, product: str, volume_data: Optional[Dict[str, float]]
    ) -> float:
        """流动性评分：成交量越大越好

        归一化：1万手 = 1.0
        """
        if volume_data is None:
            return 0.5
        vol = volume_data.get(product, 0)
        return float(np.clip(vol / 10000, 0.0, 1.0))

    def _score_iv_fit(
        self, product: str, date: pd.Timestamp,
        regime: Regime, iv_panel: pd.DataFrame
    ) -> float:
        """IV适配度评分：当前状态下IV是否适合

        - 卖方状态(CALM/TRANSITION/HIGH_VOL/COMPRESSION)：IV越高越好
        - 买方状态(VOL_SPIKE)：IV越低越好
        """
        if product not in iv_panel.columns:
            return 0.5
        series = iv_panel[product].dropna()
        valid = series.loc[:date]
        if len(valid) < 60:
            return 0.5

        # 计算IV百分位
        current = valid.iloc[-1]
        iv_pct = (valid < current).sum() / len(valid)

        if regime == Regime.VOL_SPIKE:
            # 买方：IV越低越好
            return 1.0 - iv_pct
        else:
            # 卖方：IV越高越好
            return iv_pct

    def _score_history(self, product: str, iv_panel: pd.DataFrame) -> float:
        """数据历史评分：>200天为满分"""
        if product not in iv_panel.columns:
            return 0.0
        series = iv_panel[product].dropna()
        return float(np.clip(len(series) / 200, 0.0, 1.0))

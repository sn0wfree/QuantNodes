"""市场状态识别器

5种状态：CALM / TRANSITION / HIGH_VOL / VOL_SPIKE / COMPRESSION
IV Crush预警：检测IV从极端高位回落
冷静期管理：连续IV crush后自动减仓
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(IntEnum):
    CALM = 0           # 平静：IV低，环境稳定，卖put spread
    TRANSITION = 1     # 过渡：IV缓慢上升，卖put+买call翼
    HIGH_VOL = 2       # 高波：IV高位稳定，卖铁鹰
    VOL_SPIKE = 3      # 波动爆发：IV快速扩张，买跨式
    COMPRESSION = 4    # 压缩：IV从高位回落，卖宽跨


@dataclass
class RegimeConfig:
    """状态识别配置"""
    # IV百分位阈值
    iv_pct_calm: float = 0.30       # <30% = 平静
    iv_pct_high: float = 0.75       # >75% = 高波/压缩

    # IV动量阈值
    iv_mom平稳: float = 0.05        # |iv_mom_20d| < 5% = 平稳
    iv_mom上升: float = 0.05        # iv_mom_20d > 5% = 上升
    iv_mom_spike: float = 0.10      # |iv_mom_5d| > 10% = 爆发

    # VIX阈值
    vix_pct_calm: float = 0.20      # VIX百分位<20% = 平静

    # 期限结构阈值
    term_slope_threshold: float = 0.0  # <0 = 倒挂 (backwardation)

    # Skew阈值
    skew_zscore_threshold: float = 1.5  # >1.5 = 极端陡峭

    # IV Crush预警
    iv_zscore_entry: float = 1.5    # IV zscore > 1.5 = 极端高位
    iv_zscore_exit: float = 1.0     # IV zscore < 1.0 = 回落正常
    iv_mom_crush_threshold: float = -0.05  # IV 5日动量 < -5%


@dataclass
class RegimeResult:
    """状态识别结果"""
    regime: Regime
    iv_percentile: float
    iv_momentum_5d: float
    iv_momentum_20d: float
    vix_percentile: float
    term_structure_slope: float
    skew_zscore: float
    iv_crush_detected: bool
    cooldown_multiplier: float


class CooldownManager:
    """冷静期管理器

    连续IV crush后自动减仓：
    - 第1次：仓位50%，持续cooldown_days天
    - 第2次（冷静期内）：仓位25%
    - 第3次：仓位12.5%
    - 最低不低于min_position_pct（20%）
    """

    def __init__(self, cooldown_days: int = 10, min_position_pct: float = 0.20):
        self.cooldown_days = cooldown_days
        self.min_position_pct = min_position_pct
        self.crush_count: int = 0
        self.last_crush_date: Optional[pd.Timestamp] = None

    def on_iv_crush(self, date: pd.Timestamp) -> None:
        """每次检测到IV crush时调用"""
        if self.last_crush_date is None or \
           (date - self.last_crush_date).days > self.cooldown_days:
            self.crush_count = 1  # 新周期
        else:
            self.crush_count += 1  # 冷静期内累加
        self.last_crush_date = date
        logger.debug(
            f"IV crush detected at {date}, crush_count={self.crush_count}"
        )

    def get_position_multiplier(self, date: pd.Timestamp) -> float:
        """返回仓位乘数：0.5^n，但不低于min_position_pct"""
        if self.last_crush_date is None:
            return 1.0
        days_since = (date - self.last_crush_date).days
        if days_since > self.cooldown_days:
            return 1.0  # 冷静期结束
        multiplier = 0.5 ** self.crush_count
        return max(self.min_position_pct, multiplier)

    def reset(self) -> None:
        """重置冷静期"""
        self.crush_count = 0
        self.last_crush_date = None


class RegimeDetector:
    """市场状态识别器

    输入：IV面板、VIX序列、期权链数据
    输出：5种状态 + IV crush预警 + 冷静期乘数
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.cooldown = CooldownManager()

    def compute_regime(
        self,
        product: str,
        date: pd.Timestamp,
        iv_panel: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        option_data: Optional[pd.DataFrame] = None,
    ) -> RegimeResult:
        """计算市场状态"""
        cfg = self.config

        # 1. IV百分位
        iv_pct = self._get_iv_percentile(product, date, iv_panel)

        # 2. IV动量
        iv_mom_5d = self._get_iv_momentum(product, date, iv_panel, window=5)
        iv_mom_20d = self._get_iv_momentum(product, date, iv_panel, window=20)

        # 3. VIX百分位
        vix_pct = self._get_vix_percentile(date, vix_series)

        # 4. 期限结构斜率
        term_slope = self._get_term_structure_slope(product, date, option_data)

        # 5. Skew Z-Score
        skew_z = self._get_skew_zscore(product, date, option_data)

        # 6. IV Crush检测
        iv_crush = self._detect_iv_crush(product, date, iv_panel)
        if iv_crush:
            self.cooldown.on_iv_crush(date)

        # 7. 冷静期乘数
        cooldown_mult = self.cooldown.get_position_multiplier(date)

        # 8. 状态判断
        regime = self._classify_regime(
            iv_pct, iv_mom_5d, iv_mom_20d, vix_pct, term_slope, skew_z
        )

        return RegimeResult(
            regime=regime,
            iv_percentile=iv_pct,
            iv_momentum_5d=iv_mom_5d,
            iv_momentum_20d=iv_mom_20d,
            vix_percentile=vix_pct,
            term_structure_slope=term_slope,
            skew_zscore=skew_z,
            iv_crush_detected=iv_crush,
            cooldown_multiplier=cooldown_mult,
        )

    def _classify_regime(
        self,
        iv_pct: float,
        iv_mom_5d: float,
        iv_mom_20d: float,
        vix_pct: float,
        term_slope: float,
        skew_z: float,
    ) -> Regime:
        """分类市场状态"""
        cfg = self.config

        # VOL_SPIKE检测（优先级最高）
        # 放宽条件：IV快速上涨 + (期限倒挂 OR skew变陡 OR IV高位)
        iv_rising_fast = abs(iv_mom_5d) > cfg.iv_mom_spike
        has_vol_signal = (term_slope < cfg.term_slope_threshold
                          or skew_z > cfg.skew_zscore_threshold
                          or iv_pct > 0.70)
        if iv_rising_fast and has_vol_signal:
            return Regime.VOL_SPIKE

        # 其他状态
        if iv_pct < cfg.iv_pct_calm and abs(iv_mom_20d) < cfg.iv_mom平稳:
            return Regime.CALM
        elif iv_pct < 0.50 and iv_mom_20d > cfg.iv_mom上升:
            return Regime.TRANSITION
        elif iv_pct < cfg.iv_pct_high:
            return Regime.HIGH_VOL
        else:  # iv_pct >= 0.75
            if iv_mom_5d < 0:  # IV从高位回落
                return Regime.COMPRESSION
            else:
                return Regime.HIGH_VOL

    def _detect_iv_crush(
        self,
        product: str,
        date: pd.Timestamp,
        iv_panel: pd.DataFrame,
    ) -> bool:
        """检测IV crush事件

        触发条件（两者结合）：
        1. IV zscore从>1.5降到<1.0
        2. IV 5日动量 < -5%
        """
        cfg = self.config

        iv_zscore_now = self._get_iv_zscore(product, date, iv_panel)
        iv_zscore_prev = self._get_iv_zscore_prev(product, date, iv_panel)
        iv_mom_5d = self._get_iv_momentum(product, date, iv_panel, window=5)

        # 条件1：IV zscore从极端高位回落
        zscore_drop = (iv_zscore_prev > cfg.iv_zscore_entry
                       and iv_zscore_now < cfg.iv_zscore_exit)

        # 条件2：IV 5日动量为负且跌幅>5%
        mom_negative = iv_mom_5d < cfg.iv_mom_crush_threshold

        return zscore_drop and mom_negative

    def _get_iv_percentile(
        self, product: str, date: pd.Timestamp, iv_panel: pd.DataFrame
    ) -> float:
        """获取IV百分位（252日滚动）"""
        if product not in iv_panel.columns:
            return 0.5
        series = iv_panel[product].dropna()
        if series.empty:
            return 0.5
        # 找到date之前的数据
        valid = series.loc[:date]
        if len(valid) < 60:
            return 0.5
        current = valid.iloc[-1]
        rank = (valid < current).sum() / len(valid)
        return rank

    def _get_iv_momentum(
        self, product: str, date: pd.Timestamp,
        iv_panel: pd.DataFrame, window: int = 5
    ) -> float:
        """获取IV动量（变化率）"""
        if product not in iv_panel.columns:
            return 0.0
        series = iv_panel[product].dropna()
        valid = series.loc[:date]
        if len(valid) < window + 1:
            return 0.0
        iv_now = valid.iloc[-1]
        iv_prev = valid.iloc[-window - 1]
        if iv_prev == 0:
            return 0.0
        return (iv_now - iv_prev) / iv_prev

    def _get_iv_zscore(
        self, product: str, date: pd.Timestamp, iv_panel: pd.DataFrame
    ) -> float:
        """获取IV Z-Score"""
        if product not in iv_panel.columns:
            return 0.0
        series = iv_panel[product].dropna()
        valid = series.loc[:date]
        if len(valid) < 60:
            return 0.0
        window = valid.iloc[-60:]
        mean = window.mean()
        std = window.std()
        if std == 0:
            return 0.0
        return (valid.iloc[-1] - mean) / std

    def _get_iv_zscore_prev(
        self, product: str, date: pd.Timestamp, iv_panel: pd.DataFrame
    ) -> float:
        """获取前一日IV Z-Score"""
        if product not in iv_panel.columns:
            return 0.0
        series = iv_panel[product].dropna()
        valid = series.loc[:date]
        if len(valid) < 61:
            return 0.0
        window = valid.iloc[-61:-1]
        mean = window.mean()
        std = window.std()
        if std == 0:
            return 0.0
        return (valid.iloc[-2] - mean) / std

    def _get_vix_percentile(
        self, date: pd.Timestamp, vix_series: Optional[pd.Series]
    ) -> float:
        """获取VIX百分位"""
        if vix_series is None or vix_series.empty:
            return 0.5
        valid = vix_series.loc[:date]
        if len(valid) < 60:
            return 0.5
        current = valid.iloc[-1]
        rank = (valid < current).sum() / len(valid)
        return rank

    def _get_term_structure_slope(
        self, product: str, date: pd.Timestamp,
        option_data: Optional[pd.DataFrame]
    ) -> float:
        """获取IV期限结构斜率

        slope = (iv_far - iv_near) / iv_near
        > 0 = contango (正常)
        < 0 = backwardation (倒挂，近月IV > 远月IV)
        """
        if option_data is None or option_data.empty:
            return 0.0

        day_data = option_data[
            (option_data["product"] == product)
            & (option_data["trade_date"] == date)
        ]
        if day_data.empty:
            return 0.0

        # 按到期日分组，取ATM IV
        if "days_to_expiry" not in day_data.columns:
            return 0.0

        expiries = day_data["days_to_expiry"].unique()
        if len(expiries) < 2:
            return 0.0

        expiries_sorted = sorted(expiries)
        near_dte = expiries_sorted[0]
        far_dte = expiries_sorted[-1]

        near_data = day_data[day_data["days_to_expiry"] == near_dte]
        far_data = day_data[day_data["days_to_expiry"] == far_dte]

        if near_data.empty or far_data.empty:
            return 0.0

        iv_near = near_data["bs_iv"].median()
        iv_far = far_data["bs_iv"].median()

        if iv_near == 0:
            return 0.0

        return (iv_far - iv_near) / iv_near

    def _get_skew_zscore(
        self, product: str, date: pd.Timestamp,
        option_data: Optional[pd.DataFrame]
    ) -> float:
        """获取Skew Z-Score

        使用OTM put IV - ATM IV作为skew度量
        """
        if option_data is None or option_data.empty:
            return 0.0

        day_data = option_data[
            (option_data["product"] == product)
            & (option_data["trade_date"] == date)
        ]
        if day_data.empty or "bs_iv" not in day_data.columns:
            return 0.0

        # 简化：用delta分位计算skew
        if "Delta" not in day_data.columns:
            return 0.0

        # OTM put: delta < -0.15
        otm_puts = day_data[day_data["Delta"] < -0.15]
        # ATM: |delta| < 0.1
        atm = day_data[day_data["Delta"].abs() < 0.1]

        if otm_puts.empty or atm.empty:
            return 0.0

        put_iv = otm_puts["bs_iv"].median()
        atm_iv = atm["bs_iv"].median()

        if atm_iv == 0:
            return 0.0

        skew = (put_iv - atm_iv) / atm_iv

        # 简单zscore：与历史比较（这里用固定阈值近似）
        # 实际应该用滚动窗口，但为了简化先用阈值
        if skew > 0.15:
            return 2.0
        elif skew > 0.10:
            return 1.5
        elif skew > 0.05:
            return 1.0
        elif skew > 0:
            return 0.5
        else:
            return 0.0

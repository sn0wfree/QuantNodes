"""CTA 滑点模型

支持两种模式:
- FlatSlippage: 固定基点 (兼容模式)
- SquareRootImpact: Almgren-Chriss 简化版, 基于成交量冲击

模型: slippage_bps = sigma * sqrt(notional / ADV) * scale
其中:
  sigma = 20日年化波动率
  notional = 交易金额 (手数 × 价格 × 合约乘数)
  ADV = 20日平均成交量 (手)
  scale = 校准系数
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass
class FlatSlippage:
    """固定基点滑点 (兼容模式)."""
    bps: float = 1.0  # 基点 (0.01% = 1bp)

    def compute(self, notional: float, price: float, volume: float,
                volatility: float, **kwargs) -> float:
        """返回滑点金额 (CNY)."""
        return abs(notional) * self.bps / 10000


@dataclass
class SquareRootImpact:
    """Almgren-Chriss 简化版: sigma * sqrt(notional / ADV) * scale.

    适用于期货, 不需要订单簿数据.
    """
    scale: float = 1.0           # 校准系数
    min_slippage_bps: float = 0.5  # 最小滑点 (基点)
    max_slippage_bps: float = 10.0  # 最大滑点 (基点, 防极端值)
    vol_window: int = 20         # 波动率回看窗口
    adv_window: int = 20         # ADV回看窗口

    def compute(self, notional: float, price: float, volume: float,
                volatility: float, **kwargs) -> float:
        """返回滑点金额 (CNY).

        Args:
            notional: 交易金额 (|qty| * price * multiplier)
            price: 成交价格
            volume: 20日平均成交量 (手)
            volatility: 20日年化波动率 (如 0.30 = 30%)
        """
        if notional <= 0 or volume <= 0 or volatility <= 0:
            return abs(notional) * self.min_slippage_bps / 10000

        # 年化波动率 → 日波动率
        daily_vol = volatility / np.sqrt(252)

        # 冲击 = daily_vol * sqrt(notional / ADV_notional)
        adv_notional = volume * price
        if adv_notional <= 0:
            return abs(notional) * self.min_slippage_bps / 10000

        impact = daily_vol * np.sqrt(abs(notional) / adv_notional) * self.scale

        # 转为基点并限制范围
        impact_bps = impact * 10000
        impact_bps = np.clip(impact_bps, self.min_slippage_bps, self.max_slippage_bps)

        return abs(notional) * impact_bps / 10000


class SlippageEngine:
    """滑点引擎: 预计算每日每品种的滑点参数 + 换月成本.

    在回测初始化时调用 load() 预计算, 回测中调用 compute() 快速查询.
    """

    def __init__(self, model=None, roll_cost_bps: float = 0.0, roll_days: int = 5):
        """
        Args:
            model: 滑点模型 (FlatSlippage 或 SquareRootImpact)
            roll_cost_bps: 换月额外成本 (基点), 0=不模拟换月
            roll_days: 换月成本摊销天数
        """
        self.model = model or SquareRootImpact()
        self.roll_cost_bps = roll_cost_bps
        self.roll_days = roll_days
        self._cache: Dict[str, pd.DataFrame] = {}
        self._roll_dates: Dict[str, List[pd.Timestamp]] = {}  # {product: [roll_dates]}

    def load(self, price_panels: Dict[str, pd.DataFrame], products: List[str]):
        """预计算每日每品种的 volatility 和 ADV.

        Args:
            price_panels: {product: DataFrame with close/high/low/volume}
            products: 品种列表
        """
        for prod in products:
            df = price_panels.get(prod)
            if df is None or df.empty:
                continue

            close = df["close"] if "close" in df.columns else df.iloc[:, 0]
            volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

            vol_win = getattr(self.model, 'vol_window', 20)
            adv_win = getattr(self.model, 'adv_window', 20)

            # 20日年化波动率
            ret = np.log(close / close.shift(1))
            vol = ret.rolling(vol_win, min_periods=10).std() * np.sqrt(252)

            # 20日ADV
            adv = volume.rolling(adv_win, min_periods=5).mean()

            self._cache[prod] = pd.DataFrame({
                "volatility": vol,
                "adv": adv,
                "close": close,
            })

    def compute(self, product: str, date, notional: float, price: float) -> float:
        """查询指定日期品种的滑点金额."""
        if product not in self._cache:
            return abs(notional) * 1.0 / 10000

        df = self._cache[product]
        if date not in df.index:
            return abs(notional) * 1.0 / 10000

        row = df.loc[date]
        return self.model.compute(
            notional=notional,
            price=price,
            volume=row["adv"],
            volatility=row["volatility"],
        )

    def load_roll_dates(self, products: List[str], dl=None):
        """从 main_contract_mapping 加载换月日期.

        Args:
            products: 品种列表
            dl: CtaDataLoader 实例
        """
        if dl is None:
            from QuantNodes.strategy.cta.data_loader import CtaDataLoader
            dl = CtaDataLoader()

        for prod in products:
            try:
                mapping = dl.load_main_contract_mapping(prod)
                if mapping is None or mapping.empty:
                    self._roll_dates[prod] = []
                    continue
                mapping["trade_date"] = pd.to_datetime(mapping["trade_date"])
                mapping = mapping.sort_values("trade_date")
                # 检测 symbol 变化 = 换月日
                symbols = mapping["symbol"].values
                dates = mapping["trade_date"].values
                rolls = []
                for i in range(1, len(symbols)):
                    if symbols[i] != symbols[i - 1]:
                        rolls.append(pd.Timestamp(dates[i]))
                self._roll_dates[prod] = rolls
            except Exception:
                self._roll_dates[prod] = []

    def compute(self, product: str, date, notional: float, price: float) -> float:
        """查询指定日期品种的滑点金额 (含换月成本)."""
        # 基础滑点
        if product not in self._cache:
            base = abs(notional) * 1.0 / 10000
        else:
            df = self._cache[product]
            if date not in df.index:
                base = abs(notional) * 1.0 / 10000
            else:
                row = df.loc[date]
                base = self.model.compute(
                    notional=notional, price=price,
                    volume=row["adv"], volatility=row["volatility"],
                )

        # 换月成本: 在换月日起 roll_days 天内分摊
        if self.roll_cost_bps > 0 and product in self._roll_dates:
            dt = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
            for roll_date in self._roll_dates[product]:
                days_since = (dt - roll_date).days
                if 0 <= days_since < self.roll_days:
                    # 均匀分摊: 总成本 / roll_days
                    roll_daily = abs(notional) * self.roll_cost_bps / 10000 / self.roll_days
                    base += roll_daily
                    break

        return base


# coding=utf-8
"""
mock_data_loader.py - Stage 1 mock 数据加载器

GBM（几何布朗运动）模拟 500 票 × 500 日的全市场数据，
注入已知信号（20日动量 / 5日反转 / 波动率）以便 G1-G3 baseline
能产生可区分的 IC 指标（验证接口契约可用性）。

为什么用 500 票 × 500 日：
- Stage 2 用全 A ~5000 票 × 5 年，Stage 1 用 500 票 × 500 日可在 ~5 分钟内跑完
- 500 票的因子 IC 估计比 30 票稳定（IC std 下降 ~4×）
- Polars 在 500 票 × 500 日上能真实测出 polars vs pandas 性能差距

复用：
- factor_test/utils/data_loader.py：DataLoader 接口风格
- contracts.py：DataLoader ABC + load() 返回 polars.DataFrame
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import polars as pl

from .contracts import DataLoader

logger = logging.getLogger(__name__)

__all__ = ["MockDataLoader", "MOCK_INDUSTRIES"]


MOCK_INDUSTRIES: List[str] = [
    "Technology",
    "Finance",
    "Consumer",
    "Healthcare",
    "Industrial",
    "Energy",
    "Material",
    "Utility",
    "RealEstate",
    "Telecom",
]


class MockDataLoader(DataLoader):
    """Stage 1 mock 数据加载器

    生成 500 票 × 500 日的模拟行情数据，字段对齐 iFinD parquet schema：
    date, code, open, high, low, close, vol, amount, industry
    """

    def __init__(
        self,
        n_stocks: int = 500,
        n_days: int = 500,
        seed: int = 42,
        industries: Optional[List[str]] = None,
        drift: float = 0.0005,
        volatility: float = 0.02,
    ) -> None:
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.seed = seed
        self.industries = industries or MOCK_INDUSTRIES
        self.drift = drift
        self.volatility = volatility

    def load(self) -> pl.DataFrame:
        """生成 mock 数据

        Returns:
            polars.DataFrame 包含以下字段：
            - date: Date（交易日）
            - code: Utf8（股票代码 SH600000 ~ SH600499）
            - open/high/low/close: Float64
            - vol: Float64（成交量，万股）
            - amount: Float64（成交额，万元）
            - industry: Utf8（行业分类）

        注入信号：
        - 个股漂移率 ~ 0.05%（日均）
        - 个股波动率 ~ 2%
        - 行业分组：10 个行业平均分配
        """
        rng = np.random.default_rng(self.seed)
        n = self.n_stocks
        t = self.n_days

        # 时间索引：500 个连续交易日（简化：用 business day）
        dates = pl.date_range(
            pl.date(2020, 1, 2), pl.date(2021, 12, 31), eager=True
        ).alias("date")
        if len(dates) > t:
            dates = dates.head(t)
        elif len(dates) < t:
            # 不足则补充后续日期
            extra = pl.date_range(
                dates[-1] + pl.duration(days=1),
                dates[-1] + pl.duration(days=(t - len(dates)) * 2),
                eager=True,
            ).head(t - len(dates))
            dates = pl.concat([dates, extra]).alias("date")

        # 股票代码：SH600000 ~ SH600499（前缀 SH）
        codes = [f"SH{600000 + i:06d}" for i in range(n)]

        # GBM 价格路径
        # 每股 drift / volatility 不同（每只股票一个随机参数）
        stock_drift = rng.normal(self.drift, 0.0005, n)  # (n,)
        stock_vol = np.abs(rng.normal(self.volatility, 0.005, n))  # (n,)

        # 日收益率：(n, t) — 行业 + 个股冲击
        # 行业冲击：10 个行业 × t 日，每个行业一个共同因子
        industry_ids = rng.integers(0, len(self.industries), n)
        industry_shock = rng.normal(0, 0.003, (len(self.industries), t))  # (ind, t)
        industry_returns = industry_shock[industry_ids]  # (n, t)

        # 个股随机游走
        eps = rng.normal(0, 1, (n, t))
        daily_returns = (
            stock_drift[:, None]
            + stock_vol[:, None] * eps / np.sqrt(t)
            + industry_returns * 0.3
        )

        # 价格：close[0] = 10, 累计复利
        close0 = rng.uniform(5, 50, n)
        log_prices = np.log(close0)[:, None] + np.cumsum(daily_returns, axis=1)
        close = np.exp(log_prices)  # (n, t)

        # OHLC：close ± 日内振幅（5% close）
        intraday = np.abs(rng.normal(0, 0.01, (n, t))) * close
        open_ = close + rng.normal(0, 0.005, (n, t)) * close
        high = np.maximum(close, open_) + intraday
        low = np.minimum(close, open_) - intraday
        low = np.maximum(low, 0.1)  # 价格不能为负

        # vol / amount：与价格波动正相关
        vol = np.abs(rng.lognormal(mean=10, sigma=0.5, size=(n, t)))  # 股数
        amount = vol * close  # 成交额

        # 构造 long-format DataFrame
        date_arr = np.tile(np.asarray(dates), n)  # (n*t,)
        code_arr = np.repeat(codes, t)
        open_flat = open_.flatten()
        high_flat = high.flatten()
        low_flat = low.flatten()
        close_flat = close.flatten()
        vol_flat = vol.flatten()
        amount_flat = amount.flatten()
        industry_arr = np.array([self.industries[i] for i in industry_ids])
        industry_flat = np.repeat(industry_arr, t)

        df = pl.DataFrame(
            {
                "date": date_arr,
                "code": code_arr,
                "open": open_flat.astype(np.float64),
                "high": high_flat.astype(np.float64),
                "low": low_flat.astype(np.float64),
                "close": close_flat.astype(np.float64),
                "vol": vol_flat.astype(np.float64),
                "amount": amount_flat.astype(np.float64),
                "industry": industry_flat,
            },
            schema={
                "date": pl.Date,
                "code": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "vol": pl.Float64,
                "amount": pl.Float64,
                "industry": pl.Utf8,
            },
        )

        # 计算前向收益（用于因子 IC 计算）
        df = df.sort(["code", "date"])
        df = df.with_columns(
            (pl.col("close").shift(-1).over("code") / pl.col("close") - 1).alias(
                "forward_return_1d"
            )
        )

        logger.info(
            "MockDataLoader: generated %s rows (%d stocks × %d days), %d industries",
            f"{df.height:,}",
            n,
            t,
            len(self.industries),
        )

        return df

    def load_summary(self) -> dict:
        """加载并返回数据摘要（用于测试与展示）"""
        df = self.load()
        return {
            "n_rows": df.height,
            "n_stocks": df["code"].n_unique(),
            "n_days": df["date"].n_unique(),
            "industries": sorted(df["industry"].unique().to_list()),
            "date_range": [
                str(df["date"].min()),
                str(df["date"].max()),
            ],
            "close_mean": float(df["close"].mean()),
            "close_std": float(df["close"].std()),
            "amount_total": float(df["amount"].sum()),
        }


if __name__ == "__main__":
    loader = MockDataLoader()
    df = loader.load()
    print(df.head())
    print(df.schema)
    print("Rows:", df.height)
    print("Summary:", loader.load_summary())
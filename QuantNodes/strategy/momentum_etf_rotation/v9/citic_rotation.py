# coding=utf-8
"""v9 中信行业轮动 — 动量 + 质量打分, Top-K 行业 ETF 选优.

参考: 中信证券《行业轮动系列》

方法学:
  - 在 43 ETF 的「行业类别」内做轮动
  - 因子: 动量 (12-1 月) + 质量 (反向波动率)
  - 每周计算综合得分 → Top-K 选优
  - 行业 ETF 内: Top-K 高配 (5x 权重), 其他低配 (1x)
  - 非行业 ETF (broad/overseas/gold) 保留等权底仓

权重构造:
  1. 基础权重: 1/N 等权
  2. 行业内动量打分 → Top-K
  3. Top-K 行业: 5 * (1/N)
  4. 其他行业: 1 * (1/N)
  5. 归一化
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v9.citic_all_weather import (
    classify_etf,
)


def compute_industry_score(returns_df, lookback_mom=52, skip_mom=4,
                            lookback_vol=26):
    """行业打分: 动量 - 波动率.

    返回: DataFrame, index=returns_df.index, columns=returns_df.columns
    """
    cumulative = (1 + returns_df).rolling(lookback_mom).apply(np.prod, raw=True) - 1
    if skip_mom > 0:
        cumulative = cumulative.shift(skip_mom)
    vol = returns_df.rolling(lookback_vol).std()

    mean = cumulative.mean(axis=1)
    std = cumulative.std(axis=1) + 1e-10
    z_mom = cumulative.sub(mean, axis=0).div(std, axis=0)

    mean = vol.mean(axis=1)
    std = vol.std(axis=1) + 1e-10
    z_vol = vol.sub(mean, axis=0).div(std, axis=0)

    score = z_mom - z_vol
    return score


def build_rotation_weights(returns_df, top_k=5, lookback_mom=52, skip_mom=4,
                            lookback_vol=26, sector_mult=4.0, sector_floor_mult=0.5,
                            cost_bps=5.0, floor=0.005, cap=0.20):
    """行业轮动权重.

    参数:
        top_k: 行业类别中, 选几个最优
        sector_mult: Top-K 行业 ETF 权重倍数
        sector_floor_mult: 非 Top-K 行业 ETF 权重倍数
    """
    etf_cls = classify_etf(returns_df.columns.tolist())
    sector_codes = [c for c, cat in etf_cls.items() if cat == 'sector']
    non_sector_codes = [c for c, cat in etf_cls.items() if cat != 'sector']

    score = compute_industry_score(
        returns_df, lookback_mom, skip_mom, lookback_vol,
    )

    codes = returns_df.columns.tolist()
    n = len(codes)
    weights = pd.DataFrame(0.0, index=returns_df.index, columns=codes)

    base_w = 1.0 / n

    for date in score.index:
        if score.loc[date].isna().all():
            continue

        sector_scores = score.loc[date, sector_codes].dropna()
        if len(sector_scores) < top_k:
            continue

        top_sectors = sector_scores.sort_values(ascending=False).head(top_k).index.tolist()
        bot_sectors = [c for c in sector_codes if c not in top_sectors]

        for code in top_sectors:
            weights.loc[date, code] = base_w * sector_mult

        for code in bot_sectors:
            weights.loc[date, code] = base_w * sector_floor_mult

        for code in non_sector_codes:
            weights.loc[date, code] = base_w

    weights = weights.clip(lower=floor, upper=cap)
    weights = weights.div(weights.sum(axis=1).replace(0, 1), axis=0).fillna(0)

    return weights, score


if __name__ == '__main__':
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(REPO))

    macro = pd.read_parquet(REPO / 'data' / 'high_freq_macro' / 'v7_6_X_macro_weekly.parquet')
    etf = pd.read_parquet(REPO / 'data' / 'high_freq_macro' / 'v7_10_Y_weekly.parquet')
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]

    weights, score = build_rotation_weights(etf_clean, top_k=5)
    print(f"权重时序: {weights.shape}")
    print(f"权重总和: {weights.sum(axis=1).describe()[['min', 'max']]}")
    print(f"\nTop-K 行业被选次数 (从 score > 0 + top 5 入选):")
    etf_cls = classify_etf(etf_clean.columns.tolist())
    sector_codes = [c for c, cat in etf_cls.items() if cat == 'sector']
    print(weights[sector_codes].mean().sort_values(ascending=False).head(10))
    print(f"\n非行业类别平均权重:")
    for cat in ['broad', 'overseas', 'gold']:
        codes = [c for c, k in etf_cls.items() if k == cat]
        if codes:
            print(f"  {cat}: {weights[codes].mean().mean():.4f}")

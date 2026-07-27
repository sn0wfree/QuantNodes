# coding=utf-8
"""v9 中信多因子选股 — 5 因子横截面打分加权.

参考: 中信证券《多因子选股系列》

方法学: 类似 BARRA 的多因子打分, 但 ETF 没有市值/PB/PE 数据,
       用以下代理因子 (基于周收益):

  1. **Momentum (动量)**: 过去 12 月收益 (skip 1 月) → 正向
  2. **Volatility (波动率, 反向)**: 26 周 std → 越小越好 (低波特质)
  3. **Quality (质量)**: 26 周 Sharpe = 收益/波动率 → 正向
  4. **Size (规模, 反向)**: 4 周均振幅 (proxy) → 越小越好 (大盘特质)
  5. **Value (价值)**: 长期反转 (52-104 周收益, 取负) → 越负越好 (反转)

Composite score = z(Mom) - z(Vol) + z(Qual) - z(Size) + z(Value)

权重构造:
  - 横截面 z-score 标准化 (每周末)
  - 复合得分 → 排序 → Top-K 候选
  - 候选池内按得分加权 (softmax 软化)
  - Top-K = 10 (高 alpha 暴露)
  - 剩余 33 个 ETF 等权 (市场暴露)

注: 严格 BARRA 模型需要市值/PB/PE/ROE 等基本面数据,
   本策略用收益类代理因子, 实质是「风格横截面打分」, 非严格 BARRA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_factor(returns_df, lookback=52, skip=4):
    """动量因子: 过去 lookback 周收益, 跳过最近 skip 周."""
    cumulative = (1 + returns_df).rolling(lookback).apply(np.prod, raw=True) - 1
    if skip > 0:
        cumulative = cumulative.shift(skip)
    return cumulative


def compute_volatility_factor(returns_df, lookback=26):
    """波动率因子 (反向): 滚动 lookback 周标准差."""
    return returns_df.rolling(lookback).std()


def compute_quality_factor(returns_df, lookback=26):
    """质量因子: 滚动 lookback 周 Sharpe."""
    mean_ret = returns_df.rolling(lookback).mean()
    std_ret = returns_df.rolling(lookback).std()
    sharpe = mean_ret / (std_ret + 1e-10)
    return sharpe


def compute_size_factor(returns_df, lookback=4):
    """规模因子 (反向): 4 周均振幅 (proxy for 流通市值)."""
    amplitude = returns_df.abs().rolling(lookback).mean()
    return amplitude


def compute_value_factor(returns_df, lookback_long=104, lookback_short=52):
    """价值因子: 长期反转 (52-104 周累计收益, 取负)."""
    long_cum = (1 + returns_df).rolling(lookback_long).apply(np.prod, raw=True) - 1
    return -long_cum


def cross_section_zscore(factor_df):
    """横截面 z-score 标准化 (按行)."""
    mean = factor_df.mean(axis=1)
    std = factor_df.std(axis=1) + 1e-10
    return factor_df.sub(mean, axis=0).div(std, axis=0)


def compute_multifactor_scores(returns_df, lookback_mom=52, skip_mom=4,
                                lookback_vol=26, lookback_qual=26,
                                lookback_size=4, lookback_value=104):
    """5 因子 → 复合得分.

    返回: DataFrame, index=returns_df.index, columns=returns_df.columns
    """
    mom = compute_momentum_factor(returns_df, lookback_mom, skip_mom)
    vol = compute_volatility_factor(returns_df, lookback_vol)
    qual = compute_quality_factor(returns_df, lookback_qual)
    size = compute_size_factor(returns_df, lookback_size)
    value = compute_value_factor(returns_df, lookback_value, lookback_mom)

    z_mom = cross_section_zscore(mom)
    z_vol = cross_section_zscore(vol)
    z_qual = cross_section_zscore(qual)
    z_size = cross_section_zscore(size)
    z_value = cross_section_zscore(value)

    score = z_mom - z_vol + z_qual - z_size + z_value
    return score


def build_multifactor_weights(returns_df, top_k=10, lookback_mom=52, skip_mom=4,
                              lookback_vol=26, lookback_qual=26,
                              lookback_size=4, lookback_value=104,
                              cost_bps=5.0, floor=0.005, cap=0.15):
    """多因子加权.

    策略:
      - 计算复合得分
      - 选 Top-K 候选
      - 候选池内按 softmax(scores * temperature) 加权
      - 剩余等权 (1 - 候选权重) / N_剩余
    """
    score = compute_multifactor_scores(
        returns_df, lookback_mom, skip_mom,
        lookback_vol, lookback_qual, lookback_size, lookback_value,
    )

    temperature = 1.0

    codes = returns_df.columns.tolist()
    n = len(codes)
    weights = pd.DataFrame(0.0, index=returns_df.index, columns=codes)

    for date in score.index:
        if score.loc[date].isna().all():
            continue
        scores = score.loc[date].dropna()
        if len(scores) < top_k:
            continue
        ranked = scores.sort_values(ascending=False)
        topk_codes = ranked.head(top_k).index.tolist()

        topk_scores = ranked.head(top_k).values
        softmax_w = np.exp(topk_scores * temperature)
        softmax_w = softmax_w / softmax_w.sum()

        candidate_total = 0.5
        for code, w in zip(topk_codes, softmax_w):
            weights.loc[date, code] = w * candidate_total

        rest_codes = [c for c in codes if c not in topk_codes]
        if rest_codes:
            rest_w = (1 - candidate_total) / len(rest_codes)
            for code in rest_codes:
                weights.loc[date, code] = rest_w

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

    weights, score = build_multifactor_weights(etf_clean, top_k=10)
    print(f"权重时序: {weights.shape}")
    print(f"权重总和: {weights.sum(axis=1).describe()[['min', 'max']]}")
    print("\nTop-10 候选 (按被选为 Top-K 的次数):")
    topk_count = (weights > 0.05).sum(axis=0).sort_values(ascending=False).head(10)
    print(topk_count)
    print("\nScore 描述:")
    print(score.describe().T[['mean', 'std', 'min', 'max']].head(10))

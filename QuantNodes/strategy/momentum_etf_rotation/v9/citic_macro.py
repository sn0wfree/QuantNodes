# coding=utf-8
"""v9 中信大类资产配置 — 5 宏观因子战术择时.

参考: 中信证券《大类资产配置研究》系列报告

方法学:
  5 个宏观因子 (周频):
    1. 增长: 宏观增长因子
    2. 通胀: 宏观通胀因子_生活端
    3. 流动性/信贷: 信用利差因子 (反向, 越低越宽松)
    4. 汇率: 宏观汇率因子 (反向, 升值利好股)
    5. 利率: 期限利差因子_债 (反向, 走阔利好股)

  每个因子计算 52 周滚动 z-score:
    z > 0: 利好风险资产
    z < 0: 利空风险资产
    z ≈ 0: 中性

  5 因子等权合成综合得分:
    macro_score = (z_growth - z_credit + z_fx + z_rate_yield) / 4
    (通胀因子权重为 0, 因为它对股/债影响方向相反, 不确定性强)

  综合得分 → 战术权重:
    macro_score > 0.5:  股票 80% + 防御 20%
    macro_score < -0.5: 股票 30% + 防御 70%
    中间: 线性插值

  资产分类:
    风险资产: 宽基 ETF (broad)
    防御资产: 黄金 ETF (gold) + 防御性宽基

  基础权重: 风险平价 × 战术权重
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_risk_parity_base,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_all_weather import (
    classify_etf,
)


MACRO_FACTOR_MAP = {
    '增长': '宏观增长因子',
    '通胀': '宏观通胀因子_生活端',
    '信贷': '信用利差因子',
    '汇率': '宏观汇率因子',
    '利率': '期限利差因子_债',
}

MACRO_SIGN = {
    '增长': +1,
    '通胀': 0,
    '信贷': -1,
    '汇率': -1,
    '利率': +1,
}


def compute_macro_score(macro_df, lookback=52):
    """5 宏观因子 → 综合得分.

    返回: Series, index=macro_df.index
    """
    parts = []
    weights = []
    for name, col in MACRO_FACTOR_MAP.items():
        if col not in macro_df.columns:
            continue
        s = macro_df[col]
        z = (s - s.rolling(lookback).mean()) / (s.rolling(lookback).std() + 1e-10)
        sign = MACRO_SIGN[name]
        if sign == 0:
            continue
        parts.append(sign * z)
        weights.append(1.0)

    if not parts:
        return pd.Series(0.0, index=macro_df.index)

    score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    score = score.clip(-2.0, 2.0)
    return score


def build_tactical_tilt(etf_cls, macro_score, score_threshold=0.5,
                        risk_min=0.30, risk_max=0.80, cap=2.0, floor=0.3):
    """根据 macro_score 构建战术权重 (风险资产占比).

    macro_score → risk_weight (线性插值):
        score >  threshold: risk_weight = risk_max
        score < -threshold: risk_weight = risk_min
        中间: 线性插值

    返回: DataFrame, index=macro_score.index, columns=etf_codes
    """
    codes = list(etf_cls.keys())

    risk_w = pd.Series(
        np.where(
            macro_score > score_threshold, risk_max,
            np.where(macro_score < -score_threshold, risk_min,
                     risk_min + (risk_max - risk_min) * (macro_score + score_threshold) / (2 * score_threshold))
        ),
        index=macro_score.index,
    )
    risk_w = risk_w.clip(risk_min, risk_max)
    defensive_w = 1.0 - risk_w

    tilt = pd.DataFrame(0.0, index=macro_score.index, columns=codes)
    for code in codes:
        cat = etf_cls[code]
        if cat in ('broad', 'sector'):
            tilt[code] = risk_w
        elif cat == 'overseas':
            tilt[code] = risk_w * 0.5
        elif cat == 'gold':
            tilt[code] = defensive_w * 2.0

    return tilt.clip(floor, cap)


def run_macro_allocation(returns_df, macro_df, lookback_rp=52, lookback_score=52,
                         cost_bps=5.0, floor=0.005, cap=0.15):
    etf_cls = classify_etf(returns_df.columns.tolist())

    macro_score = compute_macro_score(macro_df, lookback=lookback_score)
    tactical_tilt = build_tactical_tilt(etf_cls, macro_score)

    rp_w = compute_risk_parity_base(returns_df, lookback=lookback_rp)
    rp_w = rp_w.reindex(returns_df.index, method='ffill').fillna(0)

    tactical_tilt = tactical_tilt.reindex(rp_w.index, method='ffill').fillna(1.0)

    raw = rp_w * tactical_tilt

    raw = raw.div(raw.sum(axis=1).replace(0, 1), axis=0).fillna(0)
    raw = raw.clip(lower=floor)

    excess = raw.sum(axis=1) - 1.0
    norm = raw.div(raw.sum(axis=1).replace(0, 1), axis=0).fillna(0)
    raw = raw.add((-excess).values.reshape(-1, 1) * norm.values, axis=0)
    raw = raw.clip(lower=0, upper=cap)

    weights = raw.div(raw.sum(axis=1).replace(0, 1), axis=0).fillna(0)

    meta = {
        'classification': etf_cls,
        'macro_score': macro_score,
        'tactical_tilt': tactical_tilt,
    }
    return weights, meta


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

    weights, meta = run_macro_allocation(etf_clean, macro)
    print(f"权重时序: {weights.shape}")
    print(f"权重总和: {weights.sum(axis=1).describe()[['min', 'max']]}")
    print(f"\nmacro_score 统计:")
    print(meta['macro_score'].describe())
    print(f"\n战术 tilt 范围 (broad 类别):")
    broad_codes = [c for c, cat in meta['classification'].items() if cat == 'broad']
    if broad_codes:
        print(meta['tactical_tilt'][broad_codes[0]].describe())
    print(f"\nTop-10 平均权重:")
    print(weights.mean(axis=0).sort_values(ascending=False).head(10))

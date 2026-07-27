# coding=utf-8
"""v9 中信里昂全天候 (All-Weather) — 风险平价 + 增长/通胀象限轮动.

参考:
  - Bridgewater All-Weather (Ray Dalio)
  - 中信里昂 中国版全天候策略

方法学:
  1. 基础: 风险平价 (Risk Parity) 权重
  2. 宏观象限: 增长 × 通胀 二维定位
     - ↑增长 ↓通胀: 加股票 / 信用债
     - ↑增长 ↑通胀: 平衡 / 适度加商品
     - ↓增长 ↑通胀: 防御 / 现金 + 短债 + 黄金
     - ↓增长 ↓通胀: 防御性股票 / 长债
  3. 资产分类: 43 ETF 分为 4 类
     - 宽基 (broad): 510300, 510500, 510050, 159915, 588000, 159901
     - 行业 (sector): 512xxx, 515xxx 等
     - 海外 (overseas): 513xxx, 510900, 159920, 159996
     - 商品 (gold): 518880, 518800, 159985
  4. 周频调仓, 风险预算 × 象限系数

公式:
  w_i^{final}(t) = w_i^{rp}(t) × Q_i(g_t, i_t)
  归一化: w_i^{final} = w_i^{final} / Σ w_j^{final}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_risk_parity_base,
)


CITIC_ETF_CLASSIFICATION = {
    'broad': [
        '510300', '510500', '510050', '159915', '588000', '159901',
    ],
    'sector': [
        '512760', '512480', '515030', '515790', '512690', '512170',
        '512010', '515050', '159928', '512880', '512000', '512800',
        '515220', '512200', '512400', '512660', '512980', '515880',
        '512120', '159741', '161226', '159981', '159766',
    ],
    'overseas': [
        '510900', '159920', '159996', '513010', '513050', '513100',
        '513300', '513500', '513520', '513880', '159941', '159740',
    ],
    'gold': [
        '518880', '518800', '159985',
    ],
}


QUADRANT_TILT = {
    ('up', 'down'): {
        'broad': 1.30, 'sector': 1.20, 'overseas': 1.10, 'gold': 0.60,
    },
    ('up', 'up'): {
        'broad': 1.00, 'sector': 1.15, 'overseas': 1.10, 'gold': 1.10,
    },
    ('down', 'up'): {
        'broad': 0.70, 'sector': 0.80, 'overseas': 0.90, 'gold': 1.50,
    },
    ('down', 'down'): {
        'broad': 0.85, 'sector': 0.80, 'overseas': 0.95, 'gold': 1.30,
    },
}


def classify_etf(etf_columns):
    """根据 ETF 代码分类.

    返回: dict {etf_code: 'broad'|'sector'|'overseas'|'gold'}
    """
    cls = {}
    for code in etf_columns:
        matched = False
        for cat, codes in CITIC_ETF_CLASSIFICATION.items():
            if code in codes:
                cls[code] = cat
                matched = True
                break
        if not matched:
            if code.startswith('518'):
                cls[code] = 'gold'
            elif code.startswith('513') or code.startswith('510900') or code.startswith('159920') or code.startswith('159996'):
                cls[code] = 'overseas'
            elif code.startswith('510') or code.startswith('159') or code.startswith('588'):
                cls[code] = 'broad'
            else:
                cls[code] = 'sector'
    return cls


def compute_macro_quadrant(macro_df, lookback=52):
    """计算当前宏观象限 (增长 × 通胀).

    使用 宏观增长因子 (增长 proxy) 和 宏观通胀因子_生活端 (通胀 proxy).
    滚动 z-score 标准化, > 0 为 up, < 0 为 down.

    返回: DataFrame, 列为 ['growth', 'inflation', 'quadrant']
    """
    growth = macro_df['宏观增长因子']
    inflation = macro_df['宏观通胀因子_生活端']

    growth_z = (growth - growth.rolling(lookback).mean()) / (growth.rolling(lookback).std() + 1e-10)
    inflation_z = (inflation - inflation.rolling(lookback).mean()) / (inflation.rolling(lookback).std() + 1e-10)

    g_state = pd.Series(np.where(growth_z > 0, 'up', 'down'), index=growth.index, name='growth')
    i_state = pd.Series(np.where(inflation_z > 0, 'up', 'down'), index=inflation.index, name='inflation')
    quadrant = g_state.astype(str) + '/' + i_state.astype(str)

    return pd.DataFrame({
        'growth_z': growth_z,
        'inflation_z': inflation_z,
        'growth': g_state,
        'inflation': i_state,
        'quadrant': quadrant,
    })


def build_quadrant_tilt_matrix(etf_cls, quadrant_series, cap=2.0, floor=0.4):
    """根据象限时序, 构建每个 ETF 的象限调整系数时序.

    参数:
        etf_cls: dict {etf_code: category}
        quadrant_series: Series of quadrant strings
        cap: 最大调整系数
        floor: 最小调整系数

    返回: DataFrame, index=quadrant_series.index, columns=etf_codes
    """
    codes = list(etf_cls.keys())
    tilt = pd.DataFrame(1.0, index=quadrant_series.index, columns=codes)

    for date, quad in quadrant_series.items():
        g_state, i_state = quad.split('/')
        factors = QUADRANT_TILT.get((g_state, i_state), None)
        if factors is None:
            continue
        for code in codes:
            cat = etf_cls[code]
            tilt.loc[date, code] = factors.get(cat, 1.0)

    return tilt.clip(floor, cap)


def run_all_weather(returns_df, macro_df, lookback_rp=52, lookback_macro=52,
                    cost_bps=5.0, floor=0.005, cap=0.15):
    """运行中信里昂全天候策略.

    参数:
        returns_df: ETF 周收益 DataFrame
        macro_df: 宏观因子 DataFrame
        lookback_rp: 风险平价滚动窗口
        lookback_macro: 宏观象限滚动窗口
        cost_bps: 单边交易成本 (bps)
        floor/cap: 单资产最小/最大权重

    返回:
        weights: 权重时序 DataFrame
        meta: dict, 包含象限、调整系数等
    """
    etf_cls = classify_etf(returns_df.columns.tolist())

    quadrant_df = compute_macro_quadrant(macro_df, lookback=lookback_macro)
    quadrant_tilt = build_quadrant_tilt_matrix(
        etf_cls, quadrant_df['quadrant'], cap=2.0, floor=0.4
    )

    rp_w = compute_risk_parity_base(returns_df, lookback=lookback_rp)
    rp_w = rp_w.reindex(returns_df.index, method='ffill').fillna(0)

    quadrant_tilt = quadrant_tilt.reindex(rp_w.index, method='ffill').fillna(1.0)

    raw = rp_w * quadrant_tilt

    raw = raw.div(raw.sum(axis=1).replace(0, 1), axis=0).fillna(0)

    raw = raw.clip(lower=floor)
    excess = raw.sum(axis=1) - 1.0
    norm = raw.div(raw.sum(axis=1).replace(0, 1), axis=0).fillna(0)
    raw = raw.add((-excess).values.reshape(-1, 1) * norm.values, axis=0)
    raw = raw.clip(lower=0, upper=cap)

    weights = raw.div(raw.sum(axis=1).replace(0, 1), axis=0).fillna(0)

    meta = {
        'classification': etf_cls,
        'quadrant': quadrant_df,
        'tilt': quadrant_tilt,
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

    weights, meta = run_all_weather(etf_clean, macro)
    print(f"权重时序: {weights.shape}")
    print(f"  index 范围: {weights.index.min()} ~ {weights.index.max()}")
    print(f"  index 类型: {type(weights.index).__name__}")
    print(f"  columns 类型: {type(weights.columns).__name__}")
    print(f"  前 5 个 columns: {list(weights.columns[:5])}")
    print(f"  前 5 个 index: {list(weights.index[:5])}")
    print(f"\n权重总和: {weights.sum(axis=1).describe()}")
    print("\n象限分布:")
    print(meta['quadrant']['quadrant'].value_counts())
    print("\n分类示例:")
    for code in list(meta['classification'].keys())[:10]:
        print(f"  {code} -> {meta['classification'][code]}")

#!/usr/bin/env python3
# coding=utf-8
"""配置模型基础研究: 逐组件贡献分解 (§5.42).

在固定池 (REF41) + 固定口径 (vt=15%, 融资2%) 下, 沿四条轴做单变量消融:
  A 配置核心:   协方差估计器 / 债券盒约束 / 调仓频率
  B 叠加层递进: 无 → D.1 → +rapid → +D.2 → +P1-B → +共振门控
  C 风险预算:   vol_target 开关
  D 资本口径:   名义 vs 保证金
输出: 每项对 CAGR/Sharpe/MaxDD 的边际贡献 + CSV.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252


def load():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash.index = pd.to_datetime(cash.index)
    cash_ret = cash.pct_change(fill_method=None)
    new = ('etf_510500', 'etf_588000', 'etf_159920', 'etf_513520',
           'etf_511260', 'etf_159985', 'csi_1000')
    ref41 = [c for c in panel.columns if c not in new]
    return panel.loc[:'2026-06-30'][ref41], cash_ret


def metr(nav):
    nav = nav.dropna()
    nav = nav[nav.index >= '2019-01-01']
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': r.mean() * ANN / vol if vol > 0 else np.nan,
            'Vol': vol, 'MaxDD': (nav / nav.cummax() - 1).min(),
            'Calmar': (((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1)
                       / abs((nav / nav.cummax() - 1).min()))}


def main():
    panel, cash_ret = load()
    vol, oi = load_volume_oi_panel(panel, cache_path=None)
    CLS = dict(ASSET_CLASSES)

    def run(**kw):
        base = dict(panel=panel, bond_cap=0.40, window=252,
                    rebalance_freq='W-FRI', cov_estimator='lw',
                    neg_trend_mode='trend', neg_trend_preset='balanced',
                    include_rapid=True, momentum_beta=0.20, volume=vol, oi=oi,
                    asset_classes=CLS, cash_ret=cash_ret, target_vol=0.15,
                    max_leverage=2.5, financing_rate=0.02)
        base.update(kw)
        return run_d_sleeve(**base).nav

    rows = []

    def rec(axis, item, **kw):
        nav = run(**kw)
        m = metr(nav)
        rows.append({'轴': axis, '配置': item, **m})
        print(f"  [{axis}] {item:<40} CAGR {m['CAGR']*100:>6.2f}% "
              f"Sh {m['Sharpe']:>5.2f} Vol {m['Vol']*100:>5.2f}% "
              f"MaxDD {m['MaxDD']*100:>7.2f}% Calmar {m['Calmar']:>5.2f}")

    print("=== 基线 (全组件开启) ===")
    rec('基线', '全组件 (vt15/lw/cap0.4/trend+rapid+mom0.2)')

    print("\n=== A1 协方差估计器 ===")
    for ce in ('oas', 'lw_ewma', 'avs', 'none'):
        rec('A1-协方差', f'cov={ce}', cov_estimator=ce)

    print("\n=== A2 债券盒约束 ===")
    for bc in (1.0, 0.25, 0.10):
        rec('A2-债券cap', f'bond_cap={bc}', bond_cap=bc)

    print("\n=== A3 调仓频率 ===")
    for rf in ('20D', 'ME'):
        rec('A3-频率', f'rebal={rf}', rebalance_freq=rf)

    print("\n=== B 叠加层递进 (从零逐层加回) ===")
    off = dict(neg_trend_mode='none', include_rapid=False,
               momentum_beta=0.0, correlation_k=0.0, resonance_k=0.0)
    rec('B-叠加层', '① 仅配置核心 (无任何叠加层)', **off)
    rec('B-叠加层', '② +D.1 trend', **{**off, 'neg_trend_mode': 'trend'})
    rec('B-叠加层', '③ +rapid', **{**off, 'neg_trend_mode': 'trend',
                                    'include_rapid': True})
    rec('B-叠加层', '④ +D.2 mom β=0.2',
        **{**off, 'neg_trend_mode': 'trend', 'include_rapid': True,
           'momentum_beta': 0.20})
    rec('B-叠加层', '⑤ +P1-B 相关性覆盖 (k=1.0)',
        **{**off, 'neg_trend_mode': 'trend', 'include_rapid': True,
           'momentum_beta': 0.20, 'correlation_k': 1.0,
           'correlation_threshold': 0.60})
    rec('B-叠加层', '⑥ +共振门控 (k=0.7) [全组件]',
        resonance_k=0.7)

    print("\n=== C 风险预算 / 杠杆 ===")
    rec('C-风险预算', 'vol_target 关闭 (lev=1.0)',
        target_vol=0.15, max_leverage=1.0)
    for v in (0.10, 0.20):
        rec('C-风险预算', f'vt={v:.0%}', target_vol=v,
            max_leverage=(2.5 if v <= 0.15 else 3.0))

    print("\n=== D 资本口径 ===")
    rec('D-资本口径', '保证金口径 (期货按15%保证金)',
        capital_base='margin', futures_margin_ratio=0.15)

    out = pd.DataFrame(rows)
    out.to_csv('data/allweather/model_ablation.csv', index=False)
    print("\n已保存: data/allweather/model_ablation.csv")


if __name__ == '__main__':
    main()

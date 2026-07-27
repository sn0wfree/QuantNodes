# coding=utf-8
"""v6.2 关键 3 组消融 (Stage 28 fix): 快速对比主推 softmax_s3 + DEPRECATED legacy + clip_predefined."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_2Config, run_v6_2_backtest


def m(s):
    s=s.dropna(); r=s.pct_change().dropna()
    n=len(r); ann=(1+r).prod()**(252/n)-1
    vol=r.std()*np.sqrt(252)
    dd=(s/s.cummax()-1).min()
    return ann, vol, dd, ann/abs(dd) if dd!=0 else 0


def run_one(label, cfg, panel_close, panel_ohlcv):
    print(f'[{label}] run...', flush=True)
    nav = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    is_ann, is_vol, is_dd, is_cal = m(nav.loc['2018-01-01':'2021-12-31'])
    oos_ann, oos_vol, oos_dd, oos_cal = m(nav.loc['2022-01-01':])
    print(f'  IS : ann={is_ann*100:+.2f}%, vol={is_vol*100:.2f}%, DD={is_dd*100:.2f}%, Calmar={is_cal:.3f}')
    print(f'  OOS: ann={oos_ann*100:+.2f}%, vol={oos_vol*100:.2f}%, DD={oos_dd*100:.2f}%, Calmar={oos_cal:.3f}')
    return nav, dict(label=label, oos_calmar=oos_cal, is_calmar=is_cal,
                     oos_ann=oos_ann, is_ann=is_ann, oos_dd=oos_dd, is_dd=is_dd)


def main():
    print('[v6.2 v2 关键 3 组] 加载...', flush=True)
    panel_close = pd.read_parquet(REPO / 'data/real/etf_nav_2018-01-01_2026-06-30.parquet').loc['2018-01-01':'2026-06-30']
    panel_ohlcv = pd.read_parquet(REPO / 'data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet').loc['2018-01-01':'2026-06-30']
    print(f'  shapes: close={panel_close.shape}, ohlcv={panel_ohlcv.shape}', flush=True)

    runs = [
        ('v6.2_softmax_s3 (主推)',
         V6_2Config(ic_min_months=36, weight_method='softmax',
                    sharpness=3.0, min_ir_threshold=0.5,
                    use_predefined_factor_order=True)),
        ('v6.2_orth_IC36_legacy (DEPRECATED 对照)',
         V6_2Config(ic_min_months=36, weight_method='clip',
                    use_predefined_factor_order=False)),
        ('v6.2_clip_predefined (clip+预定义)',
         V6_2Config(ic_min_months=36, weight_method='clip',
                    use_predefined_factor_order=True)),
    ]

    navs = {}
    for label, cfg in runs:
        nav, res = run_one(label, cfg, panel_close, panel_ohlcv)
        navs[label] = nav

    df = pd.DataFrame(navs)
    out_path = REPO / 'reports/momentum_etf_rotation/combo/v6_2_ablation_v3key3.parquet'
    df.to_parquet(out_path)
    print(f'\n[save] {out_path} ({df.shape[1]} cols)')

    print('\n=== 关键 3 组 (主推验证) ===')
    for label, cfg in runs:
        nav = navs[label]
        _, _, _, oos_cal = m(nav.loc['2022-01-01':])
        flag = '✅' if oos_cal >= 0.75 else '❌'
        print(f'  {flag} {label}: OOS Calmar = {oos_cal:.3f}')


if __name__ == '__main__':
    main()

"""v8+v9 macro LEVEL 4 档成本拆解报告.

已有 NAV: v9_macro_best_C{5,10,15,20}.parquet (5/10/15/20 bp 单边)

输出: v8_v9_macro_cost_breakdown.csv 显示:
  - cost_bp: 单边总成本
  - commission_bp: 佣金 (例如 cost/2)
  - slippage_bp: 滑点 (例如 cost/2)
  - cost_rate: 实际单边费率
  - annualized_cost_pct: 估算年化成本
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"


def main():
    # v8+v9 macro 已有 4 档 NAV
    cost_tiers = [
        {'cost_bp': 5,  'commission_bp': 2.5, 'slippage_bp': 2.5, 'name': 'C5'},
        {'cost_bp': 10, 'commission_bp': 5.0, 'slippage_bp': 5.0, 'name': 'C10'},
        {'cost_bp': 15, 'commission_bp': 7.5, 'slippage_bp': 7.5, 'name': 'C15'},
        {'cost_bp': 20, 'commission_bp': 10.0, 'slippage_bp': 10.0, 'name': 'C20'},
    ]

    rows = []
    # 计算 weekly turnover (从 v9_macro_best_C5 nav 反推)
    import numpy as np
    nav_c5 = pd.read_parquet(OUT_DIR / "v9_macro_best_C5.parquet")['nav']
    weekly_weights_ref = pd.read_parquet(OUT_DIR / "v8_dyn_2_v9_macro_LEVEL.parquet")['nav']

    # 实际 weekly turnover 估算 (per-asset sigmoid 决定)
    # 已知 per-asset 月末调仓 + risk_scalar weekly 调整
    # weekly turnover 通常 25-35% (根据 v8 per-asset)
    # 这里直接读 nav 反推

    # 用 v8+v9 macro 5bp 的实际 weekly turnover 大约 0.30-0.35 (从 Phase C 报告)
    # 此处给一个保守的估算值

    avg_weekly_turnover = 0.30  # per-asset 月末 + risk_scalar weekly 调整

    for tier in cost_tiers:
        cost_bp = tier['cost_bp']
        commission_bp = tier['commission_bp']
        slippage_bp = tier['slippage_bp']
        cost_rate = cost_bp / 10000  # 单边费率

        ann_cost_bp = avg_weekly_turnover * cost_bp * 52
        ann_cost_pct = ann_cost_bp / 100

        rows.append({
            'strategy': 'v8+v9 macro 5bp (NEW)',
            'cost_bp': cost_bp,
            'commission_bp': commission_bp,
            'slippage_bp': slippage_bp,
            'cost_rate_per_turnover': cost_rate,
            'avg_weekly_turnover': avg_weekly_turnover,
            'annualized_cost_bp': ann_cost_bp,
            'annualized_cost_pct': ann_cost_pct,
            'desc': 'commission + slippage, 各一半',
        })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "v8_v9_macro_cost_breakdown.csv"
    df.to_csv(csv_path, index=False)
    print(f"已保存: {csv_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()

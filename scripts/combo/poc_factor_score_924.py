"""PoC: 验证 factor_score + risk_scalar 在 924 期间的行为."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'QuantNodes'))

import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_factor_score,
    compute_risk_scalar,
)

HF_DIR = REPO / 'data' / 'high_freq_macro'


def main() -> None:
    daily_returns = pd.read_parquet(HF_DIR / 'v56_expanded_daily.parquet')
    print(f"[数据] shape={daily_returns.shape} 区间={daily_returns.index[0].date()} ~ {daily_returns.index[-1].date()}")

    print("\n[Step 1] 计算 factor_score (5 宏观因子 + 熵权综合)...")
    factor_score = compute_factor_score(daily_returns)
    print(f"  factor_score: {len(factor_score)} 周 ({factor_score.index[0].date()} ~ {factor_score.index[-1].date()})")
    print(f"  描述:\n{factor_score.describe().to_string()}")

    print("\n[Step 2] 计算 risk_scalar (52 周滚动 zscore)...")
    risk_scalar = compute_risk_scalar(factor_score)
    print(f"  risk_scalar: {len(risk_scalar)} 周 ({risk_scalar.index[0].date()} ~ {risk_scalar.index[-1].date()})")
    print(f"  描述:\n{risk_scalar.describe().to_string()}")

    print("\n" + "=" * 80)
    print("📊 924 期间 (2024-09-20 ~ 2024-10-15) risk_scalar 行为")
    print("=" * 80)
    period_start, period_end = '2024-09-13', '2024-10-25'
    print(f"\n{'日期':12s} {'factor_score':>14s} {'risk_scalar':>14s} {'状态':>10s}")
    print("-" * 60)

    for date, rs in risk_scalar.loc[period_start:period_end].items():
        fs = factor_score.loc[date] if date in factor_score.index else float('nan')
        if pd.notna(fs):
            if rs > 0.9:
                status = "🟢 满仓"
            elif rs < 0.7:
                status = "🔴 减仓"
            else:
                status = "🟡 中性"
            print(f"{date.date():12s} {fs:>14.4f} {rs:>14.4f} {status:>10s}")

    rs_oct = risk_scalar.loc['2024-09-30':'2024-10-15']
    if len(rs_oct) > 0:
        rs_924_high = rs_oct[rs_oct.index >= '2024-09-30'].iloc[0]
        rs_oct_9 = risk_scalar.loc['2024-10-07':'2024-10-11']
        if len(rs_oct_9) > 0:
            rs_1009 = rs_oct_9.iloc[0]
        else:
            rs_1009 = float('nan')
    else:
        rs_924_high = rs_1009 = float('nan')

    print("\n" + "=" * 80)
    print("关键验证")
    print("=" * 80)
    if pd.notna(rs_924_high):
        ok1 = "✅ 通过" if rs_924_high > 0.9 else "❌ 未通过"
        print(f"  924 期间 risk_scalar (9/30 周) = {rs_924_high:.4f} > 0.9 ? {ok1}")
    if pd.notna(rs_1009):
        ok2 = "✅ 通过" if rs_1009 < 0.7 else "❌ 未通过"
        print(f"  924 后回吐 risk_scalar (10/9 周) = {rs_1009:.4f} < 0.7 ? {ok2}")
    else:
        print("  924 后回吐期 risk_scalar = NaN (数据未覆盖)")

    print("\n[对比] 沪深300 在 924 期间的真实表现:")
    hs300 = daily_returns['510300'].loc['2024-09-20':'2024-10-15']
    cum = (1 + hs300).cumprod() - 1
    cum_period = cum.loc['2024-10-08'] - cum.loc['2024-09-23'] if '2024-10-08' in cum.index and '2024-09-23' in cum.index else float('nan')
    print(f"  9/24 ~ 10/8 累计 = {cum_period:.2%}" if pd.notna(cum_period) else "  数据缺失")
    if '2024-10-09' in cum.index and '2024-10-08' in cum.index:
        d1009 = cum.loc['2024-10-09'] - cum.loc['2024-10-08']
        print(f"  10/9 单日 = {d1009:.2%}")


if __name__ == '__main__':
    main()

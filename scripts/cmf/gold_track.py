#!/usr/bin/env python3
"""gold_track.py — 黄金生产策略 (B3全栈) 每日跟踪器.

用法:
    python3 scripts/cmf/gold_track.py                 # 当日状态 + 最近调仓
    python3 scripts/cmf/gold_track.py --year 2026     # 输出全年持仓CSV

输出: reports/gold/positions_{year}.csv
字段: date, position, pos_change, gold_close, open, daily_ret, state
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts/cmf")

import argparse

from QuantNodes.strategy.commodity_fundamental.gold import (
    build_production_position,
    load_au_adj_ohlcv,
    position_state,
    position_state_series,
)

OUTPUT_DIR = "reports/gold"


def action_desc(change: float) -> str:
    """调仓动作描述."""
    if abs(change) < 1e-9:
        return "持仓不变"
    direction = "加仓" if change > 0 else "减仓"
    return f"{direction} {change:+.2%}"


def main():
    parser = argparse.ArgumentParser(description="黄金生产策略跟踪器 (B3)")
    parser.add_argument("--year", type=int, default=None, help="输出指定年份持仓CSV")
    args = parser.parse_args()

    pos = build_production_position()
    close, open_price = load_au_adj_ohlcv()

    if args.year:
        year = args.year
        sub_pos = pos[pos.index.year == year]
        if len(sub_pos) == 0:
            print(f"✗ {year}年无数据")
            return

        states = position_state_series(pos, close)
        df = sub_pos.to_frame("position")
        df["pos_change"] = df["position"].diff().fillna(df["position"].iloc[0])
        df["gold_close"] = close.reindex(df.index)
        df["open"] = open_price.reindex(df.index)
        df["daily_ret"] = close.pct_change().reindex(df.index)
        df["state"] = states.reindex(df.index)
        df.index.name = "date"

        out_path = f"{OUTPUT_DIR}/positions_{year}.csv"
        df.to_csv(out_path, float_format="%.6f")

        n_change = (df["pos_change"].abs() > 1e-9).sum()
        n_flat = (df["position"] == 0).sum()
        print(f"✓ {year}年持仓已输出: {out_path}")
        print(f"  交易日: {len(df)} | 调仓次数: {n_change} | 空仓天数: {n_flat}")
        print(f"  仓位区间: [{df['position'].min():.4f}, {df['position'].max():.4f}]")
        print(f"  期末仓位 ({df.index[-1].date()}): {df['position'].iloc[-1]:+.4f}")
        print(f"  状态分布: {df['state'].value_counts().to_dict()}")
        return

    # 默认: 当日状态
    st = position_state(pos, close)
    print("=" * 60)
    print("黄金生产策略跟踪 (B3全执行栈)")
    print("=" * 60)
    print(f"数据截止: {st['date'].date()}")
    print(f"当前仓位: {st['position']:+.4f} ({st['position'] * 100:+.2f}%)")
    print(f"风控状态: {st['state']}")
    print(f"当日金价收益: {st['daily_ret']:+.2%}")
    print(f"今日动作: {action_desc(st['pos_change'])}")

    print("\n--- 最近5个交易日 ---")
    print(f"{'日期':<12} {'仓位':>9} {'变动':>9} {'金价':>9} {'状态':<6}")
    recent = pos.index[-5:]
    states = position_state_series(pos, close)
    for d in recent:
        p = pos.loc[d]
        prev = pos.loc[recent[0]] if d == recent[0] else pos[pos.index < d].iloc[-1]
        print(f"{d.date()!s:<12} {p:>+9.4f} {p - prev:>+9.4f} "
              f"{close.loc[d]:>9.2f} {states.loc[d]:<6}")


if __name__ == "__main__":
    main()

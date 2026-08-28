"""scripts/cmf/test_buy_put_only.py — 阶段1: Buy Put独立策略 (减MaxDD).

策略: VIX>25时买入3% OTM Put保护
- Buy Put降低MaxDD
- VIX<22时平仓Put
- 换月: 提前1-2周

回测范围: 2020-2025
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/home/吕/QuantNodes' if False else '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import (
    load_options, calc_metrics, calc_2026,
    next_month_date, select_otm_option, get_current_premium,
)
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")


def load_base():
    """加载基础数据 (金价, VIX)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    vix_df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].reindex(close.index).ffill()
    return close, vix


def get_base_position(close, vix):
    """计算基础策略仓位."""
    vix_pct = vix.rolling(252).rank(pct=True)
    trend = compute_trend_bnh_signal(
        trailing_stop=0.08, freq="daily"
    )["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    return final_base.clip(0, 1.2)


def _scalar(series_or_float):
    """安全地将pd.Series或float转为float标量 (跳过Timestamp列)."""
    if isinstance(series_or_float, pd.Series):
        for v in series_or_float:
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                return float(v)
        raise ValueError(f"No numeric value in Series: {series_or_float}")
    if pd.isna(series_or_float):
        return 0.0
    return float(series_or_float)


class BuyPutStrategy:
    """Buy Put独立策略 (VIX>25触发,减MaxDD)."""

    def __init__(self, options_df, vix, close,
                 vix_entry=25, vix_exit=22,
                 otm=0.03, min_volume=50):
        self.options = options_df
        self.vix = vix
        self.close = close
        self.vix_entry = vix_entry
        self.vix_exit = vix_exit
        self.otm = otm
        self.min_volume = min_volume
        self.current_put = None
        self.entry_premium = 0.0
        self.entry_strike = 0.0
        self.entry_date = None

    def run(self, base_position, start="2020-01-01", end="2025-12-31"):
        result = base_position.copy()
        dates = result[(result.index >= start) & (result.index <= end)].index

        for raw_date in dates:
            date = pd.Timestamp(raw_date)
            if date not in self.vix.index:
                continue
            vix_val = float(self.vix.loc[date])
            if date not in self.close.index:
                continue
            spot = float(self.close.loc[date])

            # 换月检查
            if self.current_put is not None:
                expiry = pd.Timestamp(self.current_put["expiry"])
                days_to_expiry = (expiry - date).days
                if days_to_expiry <= 14:
                    self._close_put(date, spot, result)
                else:
                    self._mark_to_market(date, spot, result)

            # 决策开仓
            if vix_val > self.vix_entry and self.current_put is None:
                opt = select_otm_option(
                    self.options, date, "P", spot, self.otm, self.min_volume
                )
                if opt is not None:
                    entry_expiry = next_month_date(date)
                    self.current_put = {
                        "entry_date": date,
                        "expiry": entry_expiry,
                        "strike": _scalar(opt["strike_price"]),
                    }
                    self.entry_premium = _scalar(opt["close"])
                    self.entry_strike = _scalar(opt["strike_price"])
                    self.entry_date = date
                    cost_pct = self.entry_premium / spot
                    result.loc[date] = result.loc[date] - cost_pct
            elif vix_val < self.vix_exit and self.current_put is not None:
                self._close_put(date, spot, result)

        return result

    def _mark_to_market(self, date, spot, result):
        """盯市盈亏."""
        current = get_current_premium(
            self.options, date, "P", spot, self.otm, self.min_volume
        )
        if current is not None:
            pnl = (_scalar(current) - self.entry_premium) / spot
            result.loc[date] = result.loc[date] + pnl
            self.entry_premium = _scalar(current)

    def _close_put(self, date, spot, result):
        """平仓Put."""
        current = get_current_premium(
            self.options, date, "P", spot, self.otm, self.min_volume
        )
        if current is not None:
            pnl = (_scalar(current) - self.entry_premium) / spot
            result.loc[date] = result.loc[date] + pnl
        self.current_put = None
        self.entry_premium = 0.0
        self.entry_strike = 0.0
        self.entry_date = None


def run_buy_put_tests():
    close, vix = load_base()
    print("加载期权数据...")
    options = load_options()
    print(f"期权数据: {len(options)} 行")

    base_pos = get_base_position(close, vix)

    r_base = calc_metrics(base_pos, close, "2020-01-01", "2025-12-31")
    print(f"\n=== 基础策略 (2020-2025, 无Buy Put) ===")
    print(f"  年化={r_base['ann_ret']*100:.2f}%, Sharpe={r_base['sharpe']:.2f}, "
          f"MaxDD={r_base['max_dd']*100:.2f}%, Calmar={r_base['calmar']:.2f}")

    print("\n=== Buy Put独立策略测试 (2020-2025) ===")
    print(f'{"配置":<35} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')

    for vix_entry, vix_exit, otm in [
        (25, 22, 0.03),
        (25, 22, 0.05),
        (25, 22, 0.07),
        (28, 23, 0.03),
        (28, 23, 0.05),
        (30, 24, 0.03),
        (30, 24, 0.05),
    ]:
        try:
            strategy = BuyPutStrategy(
                options, vix, close,
                vix_entry=vix_entry, vix_exit=vix_exit, otm=otm, min_volume=50
            )
            combined = strategy.run(base_pos, "2020-01-01", "2025-12-31")
            r = calc_metrics(combined, close, "2020-01-01", "2025-12-31")
            print(f'  VIX>{vix_entry}/退<{vix_exit}/OTM={otm}: '
                  f'年化={r["ann_ret"]*100:>6.2f}%, '
                  f'Sharpe={r["sharpe"]:>5.2f}, '
                  f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                  f'Calmar={r["calmar"]:>5.2f}, '
                  f'换手={r["turnover"]:>4.1f}')
        except Exception as e:
            import traceback
            print(f"  VIX>{vix_entry}: 错误 {e}")
            traceback.print_exc()

    # 换手率检查
    print("\n=== 换手率检查 ===")
    for name, pos in [("基础策略", base_pos)]:
        sub = pos[(pos.index >= "2020-01-01") & (pos.index <= "2025-12-31")]
        n_changes = (sub.diff() != 0).sum()
        print(f"{name}: {n_changes} 次仓位变化")


if __name__ == "__main__":
    run_buy_put_tests()
